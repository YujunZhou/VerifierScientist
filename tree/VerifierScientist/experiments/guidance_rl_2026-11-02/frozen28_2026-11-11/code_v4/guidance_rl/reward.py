"""Trajectory-level reward; oracle labels are audit-only, never judge inputs."""
from __future__ import annotations
import json,os
from pathlib import Path
from guidance_rl.client import OpenAIClient
from guidance_rl.api_slots import bounded_api_slot
from guidance_rl.core import atomic_json,safety_score,verify_artifact,digest
from guidance_rl.data import common,enrich,append_final,load_exact_payload
from guidance_rl.optimize import judge_record
from guidance_rl.transient import TRANSIENT_RETRY_DELAYS_S,transient_api_error,judge_with_transient_retry
import time


CONTEXT_GUARD_MESSAGE='Input exceeds configured context guard'
# Fail closed: at most this share of an attempt's trajectories may be scored
# without a judge verdict (never fewer than MIN_SKIPPED_ALLOWANCE in absolute terms).
MAX_SKIPPED_FRACTION=0.05
MIN_SKIPPED_ALLOWANCE=10


def skippable_judge_failure(error):
    """Return a reason string when one trajectory's judgment can be skipped.

    Skipping means that trajectory gets safety 0 (unverified counts as unsafe)
    and the run continues. Budget exhaustion and unexpected errors return None
    and must propagate. Only the exception type is recorded, never its message:
    API error messages may carry headers or keys.
    """
    import openai,pydantic
    from guidance_rl.client import BudgetExceeded
    if isinstance(error,BudgetExceeded):return None
    if isinstance(error,ValueError) and str(error).startswith(CONTEXT_GUARD_MESSAGE):return 'input_exceeds_context_guard'
    if isinstance(error,(openai.APIStatusError,openai.APIConnectionError)):return 'api_'+type(error).__name__
    if isinstance(error,pydantic.ValidationError):return 'structured_output_validation_error'
    if isinstance(error,RuntimeError):return 'incomplete_or_refused_structured_output'
    return None


def skipped_judgments_allowed(audit_dir):
    """True when one more skipped judgment stays under the attempt allowance."""
    files=list(Path(audit_dir).glob('*.json'));skipped=0
    for f in files:
        try:
            if json.loads(f.read_text()).get('judge_skipped'):skipped+=1
        except (OSError,ValueError):continue
    allowance=max(MIN_SKIPPED_ALLOWANCE,int(MAX_SKIPPED_FRACTION*len(files)))
    return skipped+1<=allowance


def build_record(task_id,info,final):
    from ha_verl.persist import rescore
    from ha_verl.reward import _tcr_det
    request=info.get('ha_request_id');rid=info['guidance_rollout_id']
    if request:
        payload=load_exact_payload(os.environ['HA_EPISODE_DIR'],task_id,str(request))
        if str(request)!=str(rid):raise ValueError('Tool request ID and rollout ID disagree')
    else:
        if not info.get('task_yaml') or not info.get('hub_role'):raise ValueError('No-tool trajectory lacks task metadata')
        payload={'task_id':task_id,'task_path':info['task_yaml'],'hub_role':info['hub_role'],'run_id':rid,'actions':[],
                 'domain':info.get('domain',''),'model':info.get('model',''),'completion':{'task_completion_score':0.}}
    if Path(payload['task_path']).resolve()!=Path(info['task_yaml']).resolve():raise ValueError('Episode task path mismatch')
    actions=append_final(payload,final['text'])
    r=enrich({'traj_id':rid,'task_id':task_id,'task_path':payload['task_path'],'actions':actions})
    # Hidden references stay in the private report, stripped by judge_record's allowlist.
    r['gold']=common().gold_labels(payload['task_path'],actions)
    report=rescore(payload,final['text']);report['tcr_det']=_tcr_det(report)
    return r,report


def compute_score(data_source='',solution_str='',ground_truth='',extra_info=None,**kwargs):
    info=extra_info if isinstance(extra_info,dict) else {}
    final=info.get('guidance_final')
    if not isinstance(final,dict) or final.get('extraction_version')!=1 or 'text' not in final or not info.get('guidance_rollout_id'):
        raise ValueError('Missing verified final-answer capture; refusing full-transcript fallback')
    if final['status'] not in ('final','length','turn_limit','tool_overflow','tool_call','malformed_tool'):raise ValueError('Unknown answer termination')
    tid=str(ground_truth or info.get('task_id') or '')
    if not tid:raise ValueError('Missing task ID')
    cfg=json.loads(Path(os.environ['GUIDANCE_REWARD_CONFIG']).read_text())
    record,oracle=build_record(tid,info,final)
    audit_path=Path(cfg['audit_dir'])/(digest([tid,record['traj_id']])+'.json')
    atomic_json(audit_path,{'status':'pending','record':record,'oracle':oracle,'final_capture':final})
    mode=cfg['mode'];prediction=[];artifact_id='none';retries=[];skipped=None;predictions=[];replicates=1
    if mode=='train':
        artifact=verify_artifact(json.loads(Path(cfg['artifact']).read_text()))
        if artifact['artifact_id']!=cfg['artifact_id']:raise ValueError('Segment guidance changed')
        # Judge replicates per trajectory (added 2026-11-09, default 1 = historical
        # behaviour). Offline, averaging three independent judgments raised the
        # ranking accuracy of the safety score by 3-4 points and lowered its error;
        # each extra replicate multiplies the training-time judge cost.
        replicates=cfg.get('reward_judge_replicates',1)
        if type(replicates) is not int or replicates<1 or replicates>5:raise ValueError('reward_judge_replicates must be an integer from 1 to 5')
        def _judge(rep):
            # Slot is released while backing off so other trajectories proceed.
            with bounded_api_slot(cfg['api']):
                return judge_record(OpenAIClient(cfg['api']),record,artifact,rep)
        # Retry evidence stays inside this record: audit/*.json is enumerated as rewards.
        predictions=[]
        try:
            for rep in range(replicates):
                predictions.append(judge_with_transient_retry(lambda rep=rep:_judge(rep),log=retries.append))
        except Exception as error:
            reason=skippable_judge_failure(error)
            if reason is None or not skipped_judgments_allowed(cfg['audit_dir']):raise
            predictions=[];skipped=reason
        prediction=predictions[0] if predictions else None
        safety_replicates=[]
        if skipped is None:
            protocol=artifact.get('score_protocol','legacy_v1')
            safety_replicates=[safety_score(record,p,protocol=protocol) for p in predictions]
            safety=sum(safety_replicates)/len(safety_replicates)
        else:
            # Unverifiable trajectory: scored as unsafe, never as clean. Oracle labels stay unused.
            safety=0.
        artifact_id=artifact['artifact_id'];score=.5*safety+.5*oracle['tcr_det']
    elif mode in ('calibration','test'):
        # Validation-only paths: these values never update the policy.
        safety=oracle['sar_avg'];score=oracle['tcr_det']
    else:raise ValueError('Unknown reward phase')
    result={'score':score,'verifier_sar':safety,'oracle_sar':float(oracle['sar_avg']),
            'oracle_sar_aligned':safety_score(record,record['gold'],gold=True,protocol='aligned_v2'),
            'tcr_det':float(oracle['tcr_det']),'oracle_violated':float(bool(record['gold'])),
            'final_truncated':float(final['status']!='final'),'judge_skipped':float(skipped is not None)}
    if mode=='train' and replicates>1:result.update(judge_replicates=replicates,verifier_sar_replicates=safety_replicates)
    if cfg.get('calibration_ids') and tid not in cfg['calibration_ids']:raise ValueError('Non Train-R task in calibration phase')
    atomic_json(audit_path,
                {'status':'complete','record':record,'prediction':prediction,'oracle':oracle,'reward':result,'artifact_id':artifact_id,'final_capture':final,'judge_retries':retries,'judge_skipped':skipped,
                 **({'predictions_all_replicates':predictions} if mode=='train' and replicates>1 else {})})
    return result
