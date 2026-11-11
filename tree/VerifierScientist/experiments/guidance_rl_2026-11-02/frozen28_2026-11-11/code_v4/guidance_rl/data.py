"""Read-only legacy adapters and new, isolated data artifacts."""
from __future__ import annotations
import json,os
from pathlib import Path
from collections import defaultdict
from .core import atomic_json,digest,split_tasks,evaluate

ROOT=Path(os.environ.get('GUIDANCE_PROJECT_ROOT',str(Path(__file__).resolve().parents[2])))


def common():
    from verifier_bench import common as c
    return c


def enrich(record,labels=False):
    c=common();r=dict(record);task,catalog=c.load_task(r['task_path'])
    r['visible_task']=c.task_view_text(r['task_path']);r.setdefault('domain',task.domain)
    r['resource_tools']=sorted(t.name for t in catalog.tools if t.is_resource)
    r['catalog_hash']=digest([t.model_dump(mode='json') for t in catalog.tools])
    if labels:r['annotated_rules']=c.task_rules_text(r['task_path'])
    return r


def load_exact_payload(directory,task_id,request_id):
    if any(x in task_id+request_id for x in ('/','\\','..')):raise ValueError('Invalid episode identifier')
    p=Path(directory)/f'{task_id}__{request_id}.json'
    if not p.exists():p=p.with_name(p.stem+'.used.json')
    obj=json.loads(p.read_text())
    if obj['task_id']!=task_id or obj['request_id']!=request_id:raise ValueError('Episode identity mismatch')
    return obj


def append_final(payload,text):
    actions=list(payload['actions'])
    if text:
        last=actions[-1] if actions else None;role=payload['hub_role']
        actions.append({'timestamp':last['timestamp'] if last else '2026-01-01T00:00:00Z',
            'sequence_num':last['sequence_num']+1 if last else 1,'run_id':payload['run_id'],
            'surface':'communication','agent_id':role,'agent_role':role,'target_agent':'user','target_role':'user',
            'content':text,'raw_event':{'framework':'oai','communication_kind':'user_output'}})
    return actions


def load_calibration(directory,split,n=8):
    allowed=set(split['propose']+split['select']+split['confirm']);by=defaultdict(dict)
    for p in sorted(Path(directory).glob('*.json')):
        obj=json.loads(p.read_text());r=obj['record']
        if r['task_id'] not in allowed:raise ValueError('Non Train-R record in calibration')
        by[r['task_id']][r['traj_id']]=r
    if set(by)!=allowed or any(len(v)!=n for v in by.values()):raise ValueError(f'Calibration needs exactly {n} trajectories for every Train-R task')
    return [enrich(r,labels=r['task_id'] in split['propose']) for t in sorted(by) for r in by[t].values()]


def prepare(out,per_task=4):
    import pandas as pd
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((ROOT/'HarnessAudit-main/splits/trustroute_split_v1.json').read_text())['tasks']
    seed=json.loads((ROOT/'ha_runs/verifier_bench/splits.json').read_text())['seed']
    split=split_tasks(meta,seed);atomic_json(out/'split.json',split)
    frames={}
    for name in ('train_all','test_all'):
        df=pd.read_parquet(ROOT/f'ha_runs/data/{name}_v08/{name}.parquet').copy()
        df['agent_name']='guidance_tool_agent'
        for info in df.extra_info:
            if not Path(info['task_yaml']).is_file():raise FileNotFoundError(info['task_yaml'])
        frames[name]=df;df.to_parquet(out/(name+'.parquet'),index=False)
    rids=set(split['propose']+split['select']+split['confirm'])
    train=frames['train_all'];cal=train[train.extra_info.map(lambda i:i['task_id'] in rids)].copy()
    if len(cal)!=60:raise ValueError('Expected all 60 labeled tasks in training parquet')
    cal.to_parquet(out/'calibration.parquet',index=False)
    by=defaultdict(list)
    for line in (ROOT/'ha_runs/verifier_bench/pool.jsonl').open():
        r=json.loads(line)
        if r['task_id'] in rids:by[r['task_id']].append(r)
    records=[]
    for tid in sorted(rids):
        # Deterministic sampling independent of labels or observed errors.
        rows=sorted(by[tid],key=lambda r:digest(r['traj_id']))[:per_task]
        if len(rows)<per_task:raise ValueError('Insufficient historical calibration for '+tid)
        records.extend(enrich(r,labels=tid in split['propose']) for r in rows)
    atomic_json(out/'initial_records.json',records)
    (out/'agent_loop.yaml').write_text('- name: guidance_tool_agent\n  _target_: guidance_rl.agent_loop.GuidanceToolAgentLoop\n')
    atomic_json(out/'manifest.json',{'tasks':{k:len(v) for k,v in split.items()},'train_rows':len(train),'test_rows':len(frames['test_all']),
        'historical_trajectories':len(records),'note':'Initial pool mixes historical policy versions. Fresh current-policy calibration replaces this after each segment.'})
    return out


def diagnose(out):
    rdir=ROOT/'ha_runs/verifier_bench/runs/rq1_luna_v1'
    sets=json.loads((rdir/'eval_sets.json').read_text());needed={t for v in sets.values() for t in v}
    pool={}
    for line in (ROOT/'ha_runs/verifier_bench/pool.jsonl').open():
        r=json.loads(line)
        if r['traj_id'] in needed:pool[r['traj_id']]=enrich(r)
    result={'status':'observational_not_causal','scope_check':'All three historical variants include the same scope check', 'sets':{}}
    for s,ids in sets.items():
        rows=[pool[t] for t in ids];result['sets'][s]={}
        for name in ('judge','rubric_bank','scientist_v0'):
            preds=json.loads((rdir/f'preds_{name}+scope.json').read_text())
            result['sets'][s][name]=[evaluate(rows,p) for _,p in sorted(preds.items())]
    guidance=json.loads((rdir/'guidance.json').read_text());result['guidance_chars']={k:len(v) for k,v in guidance.items()}
    result['confounds']=['wording/length','rule coverage','structured fields','exceptions','evidence instructions','different generation prompts']
    atomic_json(out,result);return result
