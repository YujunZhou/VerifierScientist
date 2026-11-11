import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock,patch
from pydantic import ValidationError
from guidance_rl.optimize import (selection_summary,_feedback,optimize,judge_record,
    ALIGNED_JUDGE_PROMPT,JUDGE_PROMPT,AlignedVerdict,Library,Equivalence)
from guidance_rl.core import make_artifact
from guidance_rl.client import BudgetExceeded


class ReviewOptimizerTests(unittest.TestCase):
    def test_summary_does_not_mix_counts_and_worst_recall(self):
        old={'pair_acc':.7,'macro_f1':.6,'mae':.1,'clean_false_alarm':None,'n_pairs':10,
             'n_tasks':3,'n_trajectories':6,'n_clean':0,
             'per_code':{'V-ID':{'tp':11,'fp':0,'fn':0,'gold':11,'recall':1.}}}
        new={**old,'pair_acc':.8,'per_code':{'V-ID':{'tp':10,'fp':0,'fn':1,'gold':11,'recall':10/11}}}
        result=selection_summary([old,new])
        self.assertEqual(result['per_code']['V-ID']['recall'],10/11)
        self.assertNotIn('tp',result['per_code']['V-ID'])
        self.assertEqual(old['per_code']['V-ID']['recall'],1.)

    def test_feedback_includes_context_and_identifies_focus_actions(self):
        record={'task_id':'t','traj_id':'a','visible_task':'goal','domain':'d',
                'actions':[{'sequence_num':i,'content':str(i)} for i in range(1,8)],
                'gold':[{'seq':4,'code':'V-ID'}]}
        result=_feedback([record],{'a':[]})[0]
        self.assertEqual(result['focus_sequences'],[4])
        self.assertEqual([a['sequence_num'] for a in result['actions']],[2,3,4,5,6])

    def test_aligned_judge_requires_severity_and_separates_stage_nonce(self):
        with self.assertRaises(ValidationError):
            AlignedVerdict(violations=[{'seq':1,'code':'V-OT','reason':'forbidden'}])
        client=Mock();client.call.return_value=AlignedVerdict(violations=[{'seq':1,'code':'V-OT','severity':'low','reason':'unnecessary'}])
        record={'traj_id':'t','visible_task':'task','actions':[{'sequence_num':1}],
                'gold':[{'private':'secret'}],'annotated_rules':'hidden'}
        artifact=make_artifact(guidance='',judge_model='m',judge_prompt=ALIGNED_JUDGE_PROMPT,
                               policy='base',score_protocol='aligned_v2')
        pred=judge_record(client,record,artifact,2,stage='round-1:select')
        self.assertEqual(pred[0]['severity'],'low')
        self.assertIs(client.call.call_args.args[2],AlignedVerdict)
        self.assertEqual(client.call.call_args.kwargs['salt'],'aligned_v2:round-1:select:rep-2')
        self.assertNotIn('secret',client.call.call_args.args[1]);self.assertNotIn('hidden',client.call.call_args.args[1])
        bad=make_artifact(guidance='',judge_model='m',judge_prompt=JUDGE_PROMPT,
                         policy='base',score_protocol='aligned_v2')
        with self.assertRaises(ValueError):judge_record(client,record,bad)


class AlignedFakeAPI:
    config={'concurrency':1}
    def __init__(self,learned_helps):self.requests=[];self.learned_helps=learned_helps
    def call(self,instructions,user,schema,**kwargs):
        self.requests.append((instructions,user,kwargs))
        if schema is AlignedVerdict:
            d=json.loads(user);violations=[]
            if d['guidance'] and self.learned_helps:
                violations=[{'seq':a['sequence_num'],'code':'V-OT','severity':'high','reason':'observed forbidden tool'}
                            for a in d['actions'] if a.get('tool_name')=='forbidden']
            return AlignedVerdict(violations=violations)
        if schema is Equivalence:return Equivalence(equivalent=True,changed_rule_ids=[],explanation='local test')
        return Library(rules=[{'rule_id':'R1','text':'Do not call forbidden tools.'}])


class AlignedOptimizerReplay(unittest.TestCase):
    def inputs(self):
        split={'propose':['p'],'select':['s'],'confirm':['c'],'train_u':['u'],'test':['t']}
        rows=[]
        for tid in ('p','s','c'):
            for i in (0,1):
                rows.append({'task_id':tid,'traj_id':tid+str(i),'domain':'d','visible_task':tid,
                    'resource_tools':[],'actions':[{'sequence_num':1,'surface':'tool_call','tool_name':'forbidden' if i else 'safe'}],
                    'gold':[{'seq':1,'code':'V-OT','severity':'high'}] if i else [],
                    **({'annotated_rules':'PROPOSE_LABELS_ONLY'} if tid=='p' else {})})
        config={'generator_model':'m','judge_model':'m','score_protocol':'aligned_v2','baseline_start':True,
                'judge_replicates':3,'bootstrap_samples':20,'min_confirm_pairs':1}
        return rows,split,config

    def test_no_improvement_keeps_named_empty_baseline(self):
        rows,split,cfg=self.inputs();api=AlignedFakeAPI(False)
        with TemporaryDirectory() as d:
            a=optimize(rows,split,cfg,api,d,0,'base')
            self.assertEqual(a['guidance'],'');self.assertEqual(a['score_protocol'],'aligned_v2')
            self.assertEqual(a['provenance']['source'],'no_guidance_baseline')
            self.assertFalse(a['provenance']['gate']['accepted'])
            selected=json.loads((Path(d)/'select_proposal_seed.json').read_text())
            self.assertEqual(len(selected['metrics']),3)
            self.assertFalse(list(Path(d).glob('confirm_rep*.json')))
            self.assertFalse(any('PROPOSE_LABELS_ONLY' in u for _,u,k in api.requests if k.get('tag','').startswith('judge:')))

    def test_promotion_requires_both_references_in_every_replicate(self):
        rows,split,cfg=self.inputs();api=AlignedFakeAPI(True)
        with TemporaryDirectory() as d:
            a=optimize(rows,split,cfg,api,d,0,'base')
            self.assertTrue(a['guidance']);self.assertTrue(a['provenance']['gate']['accepted'])
            confirms=sorted(Path(d).glob('confirm_rep*.json'));self.assertEqual(len(confirms),3)
            for path in confirms:
                result=json.loads(path.read_text())['decision']
                self.assertTrue(result['against_incumbent']['accepted']);self.assertTrue(result['against_no_guidance']['accepted'])
            for _,user,k in api.requests:
                if k.get('tag')=='candidate:error_repair':
                    self.assertTrue(all(e['task_id']=='p' for e in json.loads(user)['proposal_errors']))

    def test_failed_second_reference_retains_baseline(self):
        rows,split,cfg=self.inputs();api=AlignedFakeAPI(True)
        outcomes=[{'accepted':True},{'accepted':False}]*3
        with TemporaryDirectory() as d,patch('guidance_rl.optimize.gate',side_effect=outcomes) as mocked:
            a=optimize(rows,split,cfg,api,d,0,'base')
            self.assertEqual(mocked.call_count,6);self.assertEqual(a['guidance'],'')
            self.assertFalse(a['provenance']['gate']['accepted'])

    def test_insufficient_repeats_rejected_before_api(self):
        rows,split,cfg=self.inputs();cfg['judge_replicates']=2;api=AlignedFakeAPI(True)
        with TemporaryDirectory() as d:
            with self.assertRaises(ValueError):optimize(rows,split,cfg,api,d,0,'base')
        self.assertEqual(api.requests,[])

    def test_candidate_names_limits_paid_generation_but_keeps_baselines(self):
        rows,split,cfg=self.inputs();cfg['candidate_names']=['proposal_seed','error_repair'];api=AlignedFakeAPI(True)
        with TemporaryDirectory() as d:
            optimize(rows,split,cfg,api,d,0,'base')
            names={p.stem.removeprefix('select_') for p in Path(d).glob('select_*.json')}
            self.assertEqual(names,{'incumbent','no_guidance','proposal_seed','error_repair'})
            self.assertEqual([k['tag'] for _,_,k in api.requests if k['tag'].startswith('candidate:')],['candidate:error_repair'])

    def test_budget_failure_preserves_incumbent_without_success_artifact(self):
        rows,split,cfg=self.inputs();api=Mock();api.call.side_effect=BudgetExceeded('test cap')
        with TemporaryDirectory() as d:
            with self.assertRaises(BudgetExceeded):optimize(rows,split,cfg,api,d,0,'base')
            out=Path(d);failure=json.loads((out/'failure.json').read_text())
            kept=json.loads((out/'retained_incumbent.json').read_text())
            self.assertEqual(failure['status'],'budget_exhausted');self.assertFalse(failure['optimization_completed'])
            self.assertEqual(kept['guidance'],'');self.assertEqual(kept['score_protocol'],'aligned_v2')
            self.assertFalse((out/'artifact.json').exists());self.assertFalse((out/'result.json').exists())

    def test_requested_seed_is_refreshed_after_nonempty_incumbent(self):
        rows,split,cfg=self.inputs();cfg['candidate_names']=['proposal_seed','error_repair'];api=AlignedFakeAPI(True)
        old=make_artifact(guidance='Existing accepted rule',judge_model='m',judge_prompt=ALIGNED_JUDGE_PROMPT,
             policy='previous',score_protocol='aligned_v2',provenance={'rules':[{'rule_id':'OLD','text':'Existing accepted rule'}]})
        with TemporaryDirectory() as d:
            optimize(rows,split,cfg,api,d,1,'new-router',incumbent=old)
            self.assertTrue((Path(d)/'select_proposal_seed.json').exists())
            self.assertEqual(sum(k['tag']=='seed' for _,_,k in api.requests),1)
