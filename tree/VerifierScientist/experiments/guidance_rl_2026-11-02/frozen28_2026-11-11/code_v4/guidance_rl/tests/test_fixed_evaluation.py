import importlib.util,json,time,hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from guidance_rl.core import make_artifact,atomic_json,digest
from guidance_rl.optimize import ALIGNED_JUDGE_PROMPT
from guidance_rl.client import OpenAIClient,BudgetExceeded

EXISTS=importlib.util.find_spec('guidance_rl.fixed_evaluation') is not None
if EXISTS:
 from guidance_rl.fixed_evaluation import build_schedule,run_evaluation,report_metrics

class FakeClient(OpenAIClient):
 def __init__(self,config):super().__init__(config);self.calls=[]
 def call(self,instructions,user,schema,*,model,tag='',salt='',max_output_tokens=None):
  request=self.request_spec(instructions,user,schema,model=model,salt=salt,max_output_tokens=max_output_tokens)
  key=digest(request);p=self.root/key[:2]/(key+'.json')
  if not p.exists():
   token=self.budget.reserve(.05);self.budget.settle(token,self.config.get("test_unit_cost",.001));self.calls.append((tag,salt))
   atomic_json(p,{'request':request,'parsed':{'violations':[]},'usd_upper':self.config.get('test_unit_cost',.001),'time':time.time()})
  self.last_receipt={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
  return schema.model_validate(json.loads(p.read_text())['parsed'])

class FixedEvaluationTests(unittest.TestCase):
 def test_entry_implemented(self):self.assertTrue(EXISTS,'fixed artifact evaluation entry is missing')
 def case(self,d):
  p=Path(d);split={k:[] for k in ('propose','select','confirm','train_u','test')};records=[]
  for i in range(60):
   tid=f't{i:02}';split['propose' if i<30 else 'select' if i<40 else 'confirm'].append(tid)
   for j in range(8):records.append({'task_id':tid,'traj_id':f'{tid}-{j}','visible_task':'visible',
      'actions':[{'sequence_num':1,'surface':'tool_call','tool_name':'x'}], 'resource_tools':[],
      'gold':[] if j%2 else [{'seq':1,'code':'V-OT','severity':'low'}]})
  artifact=make_artifact(guidance='',judge_model='judge',judge_prompt=ALIGNED_JUDGE_PROMPT,policy='fixed',score_protocol='aligned_v2')
  for name,obj in [('records',records),('split',split),('artifact',artifact)]:atomic_json(p/(name+'.json'),obj)
  atomic_json(p/'budget.json',{'spent':0.,'calls':0,'reserved':{}})
  cost={'schema_version':1,'observed_at':time.time(),'sample_count':30,'unit_cost_p95':.001,'models':['judge'],'score_protocol':'legacy_v1','provenance':'fixture'}
  atomic_json(p/'cost.json',cost)
  cfg={'records':str(p/'records.json'),'split':str(p/'split.json'),'artifact':str(p/'artifact.json'), 'output_dir':str(p/'out'),
       'mode':'pilot','phase_cap_usd':.25,'bootstrap_samples':20,
       'api':{'cache_dir':str(p/'cache'),'budget_file':str(p/'budget.json'),'max_usd':10,'max_calls':20000,'max_transport_retries':0},
       'planned_stage_budget':{'cost_audit':str(p/'cost.json'),'cost_audit_sha256':hashlib.sha256((p/'cost.json').read_bytes()).hexdigest(),
        'allow_protocol_proxy':True,'stages':{'E1_pilot':{'planned_calls':30,'margin_usd':.01,'reserve_after_usd':0}}}}
  return cfg,records,split,artifact
 @unittest.skipUnless(EXISTS,'pending')
 def test_schedule_is_complete_independent_of_labels_and_order(self):
  with TemporaryDirectory() as d:
   c,r,s,a=self.case(d);x=build_schedule(r,s,a);self.assertEqual(len(x),720)
   for row in r:row['gold']=[]
   self.assertEqual(x,build_schedule(list(reversed(r)),s,a));self.assertEqual(len({(v['traj_id'],v['rep']) for v in x}),720)
   with self.assertRaises(ValueError):build_schedule(r[:-1],s,a)
 @unittest.skipUnless(EXISTS,'pending')
 def test_pilot_stops_and_full_reuses_exact_prefix(self):
  with TemporaryDirectory() as d:
   c,r,s,a=self.case(d);client=FakeClient(c['api']);out=run_evaluation(c,client)
   self.assertEqual(len(client.calls),30);self.assertFalse(out['evaluation_complete']);self.assertFalse((Path(c['output_dir'])/'summary.json').exists())
   audit=Path(c['output_dir'])/'pilot_cost_audit.json';self.assertEqual(json.loads(audit.read_text())['sample_count'],30)
   self.assertEqual(run_evaluation(c,client),out);self.assertEqual(len(client.calls),30)
   c['mode']='full';c['phase_cap_usd']=2.;c['planned_stage_budget']={'cost_audit':str(audit),'cost_audit_sha256':hashlib.sha256(audit.read_bytes()).hexdigest(),'allow_protocol_proxy':False,'stages':{'E1_remaining':{'planned_calls':690,'margin_usd':.01,'reserve_after_usd':0}}}
   result=run_evaluation(c,client);self.assertTrue(result['evaluation_complete']);self.assertEqual(len(client.calls),720)
   self.assertEqual(len(result['replicates']),3);self.assertEqual(result['replicates'][0]['n_trajectories'],240)
   self.assertEqual(len({salt for tag,salt in client.calls}),720)
 @unittest.skipUnless(EXISTS,'pending')
 def test_budget_blocks_before_calls_and_phase_cap_is_hard(self):
  with TemporaryDirectory() as d:
   c,*_=self.case(d);c['api']['max_usd']=.01;cl=FakeClient(c['api'])
   with self.assertRaises(ValueError):run_evaluation(c,cl)
   self.assertEqual(cl.calls,[])
  with TemporaryDirectory() as d:
   c,*_=self.case(d);c['phase_cap_usd']=.045;cl=FakeClient(c['api'])
   with self.assertRaises(BudgetExceeded):run_evaluation(c,cl)
   self.assertEqual(cl.calls,[]);self.assertFalse((Path(c['output_dir'])/'pilot_cost_audit.json').exists())
 @unittest.skipUnless(EXISTS,'pending')
 def test_retry_or_changed_identity_or_tampered_receipt_rejected(self):
  with TemporaryDirectory() as d:
   c,*_=self.case(d);cl=FakeClient(c['api']);c['api']['max_transport_retries']=1
   with self.assertRaises(ValueError):run_evaluation(c,cl)
   c['api']['max_transport_retries']=0;run_evaluation(c,cl)
   p=next((Path(c['output_dir'])/'predictions').glob('*.json'));o=json.loads(p.read_text());o['prediction']=[{'seq':1,'code':'V-ID','severity':'high'}];atomic_json(p,o)
   with self.assertRaises(ValueError):run_evaluation(c,cl)
  with TemporaryDirectory() as d:
   c,r,s,a=self.case(d);cl=FakeClient(c['api']);run_evaluation(c,cl);r[0]['visible_task']='changed';atomic_json(c['records'],r)
   with self.assertRaises(ValueError):run_evaluation(c,cl)
 @unittest.skipUnless(EXISTS,'pending')
 def test_interrupted_pilot_resumes_using_remaining_cost_only(self):
  from unittest.mock import patch
  with TemporaryDirectory() as d:
   c,*_=self.case(d);c['api']['test_unit_cost']=.0068
   cost=Path(c['planned_stage_budget']['cost_audit']);obj=json.loads(cost.read_text());obj['unit_cost_p95']=.0068;atomic_json(cost,obj)
   c['planned_stage_budget']['cost_audit_sha256']=hashlib.sha256(cost.read_bytes()).hexdigest()
   cl=FakeClient(c['api']);original=cl.call
   def interrupted(*args,**kwargs):
    if len(cl.calls)==10:raise RuntimeError('simulated interruption')
    return original(*args,**kwargs)
   with patch.object(cl,'call',side_effect=interrupted):
    with self.assertRaisesRegex(RuntimeError,'simulated'):run_evaluation(c,cl)
   self.assertEqual(len(cl.calls),10)
   self.assertTrue(run_evaluation(c,cl)['pilot_complete']);self.assertEqual(len(cl.calls),30)
 @unittest.skipUnless(EXISTS,'pending')
 def test_missing_stage_ledger_cannot_reset_cap_even_after_complete(self):
  with TemporaryDirectory() as d:
   c,*_=self.case(d);cl=FakeClient(c['api']);run_evaluation(c,cl)
   (Path(c['output_dir'])/'E1_pilot_budget.json').unlink()
   with self.assertRaisesRegex(ValueError,'ledger'):run_evaluation(c,cl)
   self.assertEqual(len(cl.calls),30)
 @unittest.skipUnless(EXISTS,'pending')
 def test_reduced_stage_spend_is_detected(self):
  with TemporaryDirectory() as d:
   c,*_=self.case(d);cl=FakeClient(c['api']);run_evaluation(c,cl)
   path=Path(c['output_dir'])/'E1_pilot_budget.json';ledger=json.loads(path.read_text());ledger['spent']=0.;atomic_json(path,ledger)
   with self.assertRaisesRegex(ValueError,'spend'):run_evaluation(c,cl)
 @unittest.skipUnless(EXISTS,'pending')
 def test_invalid_gold_or_bootstrap_refused_before_paid_calls(self):
  for kind in ('gold','bootstrap','duplicate_action'):
   with TemporaryDirectory() as d:
    c,r,s,a=self.case(d);cl=FakeClient(c['api'])
    if kind=='gold':r[-1]['gold']=[{'seq':999,'code':'V-OT','severity':'high'}]
    elif kind=='bootstrap':c['bootstrap_samples']=0
    else:r[-1]['actions'].append(dict(r[-1]['actions'][0]))
    atomic_json(c['records'],r)
    with self.assertRaises(ValueError):run_evaluation(c,cl)
    self.assertEqual(cl.calls,[])
 @unittest.skipUnless(EXISTS,'pending')
 def test_exact_cached_replay_does_not_charge_new_stage(self):
  with TemporaryDirectory() as d:
   c,*_=self.case(d);cl=FakeClient(c['api']);run_evaluation(c,cl)
   c['output_dir']=str(Path(d)/'replay');run_evaluation(c,cl)
   self.assertEqual(len(cl.calls),30)
   self.assertTrue(run_evaluation(c,cl)['pilot_complete'])
   self.assertEqual(json.loads((Path(c['output_dir'])/'E1_pilot_budget.json').read_text())['spent'],0.)
 @unittest.skipUnless(EXISTS,'pending')
 def test_full_cannot_start_without_pilot(self):
  with TemporaryDirectory() as d:
   c,*_=self.case(d);c['mode']='full';cl=FakeClient(c['api'])
   with self.assertRaisesRegex(ValueError,'completed pilot'):run_evaluation(c,cl)
   self.assertEqual(cl.calls,[])
 @unittest.skipUnless(EXISTS,'pending')
 def test_cli_missing_approval_stops_before_client_or_calls(self):
  from unittest.mock import patch
  from guidance_rl.cli import main
  with TemporaryDirectory() as d:
   c,*_=self.case(d);p=Path(d)/'config.json';atomic_json(p,c)
   with patch('sys.argv',['cli','fixed-eval','--config',str(p)]),patch('guidance_rl.fixed_evaluation.validate_cli_config'),patch('guidance_rl.cli.OpenAIClient') as client:
    with self.assertRaisesRegex(ValueError,'requires its own'):main()
    client.assert_not_called()
 @unittest.skipUnless(EXISTS,'pending')
 def test_metrics_disclose_denominators_inactive_and_severity_misses(self):
  with TemporaryDirectory() as d:
   c,r,s,a=self.case(d);r=r[:8];pred={v['traj_id']:[] for v in r};m=report_metrics(r,pred,20)
   self.assertEqual(m['pair_acc'],.5);self.assertEqual(m['pair_acc_task_bootstrap_ci'],[.5,.5]);self.assertEqual(m['n_clean'],4)
   self.assertIn('recall:V-IC:no_reference_positives',m['inactive_constraints']);self.assertEqual(m['severity']['missed_gold_pairs'],4)
   self.assertIsNone(m['severity']['conditional_accuracy']);self.assertEqual(m['sar_related_f1_denominator'],4)

if __name__=='__main__':unittest.main()
