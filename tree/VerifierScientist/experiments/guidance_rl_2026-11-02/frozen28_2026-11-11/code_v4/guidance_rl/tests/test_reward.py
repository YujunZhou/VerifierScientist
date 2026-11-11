import unittest,importlib.util,json,os
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch
EXISTS=importlib.util.find_spec('guidance_rl.reward') is not None
if EXISTS:from guidance_rl.reward import compute_score
class RewardTests(unittest.TestCase):
 def test_aligned_artifact_changes_penalty_but_keeps_legacy_audit(self):
  from guidance_rl.core import make_artifact
  with TemporaryDirectory() as d:
   a=make_artifact(guidance='',judge_model='m',judge_prompt='p',policy='base',score_protocol='aligned_v2')
   ap=Path(d)/'artifact.json';ap.write_text(json.dumps(a))
   cp=Path(d)/'config.json';cp.write_text(json.dumps({'artifact':str(ap),'artifact_id':a['artifact_id'],'mode':'train','audit_dir':str(Path(d)/'audit'),'api':{'budget_file':str(Path(d)/'budget.json'),'concurrency':2}}))
   v={'seq':1,'code':'V-OT','severity':'low'}
   r={'traj_id':'r','task_id':'t','actions':[{'sequence_num':1,'surface':'tool_call','tool_name':'read'}],'resource_tools':[],'gold':[v,v]}
   info={'guidance_final':{'text':'a','status':'final','extraction_version':1},'guidance_rollout_id':'r'}
   with patch.dict(os.environ,{'GUIDANCE_REWARD_CONFIG':str(cp)}),patch('guidance_rl.reward.build_record',return_value=(r,{'sar_avg':.9,'tcr_det':.8})),patch('guidance_rl.reward.OpenAIClient'),patch('guidance_rl.reward.judge_record',return_value=[v]):
    out=compute_score(ground_truth='t',extra_info=info)
   self.assertAlmostEqual(out['verifier_sar'],.95)
   self.assertAlmostEqual(out['oracle_sar'],.9)
   self.assertAlmostEqual(out['oracle_sar_aligned'],.95)
   self.assertAlmostEqual(out['score'],.875)
 def test_module_exists(self):self.assertTrue(EXISTS)
 @unittest.skipUnless(EXISTS,'pending')
 def test_missing_capture_fails_not_full_transcript(self):
  with self.assertRaises(ValueError):compute_score(ground_truth='t',solution_str='secret tool observation',extra_info={})
 @unittest.skipUnless(EXISTS,'pending')
 def test_reward_uses_predicted_safety_never_oracle(self):
  with TemporaryDirectory() as d:
   from guidance_rl.core import make_artifact
   a=make_artifact(guidance='g',judge_model='m',judge_prompt='p',policy='base')
   ap=Path(d)/'a.json';ap.write_text(json.dumps(a));cp=Path(d)/'c.json';cp.write_text(json.dumps({'artifact':str(ap),'artifact_id':a['artifact_id'],'mode':'train','audit_dir':d,'api':{'budget_file':str(Path(d)/'budget.json'),'concurrency':2}}))
   info={'guidance_final':{'text':'answer','status':'final','extraction_version':1},'guidance_rollout_id':'r','task_yaml':'a','hub_role':'hub'}
   record={'traj_id':'r','task_id':'t','actions':[],'gold':[],'visible_task':'goal','resource_tools':[]}
   with patch.dict(os.environ,{'GUIDANCE_REWARD_CONFIG':str(cp)}), patch('guidance_rl.reward.build_record',return_value=(record,{'sar_avg':0.,'tcr_det':.8})),patch('guidance_rl.reward.OpenAIClient'),patch('guidance_rl.reward.judge_record',return_value=[]):
    v=compute_score(ground_truth='t',extra_info=info)
    self.assertAlmostEqual(v['score'],.9);self.assertEqual(v['oracle_sar'],0.)
 @unittest.skipUnless(EXISTS,'pending')
 def test_reward_judge_replicates_average_safety_and_keep_first_prediction(self):
  from guidance_rl.core import make_artifact
  with TemporaryDirectory() as d:
   a=make_artifact(guidance='',judge_model='m',judge_prompt='p',policy='base',score_protocol='aligned_v2')
   ap=Path(d)/'artifact.json';ap.write_text(json.dumps(a))
   cp=Path(d)/'config.json';cp.write_text(json.dumps({'artifact':str(ap),'artifact_id':a['artifact_id'],'mode':'train','audit_dir':str(Path(d)/'audit'),'reward_judge_replicates':2,'api':{'budget_file':str(Path(d)/'budget.json'),'concurrency':2}}))
   v={'seq':1,'code':'V-OT','severity':'low'}
   r={'traj_id':'r','task_id':'t','actions':[{'sequence_num':1,'surface':'tool_call','tool_name':'read'}],'resource_tools':[],'gold':[v]}
   info={'guidance_final':{'text':'a','status':'final','extraction_version':1},'guidance_rollout_id':'r'}
   reps=[]
   def judge(client,record,artifact,rep=0,stage=None):
    reps.append(rep);return [v] if rep==0 else []
   with patch.dict(os.environ,{'GUIDANCE_REWARD_CONFIG':str(cp)}),patch('guidance_rl.reward.build_record',return_value=(r,{'sar_avg':.9,'tcr_det':.8})),patch('guidance_rl.reward.OpenAIClient'),patch('guidance_rl.reward.judge_record',side_effect=judge):
    out=compute_score(ground_truth='t',extra_info=info)
   self.assertEqual(reps,[0,1])
   self.assertAlmostEqual(out['verifier_sar'],(.95+1.)/2)
   self.assertEqual(out['judge_replicates'],2)
   audit=json.loads(next((Path(d)/'audit').glob('*.json')).read_text())
   self.assertEqual(audit['prediction'],[v]);self.assertEqual(len(audit['predictions_all_replicates']),2)
   # default config: one replicate, no replicate fields
   cp.write_text(json.dumps({'artifact':str(ap),'artifact_id':a['artifact_id'],'mode':'train','audit_dir':str(Path(d)/'audit2'),'api':{'budget_file':str(Path(d)/'budget.json'),'concurrency':2}}))
   reps.clear()
   with patch.dict(os.environ,{'GUIDANCE_REWARD_CONFIG':str(cp)}),patch('guidance_rl.reward.build_record',return_value=(r,{'sar_avg':.9,'tcr_det':.8})),patch('guidance_rl.reward.OpenAIClient'),patch('guidance_rl.reward.judge_record',side_effect=judge):
    out=compute_score(ground_truth='t',extra_info=info)
   self.assertEqual(reps,[0]);self.assertNotIn('judge_replicates',out)
   self.assertNotIn('predictions_all_replicates',json.loads(next((Path(d)/'audit2').glob('*.json')).read_text()))
if __name__=='__main__':unittest.main()
