"""End-to-end optimizer replay using a deterministic local judge."""
import json,unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from guidance_rl.optimize import optimize,Library,Verdict,Equivalence
class FakeAPI:
 config={'concurrency':1}
 def __init__(self):self.requests=[]
 def call(self,instructions,user,schema,**kwargs):
  self.requests.append((instructions,user,kwargs))
  if schema is Verdict:
   d=json.loads(user);return Verdict(violations=[{'seq':a['sequence_num'],'code':'V-OT','reason':'recorded forbidden call'} for a in d['actions'] if a.get('tool_name')=='forbidden'])
  if schema is Equivalence:return Equivalence(equivalent=True,changed_rule_ids=[],explanation='fixture')
  if kwargs.get('tag')=='candidate:remove_evidence':return Library(rules=[{'rule_id':'wrong','text':'invalid mapping'}])
  return Library(rules=[{'rule_id':'C1','text':'Do not use forbidden tools.'}])
class Replay(unittest.TestCase):
 def test_retain_incumbent_and_resume_without_requery(self):
  split={'propose':['p'],'select':['s'],'confirm':['c'],'train_u':['u'],'test':['t']}
  rows=[]
  for tid in ('p','s','c'):
   for i in range(2):
    action={'sequence_num':1,'surface':'tool_call','tool_name':'forbidden' if i else 'safe'}
    r={'task_id':tid,'traj_id':tid+str(i),'visible_task':tid,'actions':[action],'resource_tools':[], 'gold':[{'seq':1,'code':'V-OT'}] if i else []}
    if tid=='p':r['annotated_rules']='PROPOSAL_RULES_ONLY'
    rows.append(r)
  api=FakeAPI();cfg={'generator_model':'m','judge_model':'m','judge_replicates':1,'bootstrap_samples':10}
  with TemporaryDirectory() as d:
   a=optimize(rows,split,cfg,api,Path(d),0,'base')
   self.assertTrue((Path(d)/'remove_evidence_rejected.json').exists())
   self.assertEqual(a['provenance']['selection'],'incumbent');self.assertFalse(a['provenance']['gate']['accepted'])
   count=len(api.requests);b=optimize(rows,split,cfg,api,Path(d),0,'base');self.assertEqual(a,b);self.assertEqual(count,len(api.requests))
   self.assertFalse(any('PROPOSAL_RULES_ONLY' in u for instr,u,k in api.requests if k.get('tag','').startswith('judge:')))
   self.assertFalse(any('judge:c' in k.get('tag','') for instr,u,k in api.requests))
if __name__=='__main__':unittest.main()

class FeedbackBudget(unittest.TestCase):
 def test_feedback_skips_whole_examples_over_budget(self):
  from guidance_rl.optimize import _feedback
  rows=[{'task_id':'a','traj_id':'x','visible_task':'goal','actions':[{'sequence_num':1,'content':'x'*10000}],'gold':[{'seq':1,'code':'V-ID'}]}]
  result=_feedback(rows,{'x':[]},max_bytes=500)
  self.assertEqual(result,[])
