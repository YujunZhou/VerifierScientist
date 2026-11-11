import importlib.util,unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import Mock
EXISTS=importlib.util.find_spec('guidance_rl.optimize') is not None
if EXISTS:
 from guidance_rl.optimize import judge_record, validate_rewrite, optimize
 from guidance_rl.core import make_artifact
class OptimizeTests(unittest.TestCase):
 def test_module_exists(self):self.assertTrue(EXISTS)
 @unittest.skipUnless(EXISTS,'pending')
 def test_judge_never_receives_gold_or_hidden_rules(self):
  c=Mock();c.call.return_value.model_dump.return_value={'violations':[]}
  r={'traj_id':'x','task_id':'a','actions':[],'visible_task':'goal','gold':[{'secret':'SECRET'}],'resource_tools':[],'access_rules':'HIDDEN'}
  a=make_artifact(guidance='guide',judge_model='m',judge_prompt='p',policy='base')
  self.assertEqual(judge_record(c,r,a),[])
  prompt=c.call.call_args.args[1];self.assertNotIn('SECRET',prompt);self.assertNotIn('HIDDEN',prompt)
 @unittest.skipUnless(EXISTS,'pending')
 def test_rewrite_must_preserve_rule_ids(self):
  with self.assertRaises(ValueError):validate_rewrite([{'rule_id':'A','text':'one'}],[{'rule_id':'B','text':'two'}])
 @unittest.skipUnless(EXISTS,'pending')
 def test_disallowed_task_fails_before_any_api(self):
  with TemporaryDirectory() as d:
   c=Mock();split={'propose':['a'],'select':['b'],'confirm':['c'],'train_u':['u'],'test':['t']}
   with self.assertRaises(ValueError):optimize([{'task_id':'t'}],split,{},c,Path(d),0,'base')
   c.call.assert_not_called()
if __name__=='__main__':unittest.main()
