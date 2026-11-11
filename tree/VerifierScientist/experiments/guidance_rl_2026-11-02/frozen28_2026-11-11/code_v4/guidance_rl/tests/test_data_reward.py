import unittest,importlib.util,json
from pathlib import Path
from tempfile import TemporaryDirectory
EXISTS=importlib.util.find_spec('guidance_rl.data') is not None
if EXISTS:
 from guidance_rl.data import load_exact_payload,append_final,load_calibration
class DataTests(unittest.TestCase):
 def test_module_exists(self):self.assertTrue(EXISTS)
 @unittest.skipUnless(EXISTS,'pending')
 def test_concurrent_same_task_payloads_exact_and_repeatable(self):
  with TemporaryDirectory() as d:
   for req in ['a','b']:(Path(d)/f't__{req}.json').write_text(json.dumps({'task_id':'t','request_id':req,'actions':[req]}))
   for i in range(2):self.assertEqual(load_exact_payload(d,'t','a')['actions'],['a'])
   with self.assertRaises(FileNotFoundError):load_exact_payload(d,'t','missing')
 @unittest.skipUnless(EXISTS,'pending')
 def test_final_append_does_not_mutate_episode(self):
  p={'actions':[],'run_id':'r','hub_role':'hub'};a=append_final(p,'answer')
  self.assertEqual(p['actions'],[]);self.assertEqual(a[0]['content'],'answer');self.assertEqual(a[0]['target_role'],'user')
 @unittest.skipUnless(EXISTS,'pending')
 def test_calibration_rejects_test_and_missing_groups(self):
  with TemporaryDirectory() as d:
   p=Path(d)/'test.json';p.write_text(json.dumps({'record':{'task_id':'t','traj_id':'a'}}))
   with self.assertRaises(ValueError):load_calibration(d,{'propose':['r'],'select':[],'confirm':[]})
if __name__=='__main__':unittest.main()
