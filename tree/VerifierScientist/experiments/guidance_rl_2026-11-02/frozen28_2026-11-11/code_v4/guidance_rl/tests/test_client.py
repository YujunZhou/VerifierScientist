import importlib.util
import tempfile
import unittest
from pathlib import Path
SPEC=importlib.util.find_spec('guidance_rl.client')
if SPEC:from guidance_rl.client import Budget, BudgetExceeded, load_openai_key
class BudgetTests(unittest.TestCase):
 def test_module_exists(self):self.assertIsNotNone(SPEC)
 @unittest.skipUnless(SPEC,'pending')
 def test_reservations_prevent_concurrent_overspend(self):
  with tempfile.TemporaryDirectory() as d:
   a=Budget(Path(d)/'budget.json',1,3);b=Budget(Path(d)/'budget.json',1,3)
   token=a.reserve(.7)
   with self.assertRaises(BudgetExceeded):b.reserve(.4)
   a.settle(token,.2);b.reserve(.7)
   with self.assertRaises(BudgetExceeded):a.reserve(.2)
 @unittest.skipUnless(SPEC,'pending')
 def test_env_is_data_not_executed_and_only_openai_loaded(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'.env';p.write_text('OTHER_SECRET="do-not-use"\nOPENAI_API_KEY="test-value" # comment\n')
   self.assertEqual(load_openai_key(p),'test-value')
if __name__=='__main__':unittest.main()

class RawFailureTests(unittest.TestCase):
    def test_malformed_response_is_saved_and_usage_accounted(self):
        from guidance_rl.client import OpenAIClient
        from guidance_rl.optimize import Verdict
        from unittest.mock import Mock
        import tempfile,json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            cfg={'cache_dir':d+'/cache','budget_file':d+'/budget.json','max_usd':1,'max_calls':3,'prices_verified':True,'prices':{'m':{'input':1,'output':1}}}
            c=OpenAIClient(cfg);c.client=Mock()
            response=Mock();response.text=json.dumps({'status':'incomplete','usage':{'input_tokens':10,'output_tokens':20},'output':[]})
            response.parse.side_effect=ValueError('truncated')
            c.client.responses.with_raw_response.parse.return_value=response
            with self.assertRaises(ValueError):c.call('i','u',Verdict,model='m')
            self.assertTrue(list((Path(d)/'cache/raw').glob('*.json')))
            budget=json.loads((Path(d)/'budget.json').read_text());self.assertFalse(budget['reserved']);self.assertAlmostEqual(budget['spent'],.00003)
