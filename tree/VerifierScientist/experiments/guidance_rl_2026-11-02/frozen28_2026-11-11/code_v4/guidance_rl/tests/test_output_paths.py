import unittest
from guidance_rl import cli
class OutputPaths(unittest.TestCase):
 def test_old_results_cannot_be_training_output(self):
  self.assertTrue(hasattr(cli,'validate_output_paths'))
  from guidance_rl.data import ROOT
  exp=ROOT/'VerifierScientist/experiments/example'
  c={'experiment_dir':str(exp),'run_dir':str(ROOT/'ha_runs/old_run'),'prepared':str(exp/'data'),'initial_artifact':str(exp/'initial/a.json'),'api':{'cache_dir':str(exp/'cache'),'budget_file':str(exp/'budget.json')}}
  with self.assertRaises(ValueError):cli.validate_output_paths(c)
  c['run_dir']=str(exp/'run');cli.validate_output_paths(c)
if __name__=='__main__':unittest.main()
