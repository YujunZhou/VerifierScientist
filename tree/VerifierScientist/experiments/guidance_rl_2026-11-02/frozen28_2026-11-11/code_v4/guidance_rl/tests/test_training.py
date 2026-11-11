import unittest,importlib.util
EXISTS=importlib.util.find_spec('guidance_rl.training') is not None
if EXISTS:from guidance_rl.training import trainer_command,validate_checkpoint
class TrainingTests(unittest.TestCase):
 def test_snapshot_contract_binds_project_installed_defaults(self):
  from tempfile import TemporaryDirectory
  from pathlib import Path
  from guidance_rl.training import training_source_contract
  with TemporaryDirectory() as directory:
   root=Path(directory);defaults=root/'envs/verl08/lib/python3.11/site-packages/verl/trainer/config/default.yaml'
   defaults.parent.mkdir(parents=True);defaults.write_text('setting: before\n')
   c={'python':'python','code_root':str(root/'nested/snapshot'),'project_root':str(root),
      'model_path':'/model','prepared':str(root/'prepared'),'tool_config':'/tools','run_dir':'/run'}
   before=training_source_contract(c);defaults.write_text('setting: after\n')
   self.assertNotEqual(before,training_source_contract(c))
 def test_module_exists(self):self.assertTrue(EXISTS)
 @unittest.skipUnless(EXISTS,'pending')
 def test_validation_flags_and_explicit_resume(self):
  from tempfile import TemporaryDirectory
  from pathlib import Path
  c={'python':'python','code_root':'/code','model_path':'/base','prepared':'/data','tool_config':'/tools','run_dir':'/run'}
  a=trainer_command(c,'calibration',7,Path('/phase'),'/ckpt/global_step_7')
  self.assertIn('trainer.val_only=True',a);self.assertIn('trainer.val_before_train=True',a)
  self.assertIn('trainer.resume_mode=resume_path',a);self.assertIn('actor_rollout_ref.model.path=/base',a)
  self.assertIn('actor_rollout_ref.rollout.val_kwargs.n=8',a)
  b=trainer_command(c,'train',14,Path('/phase'),'/ckpt/global_step_7')
  self.assertIn('trainer.total_training_steps=14',b);self.assertIn('trainer.test_freq=-1',b)
 @unittest.skipUnless(EXISTS,'pending')
 def test_incomplete_checkpoint_rejected(self):
  from tempfile import TemporaryDirectory
  with TemporaryDirectory() as d:
   with self.assertRaises(ValueError):validate_checkpoint(d)
 @unittest.skipUnless(EXISTS,'pending')
 def test_rank_zero_only_checkpoint_rejected(self):
  from tempfile import TemporaryDirectory
  from pathlib import Path
  with TemporaryDirectory() as d:
   p=Path(d);(p/'actor').mkdir();(p/'data.pt').write_bytes(b'x')
   for kind in ('model','optim','extra_state'):(p/'actor'/f'{kind}_world_size_3_rank_0.pt').write_bytes(b'x')
   with self.assertRaises(ValueError):validate_checkpoint(p)
 @unittest.skipUnless(EXISTS,'pending')
 def test_partial_stage_cannot_overwrite_saved_checkpoint(self):
  from pathlib import Path
  from tempfile import TemporaryDirectory
  from guidance_rl import training
  self.assertTrue(hasattr(training,'ensure_new_training_range'))
  with TemporaryDirectory() as d:
   p=Path(d);(p/'global_step_7').mkdir()
   with self.assertRaises(ValueError):training.ensure_new_training_range(p,0)
   training.ensure_new_training_range(p,7)
 @unittest.skipUnless(EXISTS,'pending')
 def test_global_retention_across_training_processes(self):
  from pathlib import Path
  from tempfile import TemporaryDirectory
  from guidance_rl import training
  self.assertTrue(hasattr(training,'prune_checkpoints'))
  with TemporaryDirectory() as d:
   root=Path(d)
   for step in (1,6,7,13,14):
    p=root/f'global_step_{step}';(p/'actor').mkdir(parents=True);(p/'data.pt').write_bytes(b'd')
    for kind in ('model','optim','extra_state'):
     for rank in range(3):(p/'actor'/f'{kind}_world_size_3_rank_{rank}.pt').write_bytes(b'x')
   shell=root/'global_step_0';shell.mkdir();(shell/'data.pt').write_bytes(b'old rotated state')
   training.prune_checkpoints(root,keep=2)
   self.assertEqual(sorted(p.name for p in root.glob('global_step_*')),['global_step_13','global_step_14'])
if __name__=='__main__':unittest.main()
