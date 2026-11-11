"""CPU-only integration of epoch resume and post-update acceptance."""
import copy,json,os,shutil,sys,unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from guidance_rl import training
from guidance_rl.tests import test_training_resume as resume_tests
from unittest.mock import patch


class DurableCheckpointTests(unittest.TestCase):
    def test_each_publish_globally_rotates_across_phase_but_keeps_milestone(self):
        with TemporaryDirectory() as root:
            c,p=resume_tests.ExplicitResumeTests().fixture(root,step=7)
            attempt=Path(c['resume_from']['source_attempt']);run=Path(c['run_dir'])
            prior=p.parent/'global_step_6';shutil.copytree(p,prior);(prior/'checkpoint_manifest.json').unlink()
            training.publish_complete_checkpoint(c,attempt,7)
            attempt2=run/'train_14/attempt_000';attempt2.mkdir(parents=True)
            identity=training.attempt_identity(c,c['initial_artifact'],7,14);identity['source_manifest']='approved-next'
            (attempt2/'checkpoint_identity.json').write_text(json.dumps(identity))
            for step in (8,9):
                nextpath=p.parent/f'global_step_{step}'
                shutil.copytree(run/'milestones/global_step_7',nextpath);(nextpath/'checkpoint_manifest.json').unlink()
                training.publish_complete_checkpoint(c,attempt2,step)
                self.assertEqual(len(list(p.parent.glob('global_step_*'))),2)
            training.verify_checkpoint_content(run/'milestones/global_step_7',attempt)
            partial=p.parent/'global_step_10';partial.mkdir()
            with self.assertRaises(ValueError):training.publish_complete_checkpoint(c,attempt2,10)
            self.assertTrue((p.parent/'global_step_8').exists())
            self.assertTrue((p.parent/'global_step_9').exists())

    def test_milestone_survives_rotation_and_latest_is_complete(self):
        self.assertTrue(hasattr(training,'publish_complete_checkpoint'))
        with TemporaryDirectory() as root:
            c,p=resume_tests.ExplicitResumeTests().fixture(root,step=7)
            attempt=Path(c['resume_from']['source_attempt']);run=Path(c['run_dir'])
            result=training.publish_complete_checkpoint(c,attempt,7)
            milestone=run/'milestones/global_step_7'
            training.verify_checkpoint_content(milestone,attempt)
            original_inode=(p/'data.pt').stat().st_ino
            self.assertEqual(original_inode,(milestone/'data.pt').stat().st_ino)
            shutil.rmtree(p)  # simulate veRL rotation of rolling checkpoint
            training.verify_checkpoint_content(milestone,attempt)
            self.assertEqual(json.loads((run/'latest_complete.json').read_text())['checkpoint'],str(milestone))
            self.assertEqual(result['step'],7)

    def test_partial_or_corrupt_checkpoint_never_replaces_latest(self):
        self.assertTrue(hasattr(training,'publish_complete_checkpoint'))
        with TemporaryDirectory() as root:
            c,p=resume_tests.ExplicitResumeTests().fixture(root,step=4)
            attempt=Path(c['resume_from']['source_attempt']);run=Path(c['run_dir'])
            training.publish_complete_checkpoint(c,attempt,4)
            before=(run/'latest_complete.json').read_bytes()
            partial=p.parent/'global_step_5';partial.mkdir();(partial/'data.pt').write_bytes(b'partial')
            with self.assertRaises(ValueError):training.publish_complete_checkpoint(c,attempt,5)
            self.assertEqual(before,(run/'latest_complete.json').read_bytes())
            (p/'actor/model_world_size_3_rank_0.pt').write_bytes(b'bad')
            with self.assertRaises(ValueError):training.publish_complete_checkpoint(c,attempt,4)
            self.assertEqual(before,(run/'latest_complete.json').read_bytes())

    def test_insufficient_space_refuses_next_save_without_changing_latest(self):
        self.assertTrue(hasattr(training,'check_checkpoint_space'))
        with TemporaryDirectory() as root:
            c,p=resume_tests.ExplicitResumeTests().fixture(root,step=4)
            attempt=Path(c['resume_from']['source_attempt']);run=Path(c['run_dir'])
            training.publish_complete_checkpoint(c,attempt,4)
            before=(run/'latest_complete.json').read_bytes()
            with patch('shutil.disk_usage',return_value=shutil._ntuple_diskusage(100,99,1)):
                with self.assertRaises(ValueError):training.check_checkpoint_space(c)
            self.assertEqual(before,(run/'latest_complete.json').read_bytes())

    def test_explicit_milestone_resume_requires_no_later_rolling_checkpoint(self):
        with TemporaryDirectory() as root:
            c,p=resume_tests.ExplicitResumeTests().fixture(root,step=7);c['final_step']=14
            attempt=Path(c['resume_from']['source_attempt']);run=Path(c['run_dir'])
            training.publish_complete_checkpoint(c,attempt,7)
            shutil.rmtree(p)
            c['resume_from']['checkpoint']=str(run/'milestones/global_step_7')
            self.assertEqual(training.resume_state(c)['step'],7)
            (run/'checkpoints/global_step_8').mkdir()
            with self.assertRaisesRegex(ValueError,'Existing checkpoint'):training.resume_state(c)


class UpdateAcceptanceTests(unittest.TestCase):
    def fixture(self,root):
        p=Path(root);(p/'audit').mkdir()
        metrics={'actor/grad_norm':.4,'actor/lr':1e-6,'actor/pg_loss':-.1}
        (p/'metrics.jsonl').write_text(json.dumps({'step':1,'data':metrics})+'\n')
        for i in range(192):
            (p/'audit'/f'{i}.json').write_text(json.dumps({'status':'complete','artifact_id':'artifact',
                'record':{'traj_id':str(i),'task_id':str(i//8)},'reward':{'score':.8}}))
        return p

    def test_one_update_requires_complete_rewards_and_finite_training_metrics(self):
        self.assertTrue(hasattr(training,'validate_training_attempt'))
        with TemporaryDirectory() as root:
            p=self.fixture(root)
            result=training.validate_training_attempt(p,0,1,'artifact')
            self.assertEqual(result['updates'],1)
            self.assertEqual(result['reward_count'],192)
            (p/'audit/0.json').unlink()
            with self.assertRaises(ValueError):training.validate_training_attempt(p,0,1,'artifact')

    def test_nan_gradient_or_zero_lr_blocks_continuation(self):
        self.assertTrue(hasattr(training,'validate_training_attempt'))
        with TemporaryDirectory() as root:
            p=self.fixture(root)
            for key,value in [('actor/grad_norm',float('nan')),('actor/lr',0.)]:
                row={'actor/grad_norm':.4,'actor/lr':1e-6,'actor/pg_loss':-.1};row[key]=value
                (p/'metrics.jsonl').write_text(json.dumps({'step':1,'data':row})+'\n')
                with self.assertRaises(ValueError):training.validate_training_attempt(p,0,1,'artifact')

    def test_wrong_artifact_or_missing_update_metrics_rejected(self):
        self.assertTrue(hasattr(training,'validate_training_attempt'))
        with TemporaryDirectory() as root:
            p=self.fixture(root)
            with self.assertRaises(ValueError):training.validate_training_attempt(p,0,1,'other')
            (p/'metrics.jsonl').write_text('')
            with self.assertRaises(ValueError):training.validate_training_attempt(p,0,1,'artifact')


class RuntimeExecutableTests(unittest.TestCase):
    def test_runtime_bin_is_prepended_and_python_aliases_match_reviewed_python(self):
        with TemporaryDirectory() as root:
            root=Path(root);binpath=root/'bin';binpath.mkdir()
            for name in ('python','python3'):(binpath/name).symlink_to(sys.executable)
            ninja=binpath/'ninja';ninja.write_text('#!/bin/sh\necho 1.11.1\n');ninja.chmod(0o755)
            c={'project_root':str(root),'code_root':str(root/'code'),'fixtures_root':str(root/'fixtures'),
               'python':sys.executable,'runtime_bin':str(binpath)}
            env=training.runtime_env(c)
            self.assertEqual(env['PATH'].split(':')[0],str(binpath))
            (binpath/'python').unlink();(binpath/'python').symlink_to('/not/the/reviewed/python')
            with self.assertRaises(ValueError):training.runtime_env(c)

    def test_missing_ninja_is_rejected_before_model_launch(self):
        with TemporaryDirectory() as root:
            c={'project_root':root,'code_root':root,'fixtures_root':root,'python':sys.executable,'runtime_bin':root}
            with self.assertRaisesRegex(ValueError,'runtime|ninja'):training.runtime_env(c)


@unittest.skipUnless(os.getenv('RUN_VERL_DATA_RESUME')=='1','requires installed torchdata CPU environment')
class RealLoaderResumeTests(unittest.TestCase):
    def test_1_7_14_21_28_matches_uninterrupted_sampler(self):
        import torch
        from torchdata.stateful_dataloader import StatefulDataLoader
        from torchdata.stateful_dataloader.sampler import RandomSampler
        self.assertTrue(hasattr(training,'restore_epoch_boundary'))
        def loader():
            ds=list(range(178))
            return StatefulDataLoader(ds,batch_size=24,drop_last=True,num_workers=0,
                sampler=RandomSampler(ds,generator=torch.Generator().manual_seed(1)))
        baseline=loader();expected=[]
        for epoch in range(4):expected.extend(batch.tolist() for batch in baseline)
        actual=[];start=0;state=None
        for end in (1,7,14,21,28):
            current=loader()
            if state is not None:
                if start%len(current):current.load_state_dict(copy.deepcopy(state))
                else:training.restore_epoch_boundary(current,copy.deepcopy(state),start)
            iterator=iter(current)
            actual.extend(next(iterator).tolist() for _ in range(end-start))
            state=current.state_dict();start=end
        self.assertEqual(expected,actual)
        self.assertNotEqual(actual[0],actual[7])


if __name__=='__main__':unittest.main()
