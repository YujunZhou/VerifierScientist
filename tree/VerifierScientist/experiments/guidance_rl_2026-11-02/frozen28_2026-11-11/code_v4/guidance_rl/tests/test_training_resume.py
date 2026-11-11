import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from guidance_rl import training
from guidance_rl.core import make_artifact,digest


class ExplicitResumeTests(unittest.TestCase):
    def test_resume_api_exists(self):
        for name in ('resume_state','execution_plan','attempt_identity','paid_stage_requirements'):
            self.assertTrue(hasattr(training,name),name+' is missing')

    def fixture(self,d,step=4):
        root=Path(d);run=root/'run';run.mkdir();prepared=root/'data';prepared.mkdir()
        (prepared/'train_all.parquet').write_bytes(b'unchanged training data')
        (prepared/'split.json').write_text('{}')
        model=root/'model';model.mkdir();(model/'config.json').write_text('{}')
        code=root/'code';(code/'configs').mkdir(parents=True);(code/'configs/harnessaudit_v08.yaml').write_text('defaults: [ppo_trainer]\n')
        artifact=root/'artifact.json';a=make_artifact(guidance='g',judge_model='judge',judge_prompt=training.ALIGNED_JUDGE_PROMPT,
            policy='base',score_protocol='aligned_v2',provenance={'round':0,'split_hash':digest({})})
        artifact.write_text(json.dumps(a))
        c={'run_dir':str(run),'prepared':str(prepared),'model_path':str(model),'initial_artifact':str(artifact),
           'python':'python','code_root':str(code),'tool_config':str(root/'tools.yaml'),
           'seed':1,'arm':'frozen','final_step':7,'api':{'max_transport_retries':1},
           'optimization':{'score_protocol':'aligned_v2','judge_model':'judge'}}
        checkpoint=run/'checkpoints'/f'global_step_{step}';(checkpoint/'actor').mkdir(parents=True)
        (checkpoint/'data.pt').write_bytes(b'data')
        for kind in ('model','optim','extra_state'):
            for rank in range(3):(checkpoint/'actor'/f'{kind}_world_size_3_rank_{rank}.pt').write_bytes(b'shard')
        attempt=run/'train_07'/'attempt_000';attempt.mkdir(parents=True)
        if hasattr(training,'attempt_identity'):
            identity=training.attempt_identity(c,artifact,1,7)
            identity['source_manifest']='old-approved-manifest'
            (attempt/'checkpoint_identity.json').write_text(json.dumps(identity))
        c['resume_from']={'checkpoint':str(checkpoint),'artifact':str(artifact),'artifact_id':a['artifact_id'],
                          'score_protocol':'aligned_v2','source_attempt':str(attempt)}
        if hasattr(training,'seal_attempt_checkpoints'):training.seal_attempt_checkpoints(c,attempt)
        if hasattr(training,'record_resume_manifest') and (checkpoint/'checkpoint_manifest.json').is_file():training.record_resume_manifest(c)
        return c,checkpoint

    def test_explicit_mid_segment_resume_counts_only_remaining_updates(self):
        self.test_resume_api_exists()
        with TemporaryDirectory() as d:
            c,p=self.fixture(d);state=training.resume_state(c);plan=training.execution_plan(c,state)
            trains=[s for s in plan if s['mode']=='train']
            self.assertEqual([(s['start'],s['end']) for s in trains],[(4,7)])
            self.assertEqual(training.paid_stage_requirements(c,trains[0])['required_calls'],3*192*2)
            self.assertEqual(state['checkpoint'],str(p.resolve()))

    def test_resume_does_not_require_old_and_new_manifest_equality(self):
        self.test_resume_api_exists()
        with TemporaryDirectory() as d:
            c,_=self.fixture(d);c['final_step']=14
            self.assertEqual(training.resume_state(c)['step'],4)

    def test_wrong_identity_or_foreign_checkpoint_fails(self):
        self.test_resume_api_exists()
        with TemporaryDirectory() as d:
            c,p=self.fixture(d)
            c['resume_from']['artifact_id']='different'
            with self.assertRaises(ValueError):training.resume_state(c)
            c['resume_from']['artifact_id']=json.loads(Path(c['initial_artifact']).read_text())['artifact_id']
            Path(c['prepared'],'train_all.parquet').write_bytes(b'changed')
            with self.assertRaises(ValueError):training.resume_state(c)
            c['resume_from']['checkpoint']=str(Path(d)/'foreign/global_step_4')
            with self.assertRaises(ValueError):training.resume_state(c)

    def test_higher_checkpoint_or_missing_rank_fails(self):
        self.test_resume_api_exists()
        with TemporaryDirectory() as d:
            c,p=self.fixture(d);higher=p.parent/'global_step_5';higher.mkdir()
            with self.assertRaises(ValueError):training.resume_state(c)
            higher.rmdir();(p/'actor/optim_world_size_3_rank_2.pt').unlink()
            with self.assertRaises(ValueError):training.resume_state(c)

    def test_one_step_smoke_is_one_update_not_restore_validation_update(self):
        self.test_resume_api_exists()
        c={'final_step':1,'arm':'frozen','api':{'max_transport_retries':1},'optimization':{'judge_model':'judge'}}
        plan=training.execution_plan(c,None)
        trains=[s for s in plan if s['mode']=='train']
        self.assertEqual([(s['start'],s['end']) for s in trains],[(0,1)])
        self.assertEqual(training.paid_stage_requirements(c,trains[0])['required_calls'],384)

    def test_approval_is_still_first_action(self):
        with patch.object(training,'require_approval',side_effect=ValueError('not approved')) as approval,patch.object(training.subprocess,'Popen') as launch:
            with self.assertRaisesRegex(ValueError,'not approved'):training.run({},'/missing')
            approval.assert_called_once();launch.assert_not_called()

    def test_missing_budget_decision_prevents_model_and_api_start(self):
        self.test_resume_api_exists()
        with TemporaryDirectory() as d:
            c,_=self.fixture(d)
            with patch.object(training,'require_approval'),patch.object(training,'assigned_gpus') as gpus,patch.object(training.subprocess,'Popen') as launch,patch.object(training,'OpenAIClient') as client:
                with self.assertRaises(ValueError):training.run(c,'new-approval')
                launch.assert_not_called();client.assert_not_called();gpus.assert_not_called()
            decision=json.loads((Path(c['run_dir'])/'launch_budget_decision.json').read_text())
            self.assertFalse(decision['accepted'])

    def test_each_paid_stage_rechecks_before_launch(self):
        self.test_resume_api_exists()
        with TemporaryDirectory() as d:
            c,p=self.fixture(d);phase=Path(c['run_dir'])/'continued_train_07'
            with patch.object(training,'require_approval'),patch.object(training.subprocess,'Popen') as launch:
                with self.assertRaises(ValueError):training.execute_phase(c,'train',7,phase,p,c['initial_artifact'],{},'new-approval')
                launch.assert_not_called()
            self.assertFalse(json.loads((phase/'budget_decision.json').read_text())['accepted'])

    def test_optimizer_counts_all_candidates_confirmations_and_retries(self):
        self.test_resume_api_exists()
        c={'api':{'max_transport_retries':1},'optimization':{'judge_model':'judge','generator_model':'judge',
            'semantic_auditor_model':'auditor','score_protocol':'aligned_v2','judge_replicates':3,'baseline_start':True}}
        requirement=training.paid_stage_requirements(c,{'mode':'optimization','stage':'optimization_07'})
        # 240 proposal; 80 * 3 * 9 candidate/replicate; 160 * 3 * 3 confirmation;
        # one seed, six rewrites, three semantic audits; each can retry once.
        self.assertEqual(requirement['required_calls'],(240+80*3*9+160*3*3+10)*2)
        self.assertEqual(requirement['models'],['auditor','judge'])

    def test_fresh_launch_refuses_existing_checkpoints_before_models(self):
        self.test_resume_api_exists()
        with TemporaryDirectory() as d:
            c,_=self.fixture(d);c.pop('resume_from')
            with patch.object(training,'require_approval'),patch.object(training,'check_execution_budget') as budget,patch.object(training.subprocess,'Popen') as launch:
                with self.assertRaisesRegex(ValueError,'Existing checkpoint'):training.run(c,'new-approval')
                budget.assert_not_called();launch.assert_not_called()

    def test_nonempty_shard_replacement_is_rejected(self):
        with TemporaryDirectory() as d:
            c,p=self.fixture(d)
            (p/'actor/model_world_size_3_rank_1.pt').write_bytes(b'other nonempty model shard')
            with self.assertRaisesRegex(ValueError,'content|manifest'):training.resume_state(c)

    def test_legacy_checkpoint_without_content_manifest_is_not_blessed_on_resume(self):
        with TemporaryDirectory() as d:
            c,p=self.fixture(d);manifest=p/'checkpoint_manifest.json'
            if manifest.exists():manifest.unlink()
            with self.assertRaisesRegex(ValueError,'manifest'):training.resume_state(c)
            self.assertFalse(manifest.exists())

    def test_changed_learning_rate_source_contract_is_rejected(self):
        with TemporaryDirectory() as d:
            c,_=self.fixture(d);original=training.trainer_command
            def changed(*args,**kwargs):
                return [x.replace('actor_rollout_ref.actor.optim.lr=1e-06','actor_rollout_ref.actor.optim.lr=2e-06') for x in original(*args,**kwargs)]
            with patch.object(training,'trainer_command',side_effect=changed):
                with self.assertRaisesRegex(ValueError,'identity|contract'):training.resume_state(c)

    def test_content_manifest_is_immutable_and_incomplete_checkpoint_not_sealed(self):
        self.assertTrue(hasattr(training,'seal_attempt_checkpoints'),'sealing API missing')
        with TemporaryDirectory() as d:
            c,p=self.fixture(d);manifest=p/'checkpoint_manifest.json';before=manifest.read_bytes()
            attempt=Path(c['resume_from']['source_attempt'])
            incomplete=p.parent/'global_step_5';incomplete.mkdir();(incomplete/'data.pt').write_bytes(b'partial')
            report=training.seal_attempt_checkpoints(c,attempt)
            self.assertFalse((incomplete/'checkpoint_manifest.json').exists())
            self.assertEqual(manifest.read_bytes(),before)
            self.assertIn('global_step_5',report['incomplete'])
            (p/'data.pt').write_bytes(b'replaced nonempty data')
            with self.assertRaisesRegex(ValueError,'manifest|content'):training.seal_attempt_checkpoints(c,attempt)
            self.assertEqual(manifest.read_bytes(),before)

    def test_failed_trainer_seals_only_complete_checkpoints_before_returning_error(self):
        with TemporaryDirectory() as d:
            c,p=self.fixture(d,step=1)
            phase=Path(c['run_dir'])/'train_07_new'
            class FailedTrainer:
                def wait(self):
                    complete=p.parent/'global_step_2';(complete/'actor').mkdir(parents=True)
                    (complete/'data.pt').write_bytes(b'd')
                    for kind in ('model','optim','extra_state'):
                        for rank in range(3):(complete/'actor'/f'{kind}_world_size_3_rank_{rank}.pt').write_bytes(b'checkpoint')
                    partial=p.parent/'global_step_3';partial.mkdir();(partial/'data.pt').write_bytes(b'partial')
                    return 1
            with patch.object(training,'require_approval'),patch.object(training,'approval_digest',return_value='new-approved'),\
                 patch.object(training,'check_phase_budget'),patch.object(training,'stop_owned'),\
                 patch.object(training.subprocess,'Popen',return_value=FailedTrainer()):
                with self.assertRaisesRegex(RuntimeError,'Trainer failed'):
                    training.execute_phase(c,'train',7,phase,p,c['initial_artifact'],{'TMPDIR':'/tmp/test'},'new')
            manifest=p.parent/'global_step_2/checkpoint_manifest.json'
            self.assertTrue(manifest.exists(),'complete checkpoint from failed stage was not bound')
            saved=json.loads(manifest.read_text())
            self.assertEqual(saved['source_attempt'],str((phase/'attempt_000').resolve()))
            self.assertFalse((p.parent/'global_step_3/checkpoint_manifest.json').exists())


if __name__=='__main__':unittest.main()
