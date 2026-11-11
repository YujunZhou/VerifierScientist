import json,shutil
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from guidance_rl import training
from guidance_rl.pipeline import approval_digest
from guidance_rl.tests import test_training_resume as resume_tests


class DurableResumeManifestTests(unittest.TestCase):
    def test_rolling_resume_requires_matching_durable_copy(self):
        with TemporaryDirectory() as d:
            c,p=resume_tests.ExplicitResumeTests().fixture(d)
            durable=training.resume_manifest_path(c)
            self.assertEqual(durable.parent,Path(c['run_dir']).resolve()/'resume_manifests')
            self.assertEqual(training.resume_state(c)['step'],4)
            durable.unlink()
            with self.assertRaisesRegex(ValueError,'durable copy'):training.resume_state(c)
            training.record_resume_manifest(c)
            self.assertEqual(training.resume_state(c)['step'],4)
            tampered=json.loads(durable.read_text());tampered['step']=5;durable.write_text(json.dumps(tampered))
            with self.assertRaisesRegex(ValueError,'durable copy'):training.resume_state(c)
            with self.assertRaisesRegex(ValueError,'differs'):training.record_resume_manifest(c)

    def test_approval_digest_survives_rolling_rotation(self):
        with TemporaryDirectory() as d:
            c,p=resume_tests.ExplicitResumeTests().fixture(d)
            c['code_files']=[];c['review_inputs']=[str(training.resume_manifest_path(c))]
            before=approval_digest(c)
            shutil.rmtree(p)  # veRL max_actor_ckpt_to_keep / prune_checkpoints rotation mid-continuation
            self.assertEqual(approval_digest(c),before)

    def test_milestone_resume_binds_to_milestone_manifest(self):
        with TemporaryDirectory() as d:
            c,p=resume_tests.ExplicitResumeTests().fixture(d,step=7);c['final_step']=14
            attempt=Path(c['resume_from']['source_attempt']);run=Path(c['run_dir']).resolve()
            training.publish_complete_checkpoint(c,attempt,7)
            c['resume_from']['checkpoint']=str(run/'milestones/global_step_7')
            self.assertEqual(training.resume_manifest_path(c),run/'milestones/global_step_7/checkpoint_manifest.json')
            self.assertEqual(training.record_resume_manifest(c),run/'milestones/global_step_7/checkpoint_manifest.json')
            shutil.rmtree(p)
            self.assertEqual(training.resume_state(c)['step'],7)


if __name__=='__main__':unittest.main()
