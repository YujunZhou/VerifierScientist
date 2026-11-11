import copy
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch
from pathlib import Path
from guidance_rl import cli


class FixtureManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / 'fixture'
        self.repo.mkdir()
        launchers = self.root / 'runtime'; launchers.mkdir()
        self.python = launchers / 'python'; self.python.write_text('#!/bin/sh\nexit 0\n'); self.python.chmod(0o755)
        self.pip = launchers / 'pip'; self.pip.write_text('#!/bin/sh\nexit 1\n'); self.pip.chmod(0o755)
        self.freeze = launchers / 'pip-freeze.txt'; self.freeze.write_text('pytest==8.0.0\n')
        self.git('init', '--quiet')
        (self.repo / 'source.py').write_text('x = 1\n')
        (self.repo / '.harnessaudit_testenv').write_text(str(self.python) + '\n')
        self.git('add', '-A')
        self.commit()
        self.entry = {
            'source_restored': True, 'ready': True, 'validation_complete': True,
            'git_head': self.git('rev-parse', 'HEAD'),
            'git_tree': self.git('rev-parse', 'HEAD^{tree}'),
            'environment': {'marker': str(self.python), 'available': True,
                            'collection_passed': True, 'runtime_validated': True,
                            'launcher_python': str(self.python), 'launcher_pip': str(self.pip),
                            'pip_freeze': str(self.freeze)},
        }
        self.manifest = {'ready': True, 'required_fixtures': ['fixture'],
                         'fixtures': {'fixture': self.entry}}
        self.save()

    def git(self, *args):
        return subprocess.run(['git', '-C', str(self.repo), *args], check=True,
                              text=True, capture_output=True).stdout.strip()

    def commit(self):
        self.git('-c', 'user.email=test@local', '-c', 'user.name=Test',
                 'commit', '--quiet', '-m', 'fixture test')

    def save(self):
        (self.root / 'restore_manifest.json').write_text(json.dumps(self.manifest))

    def problems(self):
        return cli.fixture_problems(self.root, {'fixture'})

    def test_valid_clean_committed_fixture(self):
        self.assertEqual(self.problems(), [])

    def test_top_ready_cannot_override_each_runtime_check(self):
        original = copy.deepcopy(self.entry)
        for path in [('source_restored',), ('ready',), ('validation_complete',),
                     ('environment', 'available'), ('environment', 'collection_passed'),
                     ('environment', 'runtime_validated')]:
            with self.subTest(field=path):
                self.entry.clear(); self.entry.update(copy.deepcopy(original))
                target = self.entry if len(path) == 1 else self.entry[path[0]]
                target[path[-1]] = False
                self.save()
                self.assertTrue(self.problems())

    def test_dirty_and_staged_tracked_source_rejected(self):
        (self.repo / 'source.py').write_text('x = 2\n')
        self.assertTrue(self.problems())
        self.git('add', 'source.py')
        self.assertTrue(self.problems())

    def test_committed_head_or_tree_change_rejected(self):
        (self.repo / 'source.py').write_text('x = 2\n')
        self.git('add', 'source.py'); self.commit()
        self.assertTrue(self.problems())
        self.entry['git_head'] = self.git('rev-parse', 'HEAD')
        self.save()
        self.assertTrue(self.problems())

    def test_untracked_marker_is_not_available_in_worktree(self):
        marker = self.repo / '.harnessaudit_testenv'
        self.git('rm', '--cached', '.harnessaudit_testenv'); self.commit()
        self.entry.update(git_head=self.git('rev-parse', 'HEAD'),
                          git_tree=self.git('rev-parse', 'HEAD^{tree}'))
        self.save()
        self.assertTrue(marker.exists())
        self.assertTrue(self.problems())

    def test_required_set_must_match_dataset(self):
        self.assertTrue(cli.fixture_problems(self.root, {'fixture', 'missing'}))

    def test_interpreter_must_match_committed_marker(self):
        self.entry['environment']['marker'] = '/unreviewed/python'
        self.save()
        self.assertTrue(self.problems())

    def test_immutable_environment_evidence_must_exist(self):
        self.freeze.unlink()
        self.assertTrue(self.problems())

    def test_review_inventory_binds_initial_artifact_and_launchers(self):
        control=self.root/'control';control.mkdir()
        script=control/'finalize_authorized_launch.py';script.write_text('# fixture\n')
        initial=self.root/'initial.json';initial.write_text('{}')
        cfg={'code_root':str(self.root/'code'),'experiment_dir':str(self.root),
             'prepared':str(self.root/'prepared'),'tool_config':str(self.root/'tools.yaml'),
             'initial_artifact':str(initial),'model_path':str(self.root/'model'),
             'fixtures_root':str(self.root),'runtime_bin':str(self.root/'runtime')}
        frame=Mock();frame.extra_info=[]
        with patch.object(cli,'ROOT',self.root),patch('pandas.read_parquet',return_value=frame):
            sources,inputs=cli.review_files(cfg)
        self.assertIn(str(script.resolve()),sources)
        for path in (initial,self.python,self.pip,self.freeze,control/'user_authorization.json'):
            self.assertIn(str(path.resolve()),inputs)
        for name in ('ninja','python','python3'):
            self.assertIn(str((self.root/'runtime'/name).resolve()),inputs)
        for name in ('main_ppo.py','constants_ppo.py'):
            self.assertIn(str(self.root/'envs/verl08/lib/python3.11/site-packages/verl/trainer'/name),sources)

    def test_static_inventory_does_not_require_incomplete_fixture_manifest(self):
        (self.root/'restore_manifest.json').unlink()
        cfg={'code_root':str(self.root/'code'),'experiment_dir':str(self.root),
             'prepared':str(self.root/'prepared'),'tool_config':str(self.root/'tools.yaml'),
             'initial_artifact':str(self.root/'initial.json'),'model_path':str(self.root/'model'),
             'fixtures_root':str(self.root)}
        frame=Mock();frame.extra_info=[{'task_yaml':str(self.root/'task.yaml')}]
        with patch.object(cli,'ROOT',self.root),patch('pandas.read_parquet',return_value=frame):
            sources,inputs=cli.review_files(cfg,include_fixtures=False)
        self.assertIn(str((self.root/'code/verifier_bench/common.py').resolve()),sources)
        for path in (self.root/'task.yaml',self.root/'tools.yaml',self.root/'initial.json'):
            self.assertIn(str(path.resolve()),inputs)
        self.assertNotIn(str((self.root/'restore_manifest.json').resolve()),inputs)
        self.assertNotIn(str(self.python.resolve()),inputs)


if __name__ == '__main__':
    unittest.main()
