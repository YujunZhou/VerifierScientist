import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[3] / 'VerifierScientist/experiments/guidance_rl_2026-11-02/control/fixture_runtime_recover.py'
spec = importlib.util.spec_from_file_location('fixture_revalidation', SCRIPT)
recovery = importlib.util.module_from_spec(spec); spec.loader.exec_module(recovery)


class RevalidationTests(unittest.TestCase):
    def test_revalidation_clears_old_success_before_any_worker_can_fail(self):
        with TemporaryDirectory() as directory:
            exp = Path(directory); (exp / 'fixtures').mkdir(); audit = exp / 'audit'; audit.mkdir()
            path = exp / 'fixtures/restore_manifest.json'
            path.write_text(json.dumps({'ready': True, 'fixtures': {'task': {
                'ready': True, 'validation_complete': True, 'source_restored': True,
                'environment': {'runtime_validated': True}}}}))
            with patch.object(recovery, 'EXP', exp), patch.object(recovery, 'AUDIT', audit):
                recovery.invalidate_validation(['task'])
            data = json.loads(path.read_text()); row = data['fixtures']['task']
            self.assertFalse(data['ready']); self.assertFalse(row['ready'])
            self.assertFalse(row['validation_complete']); self.assertFalse(row['environment']['runtime_validated'])
            self.assertTrue(row['source_restored'])
