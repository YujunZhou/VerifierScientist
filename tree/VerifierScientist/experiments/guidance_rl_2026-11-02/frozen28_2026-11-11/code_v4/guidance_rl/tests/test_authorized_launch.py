import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[3] / 'VerifierScientist/experiments/guidance_rl_2026-11-02/control/finalize_authorized_launch.py'
spec = importlib.util.spec_from_file_location('authorized_launch', SCRIPT)
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


class AuthorizedLaunchTests(unittest.TestCase):
    def test_legacy_materializer_always_refuses_even_matching_inputs(self):
        # Later user withdrawal overrides the old broad authorization.
        for cfg in ({}, {'api': {'max_usd': 10}}, {'api': {'max_usd': 11}}):
            with self.assertRaisesRegex(RuntimeError, 'withdrawn'):
                launch.authorize(cfg, {'fingerprint': {}})

    def test_main_cannot_freeze_or_reauthorize(self):
        with patch('sys.argv', ['finalize_authorized_launch.py', '--freeze']):
            with self.assertRaisesRegex(RuntimeError, 'withdrawn'):
                launch.main()
