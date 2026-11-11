import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from guidance_rl.fixture_env import install_launcher


class FixtureEnvTests(unittest.TestCase):
    def test_launcher_ignores_parent_pythonpath_and_uses_each_worktree(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / 'fixture'; fixture.mkdir()
            poison = root / 'poison'; poison.mkdir()
            (poison / 'target.py').write_text('value="wrong parent package"')
            launcher = install_launcher(fixture, sys.executable, root / 'launchers with spaces')
            self.assertEqual((fixture / '.harnessaudit_testenv').read_text().strip(), str(launcher))
            env = dict(os.environ, PYTHONPATH=str(poison), PYTHONHOME='/not/a/python/home')
            children = []
            for number in range(2):
                ws = root / f'worktree{number}'; (ws / 'src').mkdir(parents=True)
                (ws / 'src/target.py').write_text(f'value={number}')
                children.append(subprocess.Popen([str(launcher), '-c', 'import target;print(target.value)'],
                                                  cwd=ws, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
            for number, child in enumerate(children):
                stdout, stderr = child.communicate(timeout=10)
                self.assertEqual(child.returncode, 0, stderr)
                self.assertEqual(stdout.strip(), str(number))

    def test_runtime_relink_is_disabled_even_when_package_is_installable(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); fixture = root / 'fixture'; fixture.mkdir()
            launcher = install_launcher(fixture, sys.executable, root / 'launcher')
            result = subprocess.run([str(launcher.with_name('pip')), 'install', '-e', str(fixture)],
                                    capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('immutable', result.stderr)

