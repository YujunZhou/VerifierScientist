"""Explicit immutable test-environment boundary for the new fixture copies.

The unchanged benchmark calls a `pip` adjacent to its marker's interpreter,
then falls back to worktree imports when relinking fails. We deliberately take
that fallback: a shared fixture environment must never be rewritten by eight
concurrent trajectories. The launcher also prevents training's Python 3.11
site-packages from entering a fixture's older interpreter.
"""
from pathlib import Path
import shlex


def install_launcher(fixture, venv_python, launcher_dir):
    fixture = Path(fixture).absolute()
    # Do NOT resolve the venv interpreter symlink: that would bypass its venv.
    python = Path(venv_python).absolute()
    if not python.is_file():
        raise ValueError('Fixture interpreter missing')
    directory = Path(launcher_dir).absolute()
    directory.mkdir(parents=True, exist_ok=True)
    launcher = directory / 'python'
    launcher.write_text('#!/bin/sh\nset -eu\nunset PYTHONHOME\n'
                        'export PYTHONNOUSERSITE=1\n'
                        'export PYTHONPATH="$PWD/src:$PWD"\n'
                        f'exec {shlex.quote(str(python))} "$@"\n')
    pip = directory / 'pip'
    pip.write_text('#!/bin/sh\n'
                   "echo 'Fixture environment is immutable; use worktree imports.' >&2\n"
                   'exit 1\n')
    launcher.chmod(0o755); pip.chmod(0o755)
    (fixture / '.harnessaudit_testenv').write_text(str(launcher) + '\n')
    return launcher
