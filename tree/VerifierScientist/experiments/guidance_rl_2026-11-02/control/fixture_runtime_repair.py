"""Repair only SCM-generated metadata, using exact historical build versions.

No gold code is applied. Original tasks/specs/results remain read-only. Each
repair is separately logged; strict backend validation/publishing follows.
"""
from pathlib import Path
import json
import os
import re
import subprocess
import sys
import time
import importlib.util
import argparse

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[2]
AUDIT = EXP / 'audit/fixture_runtime/repairs'
NAMES = ['pytest-dev-pytest-7432', 'pytest-dev-pytest-7521', 'pytest-dev-pytest-7982',
         'matplotlib-matplotlib-23476']


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--only',nargs='+',choices=NAMES,default=NAMES)
    names=parser.parse_args().only
    AUDIT.mkdir(parents=True, exist_ok=True)
    results = []
    for name in names:
        attempt = AUDIT / (name + '-' + str(int(time.time())))
        attempt.mkdir()
        fx = EXP / 'fixture_envs' / name / 'repos' / name
        py = fx / '.venv/bin/python'
        project = 'pytest' if name.startswith('pytest') else 'matplotlib'
        log = ROOT / 'ha_runs/swe_provision/logs' / (name + '.log')
        if not log.exists(): log = ROOT / 'harnessaudit_verl/swe_provision/logs' / (name + '.log')
        matches = re.findall(r'filename=' + project + r'-([^\s]+?)(?:-0\.editable|-cp\d+|-py\d+)', log.read_text())
        if not matches:
            # Matplotlib's historical editable metadata is also recorded here.
            matches = re.findall(r'from matplotlib==([^\s\)]+)', log.read_text())
        if not matches: raise RuntimeError('No historical generated version: ' + name)
        version = matches[-1]
        env = dict(os.environ, SETUPTOOLS_SCM_PRETEND_VERSION=version,
                   PIP_DISABLE_PIP_VERSION_CHECK='1', PIP_NO_INDEX='1',
                   PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
                   MAX_JOBS='2', MAKEFLAGS='-j2', MPLBACKEND='Agg')
        env.pop('PYTHONPATH', None); env.pop('PYTHONHOME', None)
        result = {'fixture': name, 'historical_log': str(log), 'generated_version': version,
                  'api_calls': 0, 'original_benchmark_modified': False}
        try:
            with (attempt / 'rebuild.log').open('w') as out:
                if project == 'pytest':
                    # This declared PEP517 build dependency was only present in
                    # pip's discarded isolated build env, not the runtime venv.
                    wheelhouse=EXP/'fixture_envs/repair_wheelhouse'
                    subprocess.run([str(py),'-m','pip','install','--no-index','--no-deps',
                        str(wheelhouse/'setuptools_scm-7.1.0-py3-none-any.whl'),
                        str(wheelhouse/'tomli-2.0.1-py3-none-any.whl')],
                        cwd=fx,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=120,check=True)
                subprocess.run([str(py), '-m', 'pip', 'install', '--no-index', '--no-build-isolation',
                                '--no-deps', '-e', '.'], cwd=fx, env=env,
                               stdout=out, stderr=subprocess.STDOUT, timeout=1800, check=True)
            installed=subprocess.check_output([str(py),'-c',
                'import importlib.metadata; print(importlib.metadata.version('+repr(project)+'))'],
                cwd=fx,env=env,text=True,timeout=60).strip()
            if installed != version:
                raise RuntimeError('Generated package version mismatch: '+installed+' != '+version)
            result['installed_version']=installed
            (fx / '.venv/pip.conf').write_text('[global]\nno-index = true\n')
            (fx / '.harnessaudit_testenv').write_text(str(py) + '\n')
            gi=fx/'.gitignore'
            prev=gi.read_text() if gi.exists() else ''
            if '.venv/' not in prev.splitlines(): gi.write_text(prev.rstrip()+'\n.venv/\n')
            subprocess.run(['git','add','-A'], cwd=fx, check=True, timeout=120)
            paths=subprocess.check_output(['git','ls-files','-o','-i','--exclude-standard','-z','--','.',
                ':!.venv', ':!**/__pycache__/**', ':!build/**', ':!*.pyc', ':!*.o'],cwd=fx,timeout=120)
            files=[p.decode() for p in paths.split(b'\0') if p]
            if files: subprocess.run(['git','add','-f','--',*files],cwd=fx,check=True,timeout=120)
            diff=subprocess.check_output(['git','diff','--cached'],cwd=fx,timeout=120)
            (attempt/'tracked_changes.diff').write_bytes(diff)
            subprocess.run(['git','-c','user.email=fixture-recovery@local','-c','user.name=Fixture Recovery',
                'commit','--quiet','--allow-empty','-m','Restore SCM generated metadata from historical build provenance'],cwd=fx,check=True,timeout=120)
            status=EXP/'audit/fixture_runtime'/name/'status'/(name+'.json')
            old=json.loads(status.read_text())
            (attempt/'previous_status.json').write_text(json.dumps(old,indent=2))
            old.update(ok=True,repair_evidence=str(attempt),repair_generated_version=version)
            old.pop('error',None)
            status.write_text(json.dumps(old,indent=2))
            result['rebuild_ok']=True
        except Exception as e:
            result.update(rebuild_ok=False,error=str(e))
        (attempt/'result.json').write_text(json.dumps(result,indent=2))
        results.append(result)
        print(json.dumps(result),flush=True)
    (AUDIT/'summary.json').write_text(json.dumps(results,indent=2))
    subprocess.run([sys.executable,str(EXP/'control/fixture_runtime_recover.py'),'--only',*names,
        'pylint-dev-pylint-4604','--resume','--validate','--publish','--fixture-timeout','900'],check=True)


if __name__=='__main__': main()
