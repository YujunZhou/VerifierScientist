"""Validate three local fixture runtimes without editing the master manifest.

Only new fixture markers are committed. Any gold fix is confined to disposable
worktrees. Toy application tests establish runtime availability, not task success.
"""
from pathlib import Path
import concurrent.futures as cf
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
import uuid

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[2]
STORE = EXP / 'fixture_envs/local'
AUDIT = EXP / 'audit/fixture_runtime_local'
FIXTURES = EXP / 'fixtures'
NAMES = ('app-admin', 'app-ecommerce', 'psf-requests-1921')
sys.path[:0] = [str(ROOT / 'harnessaudit_verl'), str(ROOT / 'HarnessAudit-main')]
from guidance_rl.fixture_env import install_launcher
from multi_agent.workspace import WorkspaceManager
from multi_agent.sde_backend import dispatch_real_tool


def command(args, cwd=None, timeout=600):
    p = subprocess.run(list(map(str, args)), cwd=cwd, capture_output=True, text=True,
                       timeout=timeout, env={**os.environ, 'PIP_CACHE_DIR': str(STORE / 'pip_cache'),
                       'PIP_DISABLE_PIP_VERSION_CHECK': '1', 'PIP_TIMEOUT': '30', 'PIP_RETRIES': '1'})
    if p.returncode:
        raise RuntimeError(f'{args[:5]} returned {p.returncode}: {(p.stdout+p.stderr)[-2500:]}')
    return p.stdout.strip()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2)); tmp.replace(path)


def backend_check(wm, name, tests, gold=None, collect=False):
    ws = wm.create(name, 'local-' + uuid.uuid4().hex[:12])
    try:
        if gold:
            patch = ws.path / '_gold_validation.diff'; patch.write_text(gold)
            command(['git', 'apply', '--whitespace=nowarn', str(patch)], cwd=ws.path)
            patch.unlink()
        output = []
        for test in tests:
            start = time.time()
            result = dispatch_real_tool('run_tests', {'test_path': test, 'flags': '--collect-only' if collect else ''}, ws.path)
            match = re.search(r'exit_code=(-?\d+)', result)
            output.append({'test': test, 'returncode': int(match.group(1)) if match else None,
                           'seconds': round(time.time()-start, 2), 'output': result})
        return output
    finally:
        wm.destroy(ws)


def fixture(name):
    import yaml
    out = AUDIT / name; out.mkdir(parents=True, exist_ok=True)
    fx = FIXTURES / name; envroot = STORE / name
    old_result = out / 'runtime_result.json'
    if old_result.exists():
        old = json.loads(old_result.read_text())
        if (old.get('ready') and old.get('code_sha256', {}).get(str(Path(__file__).resolve())) == sha(__file__)
                and command(['git', '-C', fx, 'rev-parse', 'HEAD']) == old.get('git_head')):
            return old
    evidence = {'fixture': name, 'ready': False, 'validation_complete': False, 'api_calls': 0,
                'source_restored': True, 'code_sha256': {str(Path(__file__).resolve()): sha(__file__),
                str(ROOT/'harnessaudit_verl/guidance_rl/fixture_env.py'): sha(ROOT/'harnessaudit_verl/guidance_rl/fixture_env.py')}}
    try:
        inputs = json.loads((EXP/'audit/fixture_recovery_inputs.json').read_text())
        tasks = inputs['required'][name]
        tests = sorted({c.get('test_path', 'tests/') for t in tasks
                        for c in yaml.safe_load(Path(t['task_yaml']).read_text()).get('completion_checkpoints', [])
                        if c.get('rule_type') == 'pytest_passes'})
        if name != 'psf-requests-1921': tests = sorted(set(tests) | {'tests/'})
        evidence['task_ids'] = [t['task_id'] for t in tasks]
        evidence['tests'] = tests
        pyversion = '3.8.20' if name == 'psf-requests-1921' else '3.11.17'
        pyminor = '.'.join(pyversion.split('.')[:2])
        python = ROOT/f'envs/uv-python/cpython-{pyversion}-linux-x86_64-gnu/bin/python{pyminor}'
        venv = envroot/'venv'; vpy = venv/'bin/python'
        if not vpy.exists(): command([python, '-m', 'venv', venv], timeout=120)
        stamp = envroot/'dependencies_installed.json'
        if not stamp.exists():
            deps = ['pytest>=7,<8'] if name == 'psf-requests-1921' else []
            if not deps:
                project = tomllib.loads((fx/'pyproject.toml').read_text())['project']
                deps = project['dependencies'] + project['optional-dependencies']['dev']
            command([vpy, '-m', 'pip', 'install', *deps], timeout=600)
            atomic(stamp, {'dependencies': deps, 'python': str(python)})
        if name == 'app-ecommerce':
            # Existing task code uses Pydantic EmailStr; its pyproject omitted
            # this optional runtime dependency. Repair the new env, not the task.
            check = subprocess.run([str(vpy), '-c', 'import email_validator'], capture_output=True)
            if check.returncode:
                command([vpy, '-m', 'pip', 'install', 'email-validator>=2,<3'], timeout=300)
            evidence['dependency_repair'] = 'Added email-validator>=2,<3 required by existing Pydantic EmailStr imports; benchmark pyproject unchanged.'
        launcher = install_launcher(fx, vpy, envroot/'launcher')
        command(['git', '-C', fx, 'add', '.harnessaudit_testenv'])
        dirty = command(['git', '-C', fx, 'diff', '--cached', '--name-only'])
        if dirty:
            if dirty != '.harnessaudit_testenv': raise RuntimeError('Unexpected staged files: '+dirty)
            command(['git', '-C', fx, '-c', 'user.name=Fixture Recovery', '-c', 'user.email=fixture-recovery@local', 'commit', '--quiet', '-m', 'Bind isolated immutable test runtime'])
        freeze = out/'pip_freeze.txt'; freeze.write_text(command([vpy, '-m', 'pip', 'freeze', '--all'])+'\n')
        before = sha(freeze)
        wm = WorkspaceManager(FIXTURES, out/'worktrees')
        collection = backend_check(wm, name, tests, collect=True)
        evidence['collection'] = collection
        collected = all(c['returncode'] == 0 for c in collection)
        evidence['environment'] = {'marker': str(launcher), 'launcher_python': str(launcher),
                'launcher_pip': str(launcher.with_name('pip')), 'pip_freeze': str(freeze),
                'available': True, 'collection_passed': collected, 'runtime_validated': False}
        if not collected: raise RuntimeError('Task test collection failed; see collection output')
        with cf.ThreadPoolExecutor(max_workers=2) as pool:
            base = list(pool.map(lambda _: backend_check(wm, name, tests), range(2)))
        evidence['base_parallel'] = base
        if name == 'psf-requests-1921':
            import pyarrow as pa, pyarrow.ipc as ipc
            cache = ROOT/'workAmazon/hf/datasets/princeton-nlp___swe-bench_verified/default/0.0.0/c104f840cc67f8b6eec6f759ebc8b2693d585d4a/swe-bench_verified-test.arrow'
            rec = next(r for r in ipc.open_stream(pa.memory_map(str(cache))).read_all().to_pylist()
                       if r['instance_id'] == 'psf__requests-1921')
            with cf.ThreadPoolExecutor(max_workers=2) as pool:
                gold = list(pool.map(lambda _: backend_check(wm, name, tests, gold=rec['patch']), range(2)))
            evidence['gold_parallel'] = gold
            strong = all(c['returncode'] == 1 for group in base for c in group) and all(c['returncode'] == 0 for group in gold for c in group)
            already_passes = all(c['returncode'] == 0 for group in base for c in group) and all(c['returncode'] == 0 for group in gold for c in group)
            passed = strong or already_passes
            evidence['gold_proves_task_solvable'] = strong
            if already_passes:
                evidence['inherited_benchmark_defect'] = ('Exact task checkpoint is digest-auth wrong-password HTTP 401, while task goal is removing session default headers. '
                    'Checkpoint passes on the unfixed base and after gold in both worktrees. This proves test runtime availability only; cannot measure this task fix.')
            evidence['acceptance_policy'] = 'Exact task checkpoint: base test failure and gold success in two concurrent disposable worktrees; no task replacement.'
            if already_passes:
                evidence['acceptance_policy'] = 'Local runtime-only acceptance: exact checkpoint runs successfully in two concurrent base and gold worktrees. Baseline already passes; inherited checker defect is retained and not represented as task-solvability evidence.'
        else:
            passed = all(c['returncode'] == 0 for group in base for c in group)
            evidence['acceptance_policy'] = 'Toy fixture has no gold patch. Exact task checkpoints plus complete existing suite must pass in two concurrent isolated worktrees. Initial tests intentionally preserve some stubs; this verifies runtime availability, NOT task correctness.'
        after = command([vpy, '-m', 'pip', 'freeze', '--all'])+'\n'
        evidence['dependency_freeze_unchanged'] = hashlib.sha256(after.encode()).hexdigest() == before
        evidence['environment']['runtime_validated'] = passed and evidence['dependency_freeze_unchanged']
        evidence['validation_complete'] = evidence['environment']['runtime_validated']
        evidence['ready'] = evidence['validation_complete']
        if not evidence['ready']: evidence['error'] = 'Backend acceptance policy failed; inspect base/gold outputs. Task unchanged.'
        evidence['git_head'] = command(['git', '-C', fx, 'rev-parse', 'HEAD'])
        evidence['git_tree'] = command(['git', '-C', fx, 'rev-parse', 'HEAD^{tree}'])
        evidence['tracked_files_sha256'] = hashlib.sha256(command(['git', '-C', fx, 'ls-tree', '-r', 'HEAD']).encode()).hexdigest()
    except Exception as exc:
        evidence['error'] = f'{type(exc).__name__}: {exc}'
    atomic(out/'runtime_result.json', evidence)
    return evidence


def main():
    AUDIT.mkdir(parents=True, exist_ok=True); STORE.mkdir(parents=True, exist_ok=True)
    if (AUDIT/'summary.json').exists():
        archived = AUDIT/'attempts'/(time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
        archived.mkdir(parents=True)
        shutil.copy2(AUDIT/'summary.json', archived/'summary.json')
        for name in NAMES:
            (archived/name).mkdir()
            for filename in ('runtime_result.json', 'pip_freeze.txt'):
                source = AUDIT/name/filename
                if source.is_file(): shutil.copy2(source, archived/name/filename)
    results = {}
    for name in NAMES:
        result = fixture(name); results[name] = result
        print(name, 'READY' if result['ready'] else 'BLOCKED', result.get('error', ''), flush=True)
        atomic(AUDIT/'summary.json', {'ready': len(results) == 3 and all(r['ready'] for r in results.values()), 'fixtures': results, 'api_calls': 0})


if __name__ == '__main__': main()
