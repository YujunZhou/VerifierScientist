"""Real allocated-GPU transport probe followed by fresh Train-R calibration.

This stage makes no OpenAI calls and performs no policy updates. Calibration
returns private oracle audit values only; no unapproved verifier is trained on.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

from .core import atomic_json,digest
from .pipeline import assigned_gpus
from .training import runtime_env, trainer_command, stop_owned


def phase(c, env, directory, dataset, n, expected):
    directory.mkdir(parents=True, exist_ok=False)
    cfg = {'mode': 'calibration', 'api': c['api'], 'artifact': c['initial_artifact'],
           'audit_dir': str(directory / 'audit'), 'calibration_ids': sorted(expected)}
    atomic_json(directory / 'reward_config.json', cfg)
    e = dict(env, GUIDANCE_REWARD_CONFIG=str(directory / 'reward_config.json'),
             HA_EPISODE_DIR=str(directory / 'episodes'), HA_RESULTS_DIR=str(directory / 'results'),
             RAY_TMPDIR=str(Path(env['TMPDIR']).parent / ('r'+digest(str(directory))[:6])))
    argv = trainer_command(c, 'calibration', 1, directory)
    replacements = {'data.val_files': str(dataset), 'actor_rollout_ref.rollout.val_kwargs.n': str(n),
                    'actor_rollout_ref.rollout.agent.num_workers': '1' if len(expected)*n%4 else '4'}
    argv = [arg.split('=', 1)[0] + '=' + replacements[arg.split('=', 1)[0]]
            if '=' in arg and arg.split('=', 1)[0] in replacements else arg for arg in argv]
    atomic_json(directory / 'command.json', {'argv': argv, 'mode': 'calibration', 'policy_updates': 0})
    proc = None
    try:
        with (directory / 'driver.log').open('w') as log:
            proc = subprocess.Popen(argv, env=e, cwd=c['code_root'], stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            if proc.wait() != 0:
                raise RuntimeError('Actual veRL probe failed: ' + str(directory / 'driver.log'))
    finally:
        stop_owned(proc)
    rows = [json.loads(p.read_text()) for p in (directory / 'audit').glob('*.json')]
    from collections import Counter
    counts = Counter(row['record']['task_id'] for row in rows)
    if set(counts) != set(expected) or any(value != n for value in counts.values()):
        raise ValueError('Missing or duplicated real probe trajectories: ' + str(counts))
    if any(row.get('status') != 'complete' or row['final_capture'].get('extraction_version') != 1 for row in rows):
        raise ValueError('Reward callback did not receive verified final capture')
    atomic_json(directory / 'complete.json', {'n_trajectories': len(rows), 'tasks': dict(counts),
                'api_calls': 0, 'policy_updates': 0, 'reward_transport_verified': True})


def run(c):
    import pandas as pd
    out = Path(c['probe_dir']); out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists():
        return
    gpus = assigned_gpus(os.environ)
    env = runtime_env(c)
    df = pd.read_parquet(Path(c['prepared']) / 'calibration.parquet')
    split = json.loads((Path(c['prepared']) / 'split.json').read_text())
    expected = set(split['propose'] + split['select'] + split['confirm'])
    if len(df) != 60 or {x['task_id'] for x in df.extra_info} != expected:
        raise ValueError('Expected all Train-R tasks, with no Test/Train-U mixing')
    # First exercise one actual task; then use the same code path for the full set.
    tid = split['propose'][0]
    one = df[df.extra_info.map(lambda x: x['task_id'] == tid)]
    one.to_parquet(out / 'one_task.parquet', index=False)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    worker_env = dict(env, CUDA_VISIBLE_DEVICES=gpus[-1])
    env.update(CUDA_VISIBLE_DEVICES=','.join(gpus[:3]), HA_SPOKE_BASE_URL=url + '/v1',
               HA_SPOKE_MODEL=c['model_path'], HA_SPOKE_API_KEY='EMPTY')
    argv = [c['python'], '-m', 'sglang.launch_server', '--model-path', c['model_path'],
            '--served-model-name', c['model_path'], '--host', '127.0.0.1', '--port', str(port),
            '--tp', '1', '--context-length', '32768', '--mem-fraction-static', '0.8',
            '--tool-call-parser', 'qwen3_coder', '--reasoning-parser', 'qwen3',
            '--disable-cuda-graph', '--trust-remote-code']
    proc = None
    try:
        with (out / 'worker.log').open('a') as log:
            proc = subprocess.Popen(argv, env=worker_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + 1200
            while True:
                if proc.poll() is not None:
                    raise RuntimeError('Frozen worker failed; see ' + str(out / 'worker.log'))
                try:
                    with urllib.request.urlopen(url + '/health_generate', timeout=30) as response:
                        if response.status == 200: break
                except OSError:
                    pass
                if time.monotonic() > deadline: raise TimeoutError('Worker health deadline')
                time.sleep(5)
            phase(c, env, out / 'transport_one', out / 'one_task.parquet', 3, {tid})
            phase(c, env, out / 'fresh_train_r', Path(c['prepared']) / 'calibration.parquet', 8, expected)
            atomic_json(out / 'complete.json', {'fresh_trajectories': 480, 'api_calls': 0, 'policy_updates': 0})
    finally:
        stop_owned(proc)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True)
    run(json.loads(Path(p.parse_args().config).read_text()))
