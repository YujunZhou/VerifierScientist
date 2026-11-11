"""Keep the user's existing allocation alive while bounded commands are repaired."""
import json
import hashlib
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from .core import atomic_json,digest
from .pipeline import require_approval


def task_fingerprint(task):
    files=task.get('input_files')
    if not isinstance(files,list) or not files or any(not isinstance(p,str) or not Path(p).is_absolute() for p in files):
        raise ValueError('Task must declare absolute reviewed input files')
    return digest({'task':task,'files':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sorted(set(files))}})


def require_task_approval(task,config):
    argv=task.get('argv')
    if not isinstance(argv,list) or not argv or any(not isinstance(a,str) for a in argv):
        raise ValueError('Expected explicit command argv')
    if task.get('operation') not in ('train','calibration','diagnostic'):
        raise ValueError('Task operation must be explicit')
    timeout=task.get('timeout_seconds')
    if isinstance(timeout,bool) or not isinstance(timeout,(int,float)) or not math.isfinite(timeout) or timeout<=0:
        raise ValueError('Explicit finite positive task timeout required')
    if not task.get('cwd') or not Path(task['cwd']).is_absolute():raise ValueError('Absolute task cwd required')
    job=os.getenv('JOB_ID')
    if not job:raise ValueError('Task must run in an identified allocation')
    approval=json.loads(Path(task['approval']).read_text())
    if approval.get('approved') is not True or not approval.get('approved_by'):
        raise ValueError('Per-task review approval missing or revoked')
    if approval.get('job_id')!=job or approval.get('operation')!=task['operation']:
        raise ValueError('Approval belongs to another allocation or operation')
    hold_path=config.get('research_hold') or str(Path(config['experiment_dir'])/'control/research_hold.json')
    # Missing/malformed hold state is an error, not permission to train.
    hold=json.loads(Path(hold_path).read_text())
    if task['operation']=='train' and (hold.get('active') is not False or hold.get('training_allowed') is not True):
        raise ValueError('Research hold prohibits training')
    if approval.get('task_fingerprint')!=task_fingerprint(task):
        raise ValueError('Task command, environment, or reviewed input changed')


def execute_task(path,output,logs,config):
    """Reject before Popen; task failures never terminate the allocation loop.

    This binds reviewed bytes, not a sandbox against a malicious same-user author.
    A reviewer must inspect operation declarations and the complete command.
    """
    if output.exists():return
    started=time.time();proc=None
    try:
        task=json.loads(path.read_text());require_task_approval(task,config)
    except Exception as error:
        atomic_json(output,{'status':'rejected','error':f'{type(error).__name__}: {error}',
                    'started':started,'finished':time.time(),'process_started':False})
        return
    try:
        logs.mkdir(parents=True,exist_ok=True)
        with (logs/(path.stem+'.log')).open('a') as log:
            proc=subprocess.Popen(task['argv'],cwd=task['cwd'],env=dict(os.environ,**task.get('env',{})),
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            rc=proc.wait(timeout=task['timeout_seconds'])
        result={'status':'complete' if rc==0 else 'failed','returncode':rc,'started':started,'finished':time.time()}
    except BaseException as error:
        if proc is not None and proc.poll() is None:
            try:os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                proc.wait()
        result={'status':'failed','error':f'{type(error).__name__}: {error}','started':started,'finished':time.time()}
        if not isinstance(error,Exception):
            atomic_json(output,{**result,'status':'interrupted'})
            raise
    atomic_json(output,result)


def run(c, approval):
    require_approval(c, approval)
    def terminated(signum,frame):raise KeyboardInterrupt('Allocation supervisor termination')
    oldterm=signal.signal(signal.SIGTERM,terminated)
    try:_allocation_loop(c)
    finally:signal.signal(signal.SIGTERM,oldterm)


def _allocation_loop(c):
    root = Path(c['experiment_dir']) / 'gpu_control'
    for name in ('tasks', 'results', 'logs'):
        (root / name).mkdir(parents=True, exist_ok=True)
    atomic_json(root / 'allocation.json', {'job_id': os.getenv('JOB_ID'),
        'host': os.uname().nodename, 'cuda_visible_devices': os.getenv('CUDA_VISIBLE_DEVICES'),
        'sge_gpu': os.getenv('SGE_HGR_gpu'), 'started': time.time(),
        'policy': 'Retain allocation on task failure; release only on explicit request or scheduler termination'})
    while True:
        for path in sorted((root / 'tasks').glob('*.json')):
            output = root / 'results' / path.name
            if output.exists():
                continue
            started = time.time()
            atomic_json(root / 'active.json', {'task': path.name, 'started': started})
            execute_task(path,output,root/'logs',c)
            atomic_json(root / 'active.json', {'task': None, 'last_finished': path.name})
        time.sleep(5)
