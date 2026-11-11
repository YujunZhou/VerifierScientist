"""Entry points. `run` always requires user/Claude approval; others use no GPU."""
import argparse,json,hashlib,os,subprocess
from pathlib import Path
from .core import atomic_json,verify_artifact
from .client import OpenAIClient
from .data import ROOT,prepare,diagnose
from .optimize import optimize
from .diagnostics import controlled_diagnosis
from .pipeline import approval_digest


def validate_output_paths(c):
    exp=Path(c['experiment_dir']).resolve()
    if not exp.is_relative_to((ROOT/'VerifierScientist/experiments').resolve()):raise ValueError('Outputs must stay in new project experiments')
    paths=[c.get('fixtures_root',str(exp/'fixtures')),c['run_dir'],c['prepared'],c['initial_artifact'],c['api']['cache_dir'],c['api']['budget_file']]
    if any(not Path(p).resolve().is_relative_to(exp) for p in paths):raise ValueError('Refusing writes outside this new experiment, including old results')


def review_files(c,include_fixtures=True):
    code=Path(c['code_root']);site=ROOT/'envs/verl08/lib/python3.11/site-packages'
    files=[Path(c['experiment_dir'])/'control/queue_wait.sh']+list((code/'guidance_rl').glob('*.py'))+list((code/'ha_verl').rglob('*.py'))+list((code/'configs').glob('*.yaml'))
    files+=[p for pattern in ('*.py','*.sh','*.qsub') for p in (Path(c['experiment_dir'])/'control').glob(pattern)]
    files+=list((code/'swe_provision').glob('*.py'))
    files+=list((ROOT/'envs/swebench_src/swebench/harness/constants').glob('*.py'))
    files+=list((ROOT/'HarnessAudit-main/multi_agent').rglob('*.py'))+[code/'verifier_bench/common.py']
    files+=[site/p for p in ('verl/experimental/agent_loop/tool_agent_loop.py','verl/experimental/agent_loop/agent_loop.py','verl/experimental/agent_loop/tool_parser.py','verl/trainer/ppo/ray_trainer.py','verl/workers/engine_workers.py')]
    files+=[site/p for p in ('verl/trainer/main_ppo.py','verl/trainer/constants_ppo.py',
        'verl/utils/checkpoint/fsdp_checkpoint_manager.py','verl/utils/checkpoint/checkpoint_manager.py')]
    files+=list((site/'verl/trainer/config').rglob('*.yaml'))
    inputs=list(Path(c['prepared']).glob('*.parquet'))+list(Path(c['prepared']).glob('*.yaml'))+[Path(c['prepared'])/'split.json',Path(c['tool_config']),Path(c['prepared'])/'initial_records.json',Path(c['initial_artifact']),Path(c['model_path'])/'config.json',Path(c['experiment_dir'])/'control/user_authorization.json']
    if c.get('planned_stage_budget',{}).get('cost_audit'):
        inputs.append(Path(c['planned_stage_budget']['cost_audit']))
    if c.get('runtime_bin'):
        inputs.extend(Path(c['runtime_bin'])/name for name in ('ninja','python','python3'))
    if c.get('resume_from'):
        from .training import resume_manifest_path
        inputs.extend([Path(c['resume_from']['artifact']),Path(c['resume_from']['source_attempt'])/'checkpoint_identity.json',
                       resume_manifest_path(c)])
    inputs+=list((ROOT/'HarnessAudit-main/multi_agent').rglob('*.yaml'))+list((ROOT/'HarnessAudit-main/multi_agent').rglob('*.yml'))
    inputs+=list((code/'swe_provision').glob('instances.json'))+list((Path(c['experiment_dir'])/'audit').glob('fixture_recovery_inputs.json'))
    inputs += [p for p in (ROOT/'HarnessAudit-main/multi_agent/fixtures').rglob('*') if p.is_file()]
    if include_fixtures:
        fixture_root=Path(c['fixtures_root'])
        inputs.append(fixture_root/'restore_manifest.json')
        restored=json.loads((fixture_root/'restore_manifest.json').read_text())
        required=restored['required_fixtures']
        for name in required:
            repo=fixture_root/name
            listed=subprocess.run(['git','-C',str(repo),'ls-files','-z'],check=True,capture_output=True).stdout.decode().split('\0')
            inputs.extend(repo/name for name in listed if name)
            environment=restored['fixtures'][name]['environment']
            inputs.extend(Path(environment[key]) for key in ('launcher_python','launcher_pip','pip_freeze'))
            # Do not hash mutable .git/worktrees or bytecode/venv caches.

    inputs+=[p for pattern in ('*config*.json','tokenizer*','chat_template*','*.safetensors.index.json') for p in Path(c['model_path']).glob(pattern) if p.is_file()]
    import pandas as pd
    for data in ('train_all.parquet','test_all.parquet'):
        inputs.extend(Path(info['task_yaml']) for info in pd.read_parquet(Path(c['prepared'])/data).extra_info)
    return sorted({str(p.resolve()) for p in files}),sorted({str(p.resolve()) for p in inputs})


def fixture_problems(root,required):
    """Validate the committed source and immutable runtime evidence used by worktrees."""
    root=Path(root).resolve();problems=[]
    try:
        restored=json.loads((root/'restore_manifest.json').read_text())
    except (OSError,ValueError) as exc:
        return ['restore_manifest_unreadable:'+type(exc).__name__]
    listed=restored.get('required_fixtures',[])
    if restored.get('ready') is not True:problems.append('restore_manifest_not_ready')
    if set(listed)!=set(required) or len(listed)!=len(set(listed)):problems.append('required_fixture_set_mismatch')
    for name in sorted(required):
        if not isinstance(name,str) or Path(name).name!=name or name in ('.','..'):
            problems.append('invalid_fixture_name');continue
        repo=root/name;entry=restored.get('fixtures',{}).get(name,{})
        for key in ('source_restored','ready','validation_complete'):
            if entry.get(key) is not True:problems.append(f'{name}:{key}_not_passed')
        environment=entry.get('environment',{})
        for key in ('available','collection_passed','runtime_validated'):
            if environment.get(key) is not True:problems.append(f'{name}:environment.{key}_not_passed')
        if repo.is_symlink() or repo.resolve().parent!=root or not (repo/'.git').is_dir() or not os.access(repo/'.git',os.R_OK|os.W_OK|os.X_OK):
            problems.append(f'{name}:repository_missing_unwritable_or_external');continue
        def git(*args):
            return subprocess.run(['git','-C',str(repo),*args],check=True,capture_output=True,
                                  text=True,timeout=30,env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'}).stdout.strip()
        try:
            if git('status','--porcelain','--untracked-files=no'):problems.append(f'{name}:tracked_or_staged_changes')
            for key,ref in (('git_head','HEAD'),('git_tree','HEAD^{tree}')):
                if git('rev-parse',ref)!=entry.get(key):problems.append(f'{name}:{key}_mismatch')
            marker=git('show','HEAD:.harnessaudit_testenv')
            if not marker or marker!=environment.get('marker') or marker!=environment.get('launcher_python'):
                problems.append(f'{name}:committed_marker_mismatch')
            for key in ('launcher_python','launcher_pip','pip_freeze'):
                value=environment.get(key)
                path=Path(value) if isinstance(value,str) and value else None
                mode=os.R_OK if key=='pip_freeze' else os.R_OK|os.X_OK
                if path is None or not path.is_absolute() or not path.is_file() or not os.access(path,mode):
                    problems.append(f'{name}:environment.{key}_missing_or_unusable')
            if environment.get('launcher_python') and environment.get('launcher_pip') and Path(environment['launcher_python']).with_name('pip')!=Path(environment['launcher_pip']):
                problems.append(f'{name}:adjacent_pip_mismatch')
        except (OSError,subprocess.SubprocessError) as exc:
            problems.append(f'{name}:git_validation_failed:{type(exc).__name__}')
    return problems


def preflight(c):
    validate_output_paths(c)
    import pandas as pd,yaml
    tool=yaml.safe_load(Path(c['tool_config']).read_text())['tools'];names=set()
    for t in tool:
        if t['class_name']!='ha_verl.tool_v09.HarnessAuditToolV09':raise ValueError('Exact episode pairing requires v09 for every tool')
        name=t['tool_schema']['function']['name']
        if name in names:raise ValueError('Duplicate tool schema '+name)
        names.add(name)
    counts={}
    for name in ('train_all','test_all','calibration'):
        df=pd.read_parquet(Path(c['prepared'])/(name+'.parquet'));counts[name]=len(df)
        if not (df.agent_name=='guidance_tool_agent').all():raise ValueError('Wrong agent loop in dataset')
        for info in df.extra_info:
            if not set(info['tool_selection'])<=names:raise ValueError('Tool registry missing per-task selection')
    if counts!={'train_all':178,'test_all':60,'calibration':60}:raise ValueError('Unexpected task counts')
    if not Path(c['python']).is_file() or not (Path(c['model_path'])/'config.json').is_file():raise ValueError('Runtime/model missing')
    required=set()
    for dataset in ('train_all','test_all'):
        for info in pd.read_parquet(Path(c['prepared'])/(dataset+'.parquet')).extra_info:
            fixture=yaml.safe_load(Path(info['task_yaml']).read_text()).get('fixture')
            if fixture:required.add(fixture)
    problems=fixture_problems(c['fixtures_root'],required)
    report={'ok':False,'datasets':counts,'tool_count':len(names),'gpu_tested':False,'required_fixtures':sorted(required),'fixture_problems':problems}
    if problems:
        atomic_json(Path(c['experiment_dir'])/'audit/offline_preflight.json',report)
        raise ValueError('Required fixture source/runtime restoration incomplete; training is blocked, tasks are not silently dropped')
    report['ok']=True
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','diagnose','controlled','optimize','preflight','review-manifest','fixed-eval','run'])
    p.add_argument('--config',required=True);p.add_argument('--approval');a=p.parse_args()
    c=json.loads(Path(a.config).read_text())
    if a.command=='fixed-eval':
        from .fixed_evaluation import validate_cli_config,run_evaluation
        from .pipeline import require_approval
        validate_cli_config(c)
        if not a.approval:raise ValueError('Fixed evaluation requires its own reviewed configuration approval')
        require_approval(c,a.approval)
        print(json.dumps(run_evaluation(c,OpenAIClient(c['api']))))
        return
    validate_output_paths(c);out=Path(c['experiment_dir'])
    if a.command=='prepare':prepare(c['prepared']);return
    if a.command=='diagnose':diagnose(out/'audit/historical_diagnosis.json');return
    if a.command=='preflight':
        result=preflight(c);atomic_json(out/'audit/offline_preflight.json',result);print(json.dumps(result));return
    if a.command in ('controlled','optimize'):
        records=json.loads((Path(c['prepared'])/'initial_records.json').read_text());split=json.loads((Path(c['prepared'])/'split.json').read_text());client=OpenAIClient(c['api'])
        if a.command=='controlled':controlled_diagnosis(records,split,c['optimization'],client,out/'controlled_diagnosis')
        else:optimize(records,split,c['optimization'],client,out/'initial_optimization',0,'historical_pool_before_new_router')
        return
    if a.command=='review-manifest':
        # Prepare the concrete review package, not the approval itself.
        preflight(c)
        if c.get('resume_from'):
            from .training import record_resume_manifest
            record_resume_manifest(c)
        c['code_files'],c['review_inputs']=review_files(c)
        atomic_json(a.config,c)
        manifest=approval_digest(c);atomic_json(out/'audit/review_manifest.json',{'manifest':manifest,'code_files':c['code_files'],'review_inputs':c['review_inputs']})
        print(json.dumps({'manifest':manifest,'approval_created':False}));return
    if a.command=='run':
        if not a.approval:raise ValueError('User review approval path is required')
        if c.get('startup_stage')=='allocation_supervisor':
            from .allocation import run
            run(c,a.approval)
            return
        files,inputs=review_files(c)
        if files!=c.get('code_files') or inputs!=c.get('review_inputs'):raise ValueError('Review inventory incomplete; regenerate manifest and review')
        preflight(c)
        from .training import run
        run(c,a.approval)

if __name__=='__main__':main()
