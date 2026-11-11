"""Four-GPU segmented GRPO runner. No launches without an exact review manifest."""
from __future__ import annotations
import hashlib,json,math,os,signal,socket,subprocess,sys,time,urllib.request
from pathlib import Path
from .core import atomic_json,digest,verify_artifact
from .client import OpenAIClient,file_lock
from .data import load_calibration
from .optimize import optimize,JUDGE_PROMPT,ALIGNED_JUDGE_PROMPT
from .pipeline import require_approval,assigned_gpus,approval_digest
from .phase_budget import check_phase_budget,check_execution_budget


def _file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def training_source_contract(c):
    """Canonical policy/trainer settings, excluding only phase bookkeeping.

    The readable override map and configuration hashes explain why a future
    recovery is rejected; changing LR, batch, KL, seed or a YAML default changes
    this contract even if the model/data paths stay the same.
    """
    argv=trainer_command(c,'train',1,Path('/__checkpoint_contract__/phase'))
    dynamic={'trainer.total_training_steps','trainer.experiment_name','trainer.resume_mode',
        'trainer.resume_from_path','trainer.rollout_data_dir','trainer.validation_data_dir',
        'trainer.default_local_dir','trainer.logger'}
    overrides={}
    for arg in argv[5:]:
        key,sep,value=arg.partition('=')
        if sep and key not in dynamic:overrides[key]=value
    code=Path(c['code_root'])
    # Include the project's Hydra entry file and installed Hydra defaults. Do not
    # hash mutable logs, results or the current approval document.
    configs=list((code/'configs').rglob('*.yaml'))
    installed=Path(c.get('project_root',code.parent))/'envs/verl08/lib/python3.11/site-packages/verl/trainer/config'
    if installed.is_dir():configs.extend(installed.rglob('*.yaml'))
    for path in (Path(c['tool_config']),Path(c['prepared'])/'agent_loop.yaml'):
        if path.is_file():configs.append(path)
    return {'entrypoint':argv[2],'config_name':argv[4],'overrides':dict(sorted(overrides.items())),
            'config_files':{str(p.resolve()):_file_sha(p) for p in sorted(set(configs))}}


def attempt_identity(c,artifact,start,end):
    """Write before a trainer starts, so partial-stage checkpoints have identity."""
    a=verify_artifact(json.loads(Path(artifact).read_text()))
    contract=training_source_contract(c)
    return {'schema_version':1,'start_step':start,'end_step':end,
        'artifact_id':a['artifact_id'],'score_protocol':a.get('score_protocol','legacy_v1'),
        'artifact':str(Path(artifact).resolve()),'model_path':str(Path(c['model_path']).resolve()),
        'model_config_sha256':_file_sha(Path(c['model_path'])/'config.json'),
        'train_data_sha256':_file_sha(Path(c['prepared'])/'train_all.parquet'),
        'split_sha256':_file_sha(Path(c['prepared'])/'split.json'),'seed':c.get('seed',1),
        'source_contract':contract,'source_contract_sha256':digest(contract)}


def _checkpoint_content_manifest(checkpoint,source):
    p=validate_checkpoint(checkpoint);source=Path(source).resolve()
    identity_path=source/'checkpoint_identity.json'
    if identity_path.is_symlink():raise ValueError('Source checkpoint identity must not be a symlink')
    identity=json.loads(identity_path.read_text())
    if not identity.get('source_manifest') or identity.get('source_contract_sha256')!=digest(identity.get('source_contract')):
        raise ValueError('Source attempt lacks an approved stable training contract')
    step=int(p.name.removeprefix('global_step_'))
    if not identity['start_step']<step<=identity['end_step']:raise ValueError('Checkpoint outside source attempt range')
    files=['data.pt']+[f'actor/{kind}_world_size_3_rank_{rank}.pt'
                       for kind in ('model','optim','extra_state') for rank in range(3)]
    return {'schema_version':1,'step':step,'checkpoint':str(p.resolve()),
        'source_attempt':str(source),'source_identity_sha256':_file_sha(identity_path),
        'source_contract_sha256':identity['source_contract_sha256'],
        'artifact_id':identity['artifact_id'],'score_protocol':identity['score_protocol'],
        'files':{name:{'sha256':_file_sha(p/name),'bytes':(p/name).stat().st_size} for name in files}}


def verify_checkpoint_content(checkpoint,source):
    manifest=Path(checkpoint)/'checkpoint_manifest.json'
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError('Missing checkpoint content manifest; old checkpoints require separate reviewed verification')
    saved=json.loads(manifest.read_text())
    if saved!=_checkpoint_content_manifest(checkpoint,source):raise ValueError('Checkpoint content manifest mismatch')
    return saved


def check_checkpoint_space(c):
    """Reserve room for one next checkpoint plus metadata/log headroom.

    This is a live filesystem check, not a quota reservation or guarantee that
    the entire future run fits. Existing complete checkpoints are never removed
    to force this check to pass.
    """
    import shutil
    root=Path(c['run_dir']);latest=root/'latest_complete.json'
    if latest.exists():
        previous=json.loads(latest.read_text());size=previous['checkpoint_bytes']
    else:
        # bf16 model files -> fp32 model plus two fp32 Adam moments, upper proxy.
        size=6*sum(p.stat().st_size for p in Path(c.get('model_path',root)).glob('*.safetensors'))
    free=shutil.disk_usage(root).free;required=size+1024**3
    decision={'free_bytes':free,'next_checkpoint_bytes':size,'required_bytes':required,'accepted':free>=required}
    atomic_json(root/'checkpoint_space.json',decision)
    if not decision['accepted']:raise ValueError('Insufficient disk for next checkpoint; existing latest and milestones retained')
    return decision


def publish_complete_checkpoint(c,attempt,step):
    """Seal a completed save, archive seven-step milestones, publish latest last."""
    import shutil,tempfile
    root=Path(c['run_dir']).resolve();attempt=Path(attempt).resolve()
    if not attempt.is_relative_to(root):raise ValueError('Checkpoint attempt escapes run')
    checkpoint=root/'checkpoints'/f'global_step_{step}'
    if checkpoint.parent.is_symlink():raise ValueError('Checkpoint directory is a symlink')
    manifest=_checkpoint_content_manifest(checkpoint,attempt)
    seal=checkpoint/'checkpoint_manifest.json'
    if seal.is_symlink():raise ValueError('Checkpoint manifest is a symlink')
    if seal.exists() and json.loads(seal.read_text())!=manifest:raise ValueError('Refusing changed checkpoint contents')
    if not seal.exists():atomic_json(seal,manifest)
    latest=root/'latest_complete.json'
    if latest.exists() and json.loads(latest.read_text())['step']>step:raise ValueError('Refusing to regress latest checkpoint')
    published=checkpoint
    if step in (7,14,21,28):
        milestones=root/'milestones'
        if milestones.is_symlink():raise ValueError('Milestone directory is a symlink')
        milestones.mkdir(exist_ok=True);destination=milestones/checkpoint.name
        if destination.is_symlink():raise ValueError('Milestone checkpoint is a symlink')
        if destination.exists():
            existing=verify_checkpoint_content(destination,attempt)
            if existing['files']!=manifest['files']:raise ValueError('Existing milestone differs from checkpoint')
        else:
            if any(p.is_symlink() for p in checkpoint.rglob('*')):raise ValueError('Checkpoint tree contains symlink')
            if shutil.disk_usage(root).free<1024**2:raise ValueError('Insufficient metadata headroom for milestone')
            temporary=Path(tempfile.mkdtemp(prefix='.'+checkpoint.name+'-',dir=milestones))
            try:
                # Same-filesystem hard links retain complete immutable checkpoint
                # bytes when veRL removes their rolling names. No copy fallback.
                for item in checkpoint.iterdir():
                    if item.name=='checkpoint_manifest.json':continue
                    if item.is_dir():shutil.copytree(item,temporary/item.name,copy_function=os.link)
                    else:os.link(item,temporary/item.name)
                atomic_json(temporary/'checkpoint_manifest.json',dict(manifest,checkpoint=str(destination)))
                os.replace(temporary,destination)
            finally:
                if temporary.exists():shutil.rmtree(temporary)
            verify_checkpoint_content(destination,attempt)
        published=destination
    current=verify_checkpoint_content(published,attempt)
    result={'step':step,'checkpoint':str(published),'source_attempt':str(attempt),
        'artifact_id':current['artifact_id'],'score_protocol':current['score_protocol'],
        'manifest_sha256':_file_sha(published/'checkpoint_manifest.json'),
        'checkpoint_bytes':sum(v['bytes'] for v in current['files'].values()),
        'milestone':step in (7,14,21,28),'verified_complete':True}
    atomic_json(latest,result)
    # A fresh veRL manager knows only checkpoints saved in its own process.
    # Rotate globally after publication so segment boundaries cannot accumulate
    # four rolling checkpoints. Durable milestones are in a different directory.
    removed=prune_checkpoints(root/'checkpoints',keep=2)
    atomic_json(root/'checkpoint_retention_latest.json',{'step':step,'removed':removed,'keep':2})
    return result


def seal_attempt_checkpoints(c,attempt):
    """Called only after this trainer exits, including a failed/partial stage.

    Hash all ten required files of structurally complete checkpoints. Never
    overwrite an existing manifest whose content/attempt binding differs. This
    function is not invoked by resume and is not an old-checkpoint migration.
    """
    root=Path(c['run_dir']).resolve();attempt=Path(attempt).resolve()
    if not attempt.is_relative_to(root):raise ValueError('Source attempt must belong to this run')
    identity=json.loads((attempt/'checkpoint_identity.json').read_text())
    report={'source_attempt':str(attempt),'sealed':[],'incomplete':[]}
    for p in sorted((root/'checkpoints').glob('global_step_*')):
        name=p.name.removeprefix('global_step_')
        if not name.isdigit() or not identity['start_step']<int(name)<=identity['end_step']:continue
        if p.is_symlink() or p.resolve().parent!=root/'checkpoints':raise ValueError('Checkpoint escapes this run')
        try:validate_checkpoint(p)
        except ValueError:
            report['incomplete'].append(p.name);continue
        expected=_checkpoint_content_manifest(p,attempt);manifest=p/'checkpoint_manifest.json'
        if manifest.is_symlink():raise ValueError('Checkpoint content manifest is a symlink')
        if manifest.exists():
            if json.loads(manifest.read_text())!=expected:raise ValueError('Refusing to overwrite mismatched checkpoint content manifest')
        else:atomic_json(manifest,expected)
        report['sealed'].append(p.name)
    atomic_json(attempt/'checkpoint_sealing.json',report)
    return report


def resume_manifest_path(c):
    """Durable review copy of the resume checkpoint's sealed content manifest.

    Rolling checkpoints under run/checkpoints are rotated by veRL and by
    prune_checkpoints while a continuation runs, so an approval that hashed
    run/checkpoints/global_step_N/checkpoint_manifest.json becomes
    unverifiable mid-run (task 013 failed there after sealing step 7).
    Milestones are never rotated and are referenced directly.
    """
    root=Path(c['run_dir']).resolve();p=Path(c['resume_from']['checkpoint'])
    if p.resolve().parent==root/'milestones':return p.resolve()/'checkpoint_manifest.json'
    return root/'resume_manifests'/(p.name+'.checkpoint_manifest.json')


def record_resume_manifest(c):
    """Copy the live sealed manifest to its durable review path; never overwrite a different one."""
    live=Path(c['resume_from']['checkpoint'])/'checkpoint_manifest.json';durable=resume_manifest_path(c)
    if live.is_symlink() or not live.is_file():raise ValueError('Resume checkpoint lacks a sealed content manifest')
    current=json.loads(live.read_text())
    if durable.is_symlink():raise ValueError('Durable resume manifest must not be a symlink')
    if durable.exists():
        if json.loads(durable.read_text())!=current:raise ValueError('Durable resume manifest differs from the live checkpoint manifest')
    else:
        durable.parent.mkdir(parents=True,exist_ok=True);atomic_json(durable,current)
    return durable


def resume_state(c):
    """Resolve only an explicitly reviewed checkpoint; never select latest."""
    spec=c.get('resume_from')
    if spec is None:return None
    if not isinstance(spec,dict):raise ValueError('resume_from must explicitly identify checkpoint, artifact and source_attempt')
    root=Path(c['run_dir']).resolve();p=Path(spec['checkpoint'])
    if p.is_symlink() or p.parent.is_symlink() or p.resolve().parent not in (root/'checkpoints',root/'milestones'):
        raise ValueError('Resume checkpoint must belong to this run rolling or milestone directory')
    p=p.resolve();name=p.name.removeprefix('global_step_')
    if p.name!='global_step_'+name or not name.isdigit():raise ValueError('Invalid checkpoint step')
    step=int(name);final=c.get('final_step',28)
    if not isinstance(final,int) or isinstance(final,bool) or not 0<step<final:
        raise ValueError('Resume step must precede final_step; restore-only is a separate non-updating validation')
    validate_checkpoint(p);ensure_new_training_range(root/'checkpoints',step)
    source=Path(spec['source_attempt'])
    if source.is_symlink() or not source.resolve().is_relative_to(root):raise ValueError('Source attempt must belong to this run')
    old=json.loads((source/'checkpoint_identity.json').read_text())
    if not old.get('source_manifest'):raise ValueError('Source attempt lacks its original approval identity')
    if not isinstance(old.get('start_step'),int) or not isinstance(old.get('end_step'),int) or not old['start_step']<step<=old['end_step']:
        raise ValueError('Checkpoint is outside its source attempt step range')
    a=verify_artifact(json.loads(Path(spec['artifact']).read_text()))
    if a['artifact_id']!=spec['artifact_id'] or a.get('score_protocol','legacy_v1')!=spec['score_protocol']:
        raise ValueError('Explicit resume artifact identity mismatch')
    if spec['score_protocol']!=c['optimization'].get('score_protocol','legacy_v1') or a['judge_model']!=c['optimization']['judge_model']:
        raise ValueError('Resume changes judge model or score protocol')
    expected=attempt_identity(c,spec['artifact'],old['start_step'],old['end_step'])
    if any(old.get(k)!=v for k,v in expected.items()):raise ValueError('Resume source model, data, artifact or seed identity mismatch')
    saved=verify_checkpoint_content(p,source)
    durable=resume_manifest_path(c)
    if durable.is_symlink() or not durable.is_file() or json.loads(durable.read_text())!=saved:
        raise ValueError('Reviewed durable copy of the resume checkpoint manifest is missing or differs; rerun review-manifest')
    # The new configuration has its own approval. Its manifest intentionally need
    # not equal the old manifest, since resume_from and final_step have changed.
    return {'step':step,'checkpoint':str(p),'artifact':str(Path(spec['artifact']).resolve()),
        'artifact_id':a['artifact_id'],'score_protocol':spec['score_protocol'],'source_attempt':str(source.resolve())}


def execution_plan(c,state=None):
    final=c.get('final_step',28)
    if isinstance(final,bool) or final not in (1,7,14,21,28):raise ValueError('final_step must be an explicit checkpoint boundary')
    start=state['step'] if state else 0;steps=[]
    def boundary(end):
        steps.append({'stage':f'test_{end:02d}','mode':'test','start':end,'end':end})
        if c.get('arm')=='adaptive' and (end<final or c.get('optimize_at_final',False)):
            if end//7>=c['optimization'].get('max_rounds',4):raise ValueError('Optimization exceeds configured round limit')
            steps.extend([{'stage':f'calibration_{end:02d}','mode':'calibration','start':end,'end':end},
                          {'stage':f'optimization_{end:02d}','mode':'optimization','start':end,'end':end,'round':end//7}])
    if not state:
        steps.extend([{'stage':'test_00','mode':'test','start':0,'end':1},
                      {'stage':'train_01_smoke','mode':'train','start':0,'end':1}])
        start=1
    elif start in (7,14,21):
        # Explicit continuation re-evaluates this boundary in a new namespace;
        # it does not repeat a policy update or silently reuse old optimization.
        boundary(start)
    for end in (7,14,21,28):
        if start<end<=final:
            steps.append({'stage':f'train_{end:02d}','mode':'train','start':start,'end':end})
            boundary(end);start=end
    return steps


def paid_stage_requirements(c,stage):
    retries=c['api'].get('max_transport_retries',0)
    if isinstance(retries,bool) or not isinstance(retries,int) or retries not in (0,1):raise ValueError('Plan supports zero or one transport retry')
    opt=c['optimization'];models={opt['judge_model']}
    if stage['mode']=='train':
        reward_reps=c.get('reward_judge_replicates',1)
        if isinstance(reward_reps,bool) or not isinstance(reward_reps,int) or reward_reps<1:raise ValueError('Invalid reward judge replicate count')
        logical=(stage['end']-stage['start'])*24*8*reward_reps
    elif stage['mode']=='optimization':
        names=set(opt.get('candidate_names',['expand_same_meaning','compress_same_meaning','plain_format',
            'remove_exceptions','remove_evidence','error_repair']))
        specs=names-{'proposal_seed'};equivalent=specs&{'expand_same_meaning','compress_same_meaning','plain_format'}
        reps=opt.get('judge_replicates',3 if opt.get('score_protocol')=='aligned_v2' else 2)
        if isinstance(reps,bool) or not isinstance(reps,int) or reps<1:raise ValueError('Invalid judge replicate count')
        seed_candidate=int(bool(opt.get('baseline_start')) and ('candidate_names' not in opt or 'proposal_seed' in names))
        draws=opt.get('repair_draws',1)
        if isinstance(draws,bool) or not isinstance(draws,int) or draws<1:raise ValueError('Invalid repair draw count')
        extra_draws=(draws-1) if 'error_repair' in specs else 0
        candidates=2+len(specs)+seed_candidate+extra_draws
        # Fresh calibration: 30/10/20 tasks, eight trajectories each. Count all
        # candidates, possible confirmation against all references, and generator
        # /semantic-auditor calls, even when some might be rejected or cache hits.
        logical=30*8+10*8*reps*candidates+20*8*reps*(3 if opt.get('score_protocol')=='aligned_v2' else 2)
        logical+=1+len(specs)+len(equivalent)+extra_draws
        models.add(opt['generator_model'])
        if equivalent:models.add(opt.get('semantic_auditor_model',opt['generator_model']))
    else:return None
    return {'stage':stage['stage'],'required_calls':logical*(1+retries),
            'logical_calls':logical,'max_transport_retries':retries,'models':sorted(models)}


def validate_checkpoint(path):
    p=Path(path);actor=p/'actor'
    if p.is_symlink() or actor.is_symlink() or (p/'data.pt').is_symlink():raise ValueError('Checkpoint symlinks are not resumable')
    if not (p/'data.pt').is_file() or not actor.is_dir():raise ValueError('Incomplete full checkpoint: '+str(p))
    if (p/'data.pt').stat().st_size==0:raise ValueError('Empty data checkpoint')
    for kind in ('model','optim','extra_state'):
        for rank in range(3):
            part=actor/f'{kind}_world_size_3_rank_{rank}.pt'
            if part.is_symlink() or not part.is_file() or part.stat().st_size==0:raise ValueError('Checkpoint missing/empty/symlink shard '+str(part))
    return p


def ensure_new_training_range(directory,start):
    for p in Path(directory).glob('global_step_*'):
        if p.name.removeprefix('global_step_').isdigit() and int(p.name.removeprefix('global_step_'))>start:
            raise ValueError('Existing checkpoint beyond explicit resume point; review recovery instead of overwriting: '+str(p))


def prune_checkpoints(directory,keep=2):
    import shutil
    root=Path(directory).resolve()
    paths=[p for p in root.glob('global_step_*') if p.name.removeprefix('global_step_').isdigit()]
    if keep<2:raise ValueError('Keep at least two checkpoints for recovery')
    if any(p.is_symlink() or p.resolve().parent!=root for p in paths):raise ValueError('Refusing checkpoint symlink traversal')
    paths.sort(key=lambda p:int(p.name.removeprefix('global_step_')))
    # veRL rotates actor/ only, leaving an old data.pt shell. Never treat that as resumable.
    for p in paths[-keep:]:validate_checkpoint(p)
    for p in paths[:-keep]:
        if (p/'actor').exists():validate_checkpoint(p)
        elif set(x.name for x in p.iterdir()) in ({'data.pt'},{'data.pt','checkpoint_manifest.json'}) and (p/'data.pt').stat().st_size>0 and not (p/'checkpoint_manifest.json').is_symlink():pass
        else:raise ValueError('Refusing to remove an unexplained partial checkpoint: '+str(p))
    removed=[]
    for p in paths[:-keep]:
        removed.append(p.name);shutil.rmtree(p)
    return removed


def trainer_command(c,mode,end,phase,resume=None):
    code=Path(c['code_root']);data=Path(c['prepared']);phase=Path(phase)
    a=[c['python'],'-m','guidance_rl.training','--config-path='+str(code/'configs'),'--config-name=harnessaudit_v08']
    opts={'+ray_kwargs.ray_init.include_dashboard':False,'+ray_kwargs.ray_init.num_cpus':12,
      'data.train_files':str(data/'train_all.parquet'),'data.val_files':str(data/('test_all.parquet' if mode=='test' else 'calibration.parquet')),
      'data.train_batch_size':24,'data.shuffle':True,'data.seed':c.get('seed',1),'data.filter_overlong_prompts':False,
      'data.dataloader_num_workers':0,
      'reward.custom_reward_function.path':str(code/'guidance_rl/reward.py'),'reward.custom_reward_function.name':'compute_score',
      'algorithm.adv_estimator':'grpo','actor_rollout_ref.model.path':c['model_path'],
      '+actor_rollout_ref.model.override_config.attn_implementation':'sdpa','actor_rollout_ref.model.use_fused_kernels':True,
      'actor_rollout_ref.actor.use_dynamic_bsz':False,'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu':1,
      'actor_rollout_ref.rollout.log_prob_use_dynamic_bsz':False,'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu':1,
      'actor_rollout_ref.ref.log_prob_use_dynamic_bsz':False,'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu':1,
      'actor_rollout_ref.rollout.n':8,'actor_rollout_ref.rollout.temperature':1.0,
      'actor_rollout_ref.rollout.tensor_model_parallel_size':1,'actor_rollout_ref.rollout.gpu_memory_utilization':.30,
      'actor_rollout_ref.rollout.max_model_len':32768,'actor_rollout_ref.rollout.enforce_eager':True,
      'actor_rollout_ref.rollout.agent.default_agent_loop':'guidance_tool_agent',
      'actor_rollout_ref.rollout.agent.num_workers':4,
      'actor_rollout_ref.rollout.agent.agent_loop_config_path':str(data/'agent_loop.yaml'),
      'actor_rollout_ref.rollout.multi_turn.tool_config_path':c['tool_config'],
      'actor_rollout_ref.actor.ppo_max_token_len_per_gpu':36000,'actor_rollout_ref.actor.optim.lr':1e-6,
      'actor_rollout_ref.actor.optim.lr_scheduler_type':'constant','actor_rollout_ref.actor.optim.lr_warmup_steps_ratio':0.,
      'actor_rollout_ref.actor.use_kl_loss':True,'actor_rollout_ref.actor.kl_loss_coef':.001,'actor_rollout_ref.actor.kl_loss_type':'low_var_kl',
      'actor_rollout_ref.actor.entropy_coeff':0.,'actor_rollout_ref.actor.strategy':'fsdp2','actor_rollout_ref.actor.fsdp_config.strategy':'fsdp2',
      'actor_rollout_ref.actor.fsdp_config.offload_policy':False,'actor_rollout_ref.actor.fsdp_config.param_offload':False,
      'actor_rollout_ref.actor.fsdp_config.optimizer_offload':False,'actor_rollout_ref.ref.fsdp_config.param_offload':True,
      'actor_rollout_ref.ref.fsdp_config.model_dtype':'bf16','trainer.logger':'[console,file]',
      'trainer.project_name':'VerifierScientist','trainer.experiment_name':phase.name,
      'trainer.val_before_train':mode!='train','trainer.val_only':mode!='train','trainer.test_freq':-1,
      'trainer.rollout_data_dir':str(phase/'train_rollouts'),'trainer.validation_data_dir':str(phase/'val_rollouts'),'trainer.total_epochs':4,'trainer.total_training_steps':end,
      'trainer.save_freq':1,'trainer.max_actor_ckpt_to_keep':2,
      'trainer.default_local_dir':str(Path(c['run_dir'])/'checkpoints'),
      'trainer.resume_mode':'resume_path' if resume else 'disable','trainer.n_gpus_per_node':3,'trainer.nnodes':1,
      'actor_rollout_ref.rollout.val_kwargs.n':8 if mode=='calibration' else 1,
      'actor_rollout_ref.rollout.val_kwargs.do_sample':mode=='calibration',
      'actor_rollout_ref.rollout.val_kwargs.temperature':1.0 if mode=='calibration' else 0.}
    if resume:opts['trainer.resume_from_path']=str(resume)
    a.extend(k+'='+str(v) for k,v in opts.items());return a


def restore_epoch_boundary(loader,state,step):
    """Restore sampler RNG at an exhausted epoch without replaying that epoch.

    Installed veRL skips the entire loader state on a boundary. That loses the
    independent RandomSampler generator and repeats epoch zero on every segment.
    The installed single-process StatefulDataLoader supports a finished iterator:
    it restores the sampler state, then creates the next epoch's permutation.
    """
    if not step or step%len(loader):return False
    if loader.num_workers!=0 or '_iterator_finished' not in state:
        raise ValueError('Boundary restore requires the reviewed single-process loader state')
    state=dict(state);state['_iterator_finished']=True
    loader.load_state_dict(state)
    return True


def validate_training_attempt(attempt,start,end,artifact_id):
    """Require observable updates and complete rewards before another segment."""
    attempt=Path(attempt);updates=end-start
    rows=[json.loads(line) for line in (attempt/'metrics.jsonl').read_text().splitlines() if line]
    if [r['step'] for r in rows]!=list(range(start+1,end+1)):
        raise ValueError('Missing, duplicated or unexpected training update metrics')
    for row in rows:
        metrics=row['data']
        for key in ('actor/grad_norm','actor/lr','actor/pg_loss'):
            value=metrics.get(key)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError('Missing or nonfinite training metric: '+key)
        if metrics['actor/grad_norm']<0 or not math.isclose(metrics['actor/lr'],1e-6,rel_tol=1e-6):
            raise ValueError('Unexpected gradient norm or constant learning rate')
    if not rows or not any(r['data']['actor/grad_norm']>0 for r in rows):
        raise ValueError('No nonzero actor gradient was observed')
    rewards=[json.loads(p.read_text()) for p in (attempt/'audit').glob('*.json')]
    if len(rewards)!=updates*192 or len({r['record']['traj_id'] for r in rewards})!=len(rewards):
        raise ValueError('Training requires exactly 192 unique rewarded trajectories per update')
    for row in rewards:
        score=row.get('reward',{}).get('score')
        if row.get('status')!='complete' or row.get('artifact_id')!=artifact_id:
            raise ValueError('Pending reward or changed segment artifact')
        if isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score) or not 0<=score<=1:
            raise ValueError('Invalid training reward')
    result={'updates':updates,'reward_count':len(rewards),'start':start,'end':end,
        'artifact_id':artifact_id,'constant_lr':1e-6,'finite_update_metrics':True,
        'note':'Observed optimizer-update metrics; does not establish policy improvement.'}
    atomic_json(attempt/'update_acceptance.json',result)
    return result


def stop_owned(proc):
    if proc is None:return
    try:os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:os.killpg(proc.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        proc.wait()


def runtime_env(c):
    import shutil
    e=dict(os.environ);root=Path(c.get('project_root',Path(c['code_root']).parent))
    runtime_bin=Path(c.get('runtime_bin',''))
    if not c.get('runtime_bin') or not runtime_bin.is_dir():raise ValueError('Configured runtime_bin is required')
    for name in ('ninja','python','python3'):
        path=runtime_bin/name
        if not path.is_file() or not os.access(path,os.X_OK):raise ValueError('Missing executable in runtime_bin: '+name)
        if name.startswith('python') and path.resolve()!=Path(c['python']).resolve():
            raise ValueError('runtime_bin Python differs from the reviewed interpreter')
    e['PATH']=str(runtime_bin.resolve())+':'+e.get('PATH','')
    if not shutil.which('c++',path=e['PATH']):raise ValueError('C++ compiler unavailable for model kernels')
    try:subprocess.run([str(runtime_bin/'ninja'),'--version'],env=e,check=True,timeout=10,capture_output=True)
    except (OSError,subprocess.SubprocessError) as error:raise ValueError('runtime_bin ninja is not runnable') from error
    e['GUIDANCE_PROJECT_ROOT']=str(root)
    e['PYTHONPATH']=str(root/'envs/verl08/lib/python3.11/site-packages')+':'+c['code_root']+':'+str(root/'HarnessAudit-main')
    e['VERL_ROOT']=str(root/'envs/verl08/lib/python3.11/site-packages')
    e['PYTHONDONTWRITEBYTECODE']='1';e['PYTHONUNBUFFERED']='1';e['HARNESSAUDIT_ROOT']=str(root/'HarnessAudit-main')
    e['HA_FIXTURES_ROOT']=c['fixtures_root']
    e['HA_MODEL_NAME']='Qwen/Qwen3.5-4B';e['HF_HOME']=str(root/'workAmazon/hf')
    e['LD_LIBRARY_PATH']=str(root/'envs/verl08/lib/python3.11/site-packages/nvidia/cu13/lib')+':'+e.get('LD_LIBRARY_PATH','')
    cuda=c.get('cuda_home')
    if cuda:
        if not (Path(cuda)/'bin/nvcc').is_file():raise ValueError('Configured CUDA toolkit lacks nvcc')
        e['CUDA_HOME']=str(cuda);e['CUDA_PATH']=str(cuda)
        e['PATH']=str(Path(cuda)/'bin')+':'+e.get('PATH','')
        e['LD_LIBRARY_PATH']+=':'+str(Path(cuda)/'lib64')
    # Ray appends session timestamp/PID/socket names; Linux AF_UNIX caps 107 bytes.
    tmp=Path('/tmp')/f"g{os.getenv('JOB_ID','m')}-{os.getpid()}";tmp.mkdir(parents=True,exist_ok=True)
    for key,folder in {'TMPDIR':'tmp','TRITON_CACHE_DIR':'triton','FLASHINFER_WORKSPACE_BASE':'flashinfer','XDG_CACHE_HOME':'xdg','HF_MODULES_CACHE':'hf_modules','MASP_WORKSPACES_ROOT':'workspaces'}.items():
        p=tmp/folder;p.mkdir(exist_ok=True);e[key]=str(p)
    e['RAY_agent_register_timeout_ms']='600000';e['RAY_enable_metrics_collection']='false'
    return e


def cleanup_phase(c,attempt,mode,proc,original_error=None):
    errors=[]
    try:stop_owned(proc)
    except BaseException as error:errors.append(error)
    if mode=='train' and proc is not None:
        try:seal_attempt_checkpoints(c,attempt)
        except BaseException as error:errors.append(error)
    if errors:
        if original_error is not None:errors.insert(0,original_error)
        raise BaseExceptionGroup('Trainer/cleanup failures; inspect every enclosed exception',errors)


def execute_phase(c,mode,end,phase,resume,artifact,env,approval):
    require_approval(c,approval)
    phase.mkdir(parents=True,exist_ok=True)
    artifact_id=verify_artifact(json.loads(Path(artifact).read_text()))['artifact_id']
    phase_identity={'artifact_id':artifact_id,'mode':mode,'end':end,'resume':str(resume) if resume else None}
    mark=phase/'complete.json'
    if mark.exists():
        done=json.loads(mark.read_text())
        if done['manifest']!=approval_digest(c) or done.get('phase_identity')!=phase_identity:raise ValueError('Completed phase belongs to a different reviewed experiment')
        return
    start=int(Path(resume).name.removeprefix('global_step_')) if resume else 0
    if mode=='train':
        paid=paid_stage_requirements(c,{'stage':phase.name,'mode':mode,'start':start,'end':end})
        check_phase_budget(c['api'],c.get('planned_stage_budget',{}),phase.name,paid['required_calls'],
            phase/'budget_decision.json',protocol=c['optimization'].get('score_protocol','legacy_v1'),models=paid['models'])
    # Never append a new attempt to old sample files. Explicit attempts retain every raw trace.
    tries=sorted(phase.glob('attempt_*'));attempt=phase/f'attempt_{len(tries):03d}';attempt.mkdir()
    if resume:validate_checkpoint(resume)
    if mode=='train':
        ensure_new_training_range(Path(c['run_dir'])/'checkpoints',start)
        identity=attempt_identity(c,artifact,start,end)
        identity['source_manifest']=approval_digest(c)
        atomic_json(attempt/'checkpoint_identity.json',identity)
    rcfg={'mode':mode,'api':c['api'],'artifact':str(artifact),'audit_dir':str(attempt/'audit')}
    if mode=='train':
        rcfg['artifact_id']=verify_artifact(json.loads(Path(artifact).read_text()))['artifact_id']
        if 'reward_judge_replicates' in c:rcfg['reward_judge_replicates']=c['reward_judge_replicates']
    if mode=='calibration':
        split=json.loads((Path(c['prepared'])/'split.json').read_text());rcfg['calibration_ids']=split['propose']+split['select']+split['confirm']
    atomic_json(attempt/'reward_config.json',rcfg)
    e=dict(env);e['GUIDANCE_REWARD_CONFIG']=str(attempt/'reward_config.json');e['HA_EPISODE_DIR']=str(attempt/'episodes');e['HA_RESULTS_DIR']=str(attempt/'results')
    e['VERL_FILE_LOGGER_PATH']=str(attempt/'metrics.jsonl')
    if mode=='train':
        context=attempt/'checkpoint_context.json'
        atomic_json(context,{'run_dir':c['run_dir'],'model_path':c['model_path'],'attempt':str(attempt)})
        e['GUIDANCE_CHECKPOINT_CONTEXT']=str(context)
    e['RAY_TMPDIR']=str(Path(e['TMPDIR']).parent/('r'+digest(str(attempt))[:6]))
    cmd=trainer_command(c,mode,end,attempt,resume);atomic_json(attempt/'command.json',{'argv':cmd,'mode':mode,'end':end,'resume':str(resume) if resume else None})
    proc=None
    with (attempt/'driver.log').open('a') as log:
        try:
            proc=subprocess.Popen(cmd,env=e,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=c['code_root'])
            if proc.wait()!=0:raise RuntimeError('Trainer failed; see '+str(attempt/'driver.log'))
        finally:
            cleanup_phase(c,attempt,mode,proc,sys.exception())
    if mode=='train':
        validate_checkpoint(Path(c['run_dir'])/'checkpoints'/f'global_step_{end}')
        validate_training_attempt(attempt,start,end,artifact_id)
    elif mode=='calibration':load_calibration(attempt/'audit',split,n=8)
    else:
        files=list((attempt/'audit').glob('*.json'))
        import pandas as pd
        expected={i['task_id'] for i in pd.read_parquet(Path(c['prepared'])/'test_all.parquet').extra_info}
        actual=[json.loads(p.read_text())['record']['task_id'] for p in files]
        if len(actual)!=60 or set(actual)!=expected:raise ValueError('Expected one output per Test-60 task')
    atomic_json(mark,{'attempt':str(attempt),'artifact':str(artifact),'checkpoint_step':end,'manifest':approval_digest(c),'phase_identity':phase_identity})
    if mode=='train':
        removed=prune_checkpoints(Path(c['run_dir'])/'checkpoints')
        atomic_json(phase/'checkpoint_retention.json',{'removed_completed_checkpoints':removed,'keep':2,'raw_rollouts_retained':True})


def run(c,approval):
    require_approval(c,approval)
    if c.get('required_transport_probe'):
        proof=json.loads(Path(c['required_transport_probe']).read_text())
        if not proof.get('reward_transport_verified') or proof.get('n_trajectories',0)<3:
            raise ValueError('Actual GPU reward transport probe has not passed')
    run_dir=Path(c['run_dir']);run_dir.mkdir(parents=True,exist_ok=True)
    initial=Path(c['initial_artifact']);initial_version=verify_artifact(json.loads(initial.read_text()))
    split=json.loads((Path(c['prepared'])/'split.json').read_text())
    protocol=c['optimization'].get('score_protocol','legacy_v1')
    expected_prompt=ALIGNED_JUDGE_PROMPT if protocol=='aligned_v2' else JUDGE_PROMPT
    if (initial_version['judge_model']!=c['optimization']['judge_model'] or initial_version['judge_prompt']!=expected_prompt
        or initial_version.get('score_protocol','legacy_v1')!=protocol
        or initial_version['checks'] or initial_version['muted_codes']
        or initial_version['provenance'].get('round')!=0 or initial_version['provenance'].get('split_hash')!=digest(split)):
        raise ValueError('Initial artifact does not match the approved optimizer contract')
    if c.get('arm') not in ('adaptive','frozen'):raise ValueError('Expected adaptive or frozen arm')
    with file_lock(run_dir/'run.lock'):
        state=resume_state(c)
        if state is None:ensure_new_training_range(run_dir/'checkpoints',0)
        plan=execution_plan(c,state)
        paid=[paid_stage_requirements(c,s) for s in plan if s['mode'] in ('train','optimization')]
        check_execution_budget(c['api'],c.get('planned_stage_budget',{}),paid,run_dir/'launch_budget_decision.json',protocol=protocol)
        # Above is read-only estimation, not a money reservation. Recheck every
        # paid stage against live spent/reserved immediately before starting it.
        gpus=assigned_gpus(os.environ)
        final_step=c.get('final_step',28)
        phase_root=run_dir if state is None else run_dir/('continuation_'+str(state['step'])+'_'+digest(c['resume_from'])[:10])
        phase_root.mkdir(parents=True,exist_ok=True)
        atomic_json(phase_root/'execution_plan.json',{'resume':state,'stages':plan,'paid_stages':paid,'manifest':approval_digest(c)})
        env=runtime_env(c)
        check=subprocess.run([c['python'],'-c','import torch; assert torch.cuda.device_count()==4; print([torch.cuda.get_device_name(i) for i in range(4)])'],env=env,check=True,text=True,capture_output=True)
        names=check.stdout
        if not all('A100' in x or 'H100' in x for x in json.loads(names.replace("'",'"'))):raise ValueError('Unexpected GPU model')
        atomic_json(run_dir/'allocation.json',{'job_id':os.getenv('JOB_ID'),'host':socket.gethostname(),'devices':gpus,'names':names})
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        # SGLang defaults gRPC to HTTP+10000, which can exceed 65535.
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));grpc_port=sock.getsockname()[1]
        url=f'http://127.0.0.1:{port}'
        worker_env=dict(env,CUDA_VISIBLE_DEVICES=gpus[-1],SGLANG_GRPC_PORT=str(grpc_port));env['CUDA_VISIBLE_DEVICES']=','.join(gpus[:3])
        env['HA_SPOKE_BASE_URL']=url+'/v1';env['HA_SPOKE_MODEL']=c['model_path'];env['HA_SPOKE_API_KEY']='EMPTY'
        cmd=[c['python'],'-m','sglang.launch_server','--model-path',c['model_path'],'--served-model-name',c['model_path'],
             '--host','127.0.0.1','--port',str(port),'--tp','1','--context-length','32768','--mem-fraction-static','0.8',
             '--tool-call-parser','qwen3_coder','--reasoning-parser','qwen3','--disable-cuda-graph','--trust-remote-code']
        worker=None
        def interrupted(signum,frame):raise KeyboardInterrupt('Scheduler termination')
        oldterm=signal.signal(signal.SIGTERM,interrupted)
        try:
            with (run_dir/'worker.log').open('a') as log:
                worker=subprocess.Popen(cmd,env=worker_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                deadline=time.monotonic()+900
                while True:
                    if worker.poll() is not None:raise RuntimeError('Frozen worker failed to start')
                    try:
                        with urllib.request.urlopen(url+'/health_generate',timeout=60) as response:
                            if response.status==200:break
                    except OSError:pass
                    if time.monotonic()>deadline:raise TimeoutError('Frozen worker health deadline')
                    time.sleep(5)
                artifact=Path(state['artifact']) if state else initial
                previous=Path(state['checkpoint']) if state else None
                for stage in plan:
                    name=stage['stage'];end=stage['end'];mode=stage['mode']
                    if mode!='optimization':
                        execute_phase(c,mode,end,phase_root/name,previous,artifact,env,approval)
                        if mode=='train':previous=Path(c['run_dir'])/'checkpoints'/f'global_step_{end}'
                        continue
                    require_approval(c,approval)
                    requirement=paid_stage_requirements(c,stage)
                    check_phase_budget(c['api'],c.get('planned_stage_budget',{}),name,requirement['required_calls'],
                        phase_root/name/'budget_decision.json',protocol=protocol,models=requirement['models'])
                    cal=phase_root/f'calibration_{end:02d}'
                    done=json.loads((cal/'complete.json').read_text());records=load_calibration(Path(done['attempt'])/'audit',split,n=8)
                    current=json.loads(Path(artifact).read_text());opt_dir=phase_root/name
                    optimize(records,split,c['optimization'],OpenAIClient(c['api']),opt_dir,stage['round'],str(previous),current)
                    artifact=opt_dir/'artifact.json'
                atomic_json(run_dir/'complete.json',{'final_step':final_step,'last_artifact':str(artifact),
                    'full_28_step_experiment_complete':final_step==28})
        finally:
            stop_owned(worker);signal.signal(signal.SIGTERM,oldterm)


def trainer_main():
    """Project-owned veRL entrypoint; never modifies the installed package."""
    import hydra,ray,torch
    from verl.trainer import main_ppo
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from verl.utils.device import auto_set_device
    from verl.trainer.main_ppo import migrate_legacy_reward_impl

    class RestoringTrainer(RayPPOTrainer):
        def _save_checkpoint(self):
            context=json.loads(Path(os.environ['GUIDANCE_CHECKPOINT_CONTEXT']).read_text())
            check_checkpoint_space(context)
            super()._save_checkpoint()
            publish_complete_checkpoint(context,context['attempt'],self.global_steps)

        def _load_checkpoint(self):
            super()._load_checkpoint()
            if self.config.trainer.resume_mode=='resume_path' and self.global_steps%len(self.train_dataloader)==0:
                state=torch.load(Path(self.config.trainer.resume_from_path)/'data.pt',map_location='cpu',weights_only=False)
                restore_epoch_boundary(self.train_dataloader,state,self.global_steps)

    class GuidanceTaskRunner(main_ppo.TaskRunner):
        def run(self,config):
            # This assignment is local to the Ray TaskRunner process. It selects
            # our subclass while keeping the installed dataset/worker setup.
            from verl.trainer import main_ppo as entry
            entry.RayPPOTrainer=RestoringTrainer
            return super().run(config)

    @hydra.main(config_path=None,config_name=None,version_base=None)
    def main(config):
        auto_set_device(config)
        config=migrate_legacy_reward_impl(config)
        main_ppo.run_ppo(config,task_runner_class=ray.remote(num_cpus=1)(GuidanceTaskRunner))
    main()


if __name__=='__main__':trainer_main()
