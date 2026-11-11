"""Pure runtime boundaries: answer extraction, selection and review manifests."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
from .core import digest


def extract_final(raw, termination, thinking_open=False):
    # Qwen's generation may begin inside an already-open <think> in the prompt.
    text=raw.rsplit('</think>',1)[-1] if '</think>' in raw else ('' if thinking_open else raw)
    if '<think>' in text:text=text.split('<think>',1)[0]
    for token in ('<|im_end|>','<|endoftext|>','<|im_start|>assistant'):
        text=text.replace(token,'')
    status=termination
    if '<tool_call' in text or '<function=' in text:
        status='tool_call' if '</tool_call>' in text else 'malformed_tool'
        text=text.split('<tool_call',1)[0].split('<function=',1)[0]
        # A turn containing a tool invocation is not the final user answer.
        text=''
    return {'text':text.strip(),'status':status,'extraction_version':1,'raw_turn':raw}


def selection_safe(old,new):
    if old.get('score_protocol','legacy_v1')!=new.get('score_protocol','legacy_v1'):return False
    key='sar_related_f1' if old.get('score_protocol')=='aligned_v2' else 'macro_f1'
    if old.get(key) is None or new.get(key) is None or new[key]<old[key]-.02:return False
    if old.get('clean_false_alarm') is not None and new.get('clean_false_alarm') is not None and new['clean_false_alarm']>old['clean_false_alarm']+.02:return False
    for code,o in old['per_code'].items():
        n=new['per_code'].get(code,{})
        if o.get('gold',0) and o.get('recall') is not None and (n.get('recall') is None or n['recall']<o['recall']-.05):return False
    return True


def selection_safe_pooled(old,new,min_gold=30,min_clean=20,recall_codes=None,recall_veto='point'):
    """Pooled-replicate eligibility: same tolerances as `selection_safe`, but the
    false-alarm and per-code recall constraints are inactive when their
    denominators are below `min_clean` / `min_gold` (pooled counts)."""
    if old.get('score_protocol','legacy_v1')!=new.get('score_protocol','legacy_v1'):return False
    key='sar_related_f1' if old.get('score_protocol')=='aligned_v2' else 'macro_f1'
    if old.get(key) is None or new.get(key) is None or new[key]<old[key]-.02:return False
    if ((new.get('n_clean') or 0)>=min_clean and old.get('clean_false_alarm') is not None and new.get('clean_false_alarm') is not None
            and new['clean_false_alarm']>old['clean_false_alarm']+.02):return False
    if recall_veto not in ('point','confirmed'):raise ValueError('Unknown recall_veto mode')
    # 'confirmed': selection only nominates a candidate; the per-code recall check needs an
    # interval and is applied once, at the confirmation gate, not on the small select split.
    if recall_veto=='confirmed':return True
    for code,o in old['per_code'].items():
        n=new['per_code'].get(code,{})
        if recall_codes is not None and code not in recall_codes:continue
        if o.get('gold',0)>=min_gold and o.get('recall') is not None and (n.get('recall') is None or n['recall']<o['recall']-.05):return False
    return True


def select_candidate(metrics,incumbent,baselines=None,gate_protocol='per_replicate_v2',safe_kwargs=None):
    if gate_protocol not in ('per_replicate_v2','pooled_v3'):raise ValueError('Unknown gate protocol')
    if gate_protocol=='pooled_v3':
        kw=dict(safe_kwargs or {});safe_fn=lambda o,n:selection_safe_pooled(o,n,**kw)
    else:
        if safe_kwargs:raise ValueError('safe_kwargs apply to the pooled protocol only')
        safe_fn=selection_safe
    old=metrics[incumbent];eligible=[incumbent]
    for k,v in metrics.items():
        safe=safe_fn(old,v) and all(safe_fn(b,v) and b['pair_acc'] is not None and v['pair_acc'] is not None and v['pair_acc']>b['pair_acc'] for b in (baselines or []))
        if k!=incumbent and v['pair_acc'] is not None and safe:eligible.append(k)
    return max(eligible,key=lambda k:(metrics[k]['pair_acc'] if metrics[k]['pair_acc'] is not None else -1,k==incumbent))


def approval_digest(config):
    # Includes source + immutable inputs, never the secret env file.
    paths=sorted(set(config['code_files']+config.get('review_inputs',[])))
    return digest({'config':config,'files':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}})


def require_approval(config,path):
    a=json.loads(Path(path).read_text())
    if not a.get('approved_by') or a.get('manifest')!=approval_digest(config):raise ValueError('User/Claude review approval missing or stale for code/config/inputs')


def assigned_gpus(env):
    value=env.get('CUDA_VISIBLE_DEVICES','').strip()
    if not value:raise ValueError('Scheduler CUDA_VISIBLE_DEVICES missing; cannot infer ownership from nvidia-smi')
    ids=[v.strip() for v in value.split(',')]
    if len(ids)!=4 or len(set(ids))!=4 or any(not re.fullmatch(r'(?:\d+|GPU-[a-zA-Z0-9-]+)',x) for x in ids):raise ValueError('Exactly four explicitly assigned whole GPUs required')
    return ids
