"""Fail-closed, read-only cost planning before paid stages.

This is an estimate, not a money reservation. The client ledger remains the hard
cap. A p95 price is not a maximum; retries, distribution shifts and other spend
can still exhaust the cap. Cost evidence must be a reviewed audit pinned by SHA.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import time
from .core import atomic_json


class PhaseBudgetError(ValueError):
    pass


def _number(value,name,positive=False):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0 or (positive and value==0):
        raise ValueError(name+' must be a finite '+('positive' if positive else 'nonnegative')+' number')
    return float(value)


def _count(value,name):
    if isinstance(value,bool) or not isinstance(value,int) or value<1:raise ValueError(name+' must be a positive integer')
    return value


def _estimate(plan,stage,required_calls,protocol,models):
    spec=plan['stages'][stage]
    required_calls=_count(required_calls,'required_calls')
    calls=_count(spec['planned_calls'],'planned_calls')
    if calls<required_calls:raise ValueError('Planned calls omit required calls or retry attempts')
    raw=Path(plan['cost_audit']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=plan['cost_audit_sha256']:raise ValueError('Cost audit hash mismatch')
    audit=json.loads(raw)
    if audit.get('schema_version')!=1 or not audit.get('provenance'):raise ValueError('Missing reviewed cost audit provenance')
    if _count(audit['sample_count'],'sample_count')<_count(plan.get('min_cost_samples',30),'min_cost_samples'):
        raise ValueError('Too few cost samples')
    age=time.time()-_number(audit['observed_at'],'observed_at',True)
    max_age=_number(plan.get('max_cost_age_hours',48),'max_cost_age_hours')*3600
    if age<0 or age>max_age:raise ValueError('Cost audit is stale or future dated')
    if not models or not set(models)<=set(audit['models']):raise ValueError('Cost audit does not cover all paid models')
    proxy=audit['score_protocol']!=protocol
    if proxy and plan.get('allow_protocol_proxy') is not True:raise ValueError('Cost protocol mismatch requires explicit proxy acknowledgement')
    unit=_number(audit['unit_cost_p95'],'unit_cost_p95',True)
    margin=_number(spec['margin_usd'],'margin_usd')
    reserve=_number(spec['reserve_after_usd'],'reserve_after_usd')
    return {'stage':stage,'required_calls':required_calls,'planned_calls':calls,
        'unit_cost_p95':unit,'margin_usd':margin,'reserve_after_usd':reserve,
        'estimated_stage_usd':calls*unit+margin,'cost_audit':str(Path(plan['cost_audit']).resolve()),
        'cost_audit_sha256':plan['cost_audit_sha256'],'cost_sample_count':audit['sample_count'],
        'cost_protocol':audit['score_protocol'],'target_protocol':protocol,'protocol_proxy':proxy,
        'models':sorted(set(models)),'cost_provenance':audit['provenance']}


def _account(api):
    d=json.loads(Path(api['budget_file']).read_text())
    spent=_number(d['spent'],'ledger spent')
    reserved=sum(_number(v,'ledger reservation') for v in d['reserved'].values())
    return {'spent_usd':spent,'reserved_usd':reserved,'max_usd':_number(api['max_usd'],'max_usd',True),
            'budget_file':str(Path(api['budget_file']).resolve())}


def check_phase_budget(api,plan,stage,required_calls,decision_path,*,protocol,models):
    """Persist acceptance/rejection; never mutate the shared budget ledger."""
    decision={'accepted':False,'stage':stage,'checked_at':time.time(),
              'precheck_only':True,'hard_cap_owner':'OpenAIClient/Budget'}
    try:
        decision.update(_estimate(plan,stage,required_calls,protocol,models))
        decision.update(_account(api))
        total=decision['spent_usd']+decision['reserved_usd']+decision['estimated_stage_usd']+decision['reserve_after_usd']
        decision['total_required_usd']=total
        if total>decision['max_usd']+1e-12:raise ValueError('Insufficient budget for planned stage and subsequent reserve')
        decision['accepted']=True
    except Exception as error:
        decision['reason']=str(error);atomic_json(decision_path,decision)
        raise PhaseBudgetError('Phase budget precheck rejected: '+str(error)) from error
    atomic_json(decision_path,decision)
    return decision


def check_execution_budget(api,plan,stages,decision_path,*,protocol):
    """Check the entire requested continuation before any model is launched.

    Stage reserves overlap planned future stages; do not double count them.
    Every stage is checked again immediately before its paid work.
    """
    decision={'accepted':False,'checked_at':time.time(),'precheck_only':True,'stages':[]}
    try:
        if not stages:raise ValueError('No paid stages; restore-only must use a separate validation design')
        decision['stages']=[_estimate(plan,s['stage'],s['required_calls'],protocol,s['models']) for s in stages]
        decision.update(_account(api))
        running=required=0.
        for stage in decision['stages']:
            running+=stage['estimated_stage_usd']
            required=max(required,running+stage['reserve_after_usd'])
        decision['estimated_remaining_usd']=required
        decision['total_required_usd']=decision['spent_usd']+decision['reserved_usd']+required
        if decision['total_required_usd']>decision['max_usd']+1e-12:raise ValueError('Insufficient budget for the full explicit continuation')
        decision['accepted']=True
    except Exception as error:
        decision['reason']=str(error);atomic_json(decision_path,decision)
        raise PhaseBudgetError('Execution budget precheck rejected: '+str(error)) from error
    atomic_json(decision_path,decision)
    return decision
