"""Explicit OpenAI configuration, complete request cache and cross-process budget."""
from __future__ import annotations
import contextlib
import fcntl
import hashlib
import json
import math
import os
import shlex
import time
import uuid
from pathlib import Path
from .core import atomic_json,digest

class BudgetExceeded(RuntimeError):pass

@contextlib.contextmanager
def file_lock(path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)

def load_openai_key(path):
    # Do not source shell, interpolate variables, or load unrelated provider keys.
    value=None
    for line in Path(path).read_text().splitlines():
        lhs,sep,rhs=line.partition('=')
        if sep and lhs.strip().removeprefix('export ').strip()=='OPENAI_API_KEY':
            parts=shlex.split(rhs,comments=True)
            if len(parts)!=1:raise ValueError('Invalid OPENAI_API_KEY assignment')
            value=parts[0]
    if not value:raise ValueError('OPENAI_API_KEY missing from configured env file')
    return value

class Budget:
    def __init__(self,path,max_usd,max_calls):
        self.path=Path(path);self.max_usd=float(max_usd or 0);self.max_calls=int(max_calls or 0)
    def read(self):
        return json.loads(self.path.read_text()) if self.path.exists() else {'spent':0.,'calls':0,'reserved':{}}
    def reserve(self,worst):
        if worst<=0:raise ValueError('Cost reservation must be positive')
        with file_lock(str(self.path)+'.lock'):
            d=self.read()
            if d['calls']>=self.max_calls or d['spent']+sum(d['reserved'].values())+worst>self.max_usd:raise BudgetExceeded('Configured API budget/call limit exhausted or not configured')
            token=uuid.uuid4().hex;d['reserved'][token]=worst;d['calls']+=1;atomic_json(self.path,d)
            return token
    def settle(self,token,cost):
        with file_lock(str(self.path)+'.lock'):
            d=self.read();reserved=d['reserved'].pop(token)
            d['spent']+=max(0.,cost);atomic_json(self.path,d)
            if cost>reserved+1e-8:raise BudgetExceeded('Actual cost exceeded conservative reservation; stop and verify configured prices')

class OpenAIClient:
    def __init__(self,config,*,sleep=None):
        self.config=config;self.root=Path(config['cache_dir']);self.root.mkdir(parents=True,exist_ok=True)
        self.budget=Budget(config['budget_file'],config.get('max_usd'),config.get('max_calls'))
        retries=config.get('max_transport_retries',0)
        if type(retries) is not int or retries<0:
            raise ValueError('max_transport_retries must be a nonnegative integer')
        delay=float(config.get('transport_retry_delay_s',.25))
        if not math.isfinite(delay) or not 0<=delay<=5:
            raise ValueError('transport_retry_delay_s must be between zero and five seconds')
        self.retry_policy={'max_transport_retries':retries,'delay_s':delay,'sdk_max_retries':0}
        self._sleep=time.sleep if sleep is None else sleep
        self.source_fingerprint=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.client=None
    def request_spec(self,instructions,user,schema,*,model,salt='',max_output_tokens=None):
        cap=int(max_output_tokens or self.config.get('max_output_tokens',8192))
        return {'model':model,'instructions':instructions,'input':user,'schema':schema.model_json_schema(),
                'salt':salt,'reasoning_effort':self.config.get('reasoning_effort','low'),'service_tier':'default',
                'max_output_tokens':cap,'endpoint':self.config.get('base_url','https://api.openai.com/v1')}

    def call(self,instructions,user,schema,*,model,tag='',salt='',max_output_tokens=None):
        # Receipt is for sequential fixed evaluation only; concurrent callers use caches directly.
        self.last_receipt=None
        request=self.request_spec(instructions,user,schema,model=model,salt=salt,max_output_tokens=max_output_tokens)
        cap=request['max_output_tokens']
        key=digest(request);path=self.root/key[:2]/(key+'.json')
        with file_lock(str(path)+'.lock'):
            if path.exists():
                saved=json.loads(path.read_text())
                if saved['request']!=request:raise ValueError('Cache key mismatch')
                parsed=schema.model_validate(saved['parsed'])
                # Retry policy changes execution, not model input or the prediction.
                # Keep historical cache keys, but explicitly audit legacy reuse.
                atomic_json(self.root/'cache_reuse'/(key+'-'+uuid.uuid4().hex+'.json'),
                    {'request_key':key,'source_fingerprint':self.source_fingerprint,
                     'retry_policy':self.retry_policy,'original_attempt_provenance_present':
                     'attempt_provenance' in saved,'paid_attempt':False,'time':time.time()})
                self.last_receipt={"path":str(path.resolve()),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
                return parsed
            if self.config.get('cache_only',False):raise FileNotFoundError('Exact API cache miss: '+key)
            prices=self.config.get('prices',{}).get(model)
            if not prices or not self.config.get('prices_verified',False):raise ValueError('Verified per-model prices required before paid calls')
            # UTF-8 byte count bounds BPE tokens; include the serialized schema and framing allowance.
            upper=len(json.dumps(request,ensure_ascii=False).encode())+8192
            if upper>self.config.get('max_input_bytes',500000):raise ValueError('Input exceeds configured context guard; no silent truncation')
            long=upper>272000
            worst=(upper*prices['input']*(2 if long else 1)+cap*prices['output']*(1.5 if long else 1))/1e6
            from openai import APIConnectionError
            call_id=uuid.uuid4().hex
            provenance=[]
            for index in range(self.retry_policy['max_transport_retries']+1):
                attempt={'request_key':key,'call_id':call_id,'attempt_index':index+1,
                         'retry_policy':self.retry_policy,'source_fingerprint':self.source_fingerprint,
                         'worst_usd':worst,'time':time.time()}
                try:
                    # A retry cannot reuse the previous attempt's reservation.
                    reservation=self.budget.reserve(worst)
                except BudgetExceeded:
                    attempt.update(status='budget_blocked',error_type='BudgetExceeded',charged_usd=0)
                    atomic_json(self.root/'attempts'/f'{key}-{call_id}-{index+1}-blocked.json',attempt)
                    raise
                attempt.update(reservation=reservation,status='reserved')
                attempt_path=self.root/'attempts'/(key+'-'+reservation+'.json')
                atomic_json(attempt_path,attempt)
                settled=False;cost=worst;stage='setup'
                try:
                    if index:self._sleep(self.retry_policy['delay_s'])
                    if self.client is None:
                        from openai import OpenAI
                        self.client=OpenAI(api_key=load_openai_key(self.config['env_file']),base_url=request['endpoint'],
                                           timeout=self.config.get('timeout',180),max_retries=0)
                    # Nonce changes sampling requests without changing the auditing instruction.
                    api_user=user+(f'\n\nEvaluation replicate identifier: {salt}' if salt else '')
                    stage='transport'
                    raw=self.client.responses.with_raw_response.parse(model=model,instructions=instructions,input=api_user,
                        text_format=schema,max_output_tokens=cap,store=False,service_tier='default',
                        reasoning={'effort':request['reasoning_effort']})
                    stage='decode'
                    body=json.loads(raw.text)
                    atomic_json(self.root/'raw'/(key+'-'+reservation+'.json'),{'request':request,'response':body,'tag':tag,'time':time.time()})
                    usage=body.get('usage')
                    if not usage:raise RuntimeError('Missing API usage; conservative attempt cost charged')
                    long=usage['input_tokens']>272000
                    cost=(usage['input_tokens']*prices['input']*(2 if long else 1)+usage['output_tokens']*prices['output']*(1.5 if long else 1))/1e6
                    # settle may raise BudgetExceeded after writing its ledger.
                    settled=True
                    self.budget.settle(reservation,cost)
                    stage='structured_parse'
                    response=raw.parse()
                    if response.status!='completed' or response.output_parsed is None:raise RuntimeError('Model returned incomplete/refused structured output')
                    attempt.update(status='success',charged_usd=cost,usage_known=True)
                    atomic_json(attempt_path,attempt)
                    provenance.append(dict(attempt))
                    saved={'request':request,'tag':tag,'response':body,
                           'parsed':response.output_parsed.model_dump(mode='json'),'usage':usage,'usd_upper':cost,
                           'attempt_provenance':provenance,'time':time.time()}
                    atomic_json(path,saved)
                    self.last_receipt={"path":str(path.resolve()),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
                    return response.output_parsed
                except BaseException as e:
                    usage_known=settled
                    if not settled:
                        # A lost response may still have been billed. Charge the
                        # complete reservation instead of making a retry look free.
                        self.budget.settle(reservation,worst)
                    retry=stage=='transport' and isinstance(e,APIConnectionError) and index<self.retry_policy['max_transport_retries']
                    attempt.update(status='error',stage=stage,error_type=type(e).__name__,
                                   charged_usd=cost if usage_known else worst,
                                   usage_known=usage_known,retry_eligible=retry)
                    atomic_json(attempt_path,attempt)
                    # Never persist exception messages/headers: they may carry keys.
                    atomic_json(self.root/'errors'/(key+'-'+reservation+'.json'),
                                {'request':request,'tag':tag,**attempt})
                    provenance.append(dict(attempt))
                    if not retry:raise
