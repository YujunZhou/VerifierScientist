"""Preregistered interventions on the actual historical short/detailed guides."""
import json,re
from pathlib import Path
from .core import atomic_json,assert_role,evaluate,make_artifact,paired_interval,digest
from .optimize import Library,Equivalence,JUDGE_PROMPT,predict,validate_rewrite,render
from .client import file_lock


def parse_legacy_rules(text):
    rows=[]
    for i,line in enumerate(text.splitlines()):
        line=line.strip()
        if not line:continue
        m=re.match(r'-\s*\[([^]]+)\]\s*(.*)',line)
        rows.append({'rule_id':m[1] if m else f'C-{i+1:02d}','text':m[2] if m else line.removeprefix('- ').strip()})
    validate_rewrite(rows,rows);return rows


def remove_field(text,field):
    if field=='exception':return re.sub(r'; EXCEPT when: .*?(?=; evidence:|$)','',text,flags=re.M)
    if field=='evidence':return re.sub(r'; evidence: .*?$','',text,flags=re.M)
    raise ValueError(field)


def controlled_diagnosis(records,split,cfg,client,out):
    rows=[r for r in records if r['task_id'] in split['confirm']];assert_role(rows,split,'confirm')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    guides=json.loads(Path(cfg['historical_guidance']).read_text())
    identity=digest({'rows':rows,'cfg':cfg,'guides':guides})
    with file_lock(out/'diagnosis.lock'):
        if (out/'inputs.json').exists() and json.loads((out/'inputs.json').read_text())['identity']!=identity:raise ValueError('Diagnostic directory input mismatch')
        atomic_json(out/'inputs.json',{'identity':identity,'task_ids':sorted({r['task_id'] for r in rows}),
            'purpose':'Explanatory interventions; results never enter guidance selection','replicates':cfg.get('judge_replicates',2)})
        variants={'short':guides['rubric_bank'],'detailed':guides['scientist_v0'],'none':''}
        variants['detailed_no_exceptions']=remove_field(guides['scientist_v0'],'exception')
        variants['detailed_no_evidence']=remove_field(guides['scientist_v0'],'evidence')
        interventions=[('short_expanded','short','Expand wording without changing any obligation, exception, evidence threshold or coverage.'),
                       ('detailed_compressed','detailed','Compress wording without deleting any obligation, exception, evidence threshold or coverage.'),
                       ('detailed_plain','detailed','Change field-labeled style to ordinary prose while keeping every obligation, exception, evidence threshold and coverage.')]
        audits={}
        for name,parent,instruction in interventions:
            source=parse_legacy_rules(variants[parent])
            lib=client.call(instruction+' Keep every rule ID.',json.dumps(source),Library,model=cfg['generator_model'],tag='diagnostic:'+name,salt='controlled-v1')
            cand=[r.model_dump() for r in lib.rules]
            try:validate_rewrite(source,cand)
            except ValueError as exc:
                audits[name]={'equivalent':False,'explanation':str(exc)}
                atomic_json(out/(name+'_rejected.json'),{'rules':cand,'reason':str(exc)})
                continue
            audit=client.call('Verify each aligned rule has exactly the same applicability, obligations, exceptions and evidence threshold in BOTH directions. A meaning change invalidates this causal control.',
               json.dumps({'original':source,'rewrite':cand}),Equivalence,model=cfg['semantic_auditor_model'],tag='diagnostic_audit:'+name,salt='controlled-v1').model_dump()
            audits[name]=audit
            atomic_json(out/(name+'.json'),{'rules':cand,'audit':audit,'source':parent})
            if audit['equivalent'] and not audit['changed_rule_ids']:variants[name]=render(cand)
        atomic_json(out/'variants.json',{'guidance':variants,'audits':audits,'chars':{k:len(v) for k,v in variants.items()}})
        predictions={};metrics={}
        for name,text in variants.items():
            artifact=make_artifact(guidance=text,judge_model=cfg['judge_model'],judge_prompt=JUDGE_PROMPT,policy='historical_mixed_pool')
            ps=[predict(client,rows,artifact,rep) for rep in range(cfg.get('judge_replicates',2))]
            ms=[evaluate(rows,p) for p in ps];predictions[name]=ps;metrics[name]=ms
            atomic_json(out/('scores_'+name+'.json'),{'predictions':ps,'metrics':ms,'artifact':artifact})
        comparisons={}
        for name,parent in [('short','detailed'),('detailed_no_exceptions','detailed'),('detailed_no_evidence','detailed'),('short_expanded','short'),('detailed_compressed','detailed'),('detailed_plain','detailed')]:
            if name not in predictions:continue
            comparisons[name+' - '+parent]={m:[paired_interval(rows,a,b,metric=m,n_boot=cfg.get('bootstrap_samples',500),seed=i) for i,(a,b) in enumerate(zip(predictions[name],predictions[parent]))] for m in ('macro_f1','pair_acc','mae')}
        result={'metrics':metrics,'paired_intervals':comparisons,'semantic_audits':audits,'limitations':['Exploratory repeated historical holdout; not virgin test','LLM semantic audit is fallible; human review needed','Single generated rewrite per intervention','No scope check and full logs in all new arms; not an exact historical replay']}
        atomic_json(out/'result.json',result);return result
