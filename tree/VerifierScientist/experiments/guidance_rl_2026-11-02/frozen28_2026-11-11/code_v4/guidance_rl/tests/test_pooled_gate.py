"""Pooled-replicate gate (gate_protocol pooled_v3), added 2026-11-08."""
import json,unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock
from guidance_rl import core
from guidance_rl.core import gate_pooled,pool_replicates,group_mae,evaluate,make_artifact
from guidance_rl.pipeline import select_candidate,selection_safe_pooled
from guidance_rl.optimize import pooled_summary,optimize,ALIGNED_JUDGE_PROMPT,AlignedVerdict,Library,Equivalence


def _metrics(pair_acc,f1,mae,cfa=None,n_clean=0,gold=100,recall=.5,n_pairs=100):
    return {'score_protocol':'aligned_v2','pair_acc':pair_acc,'sar_related_f1':f1,'mae':mae,'clean_false_alarm':cfa,
            'n_clean':n_clean,'n_pairs':n_pairs,'per_code':{'V-OT':{'gold':gold,'recall':recall}}}


class PooledGateTests(unittest.TestCase):
    def test_detection_gain_with_noninferior_ranking_is_accepted(self):
        old=_metrics(.75,.37,.20);new=_metrics(.78,.58,.13)
        iv={'pair_acc':[-.009,.06],'sar_related_f1':[.148,.274],'mae':[-.098,-.037]}
        d=gate_pooled(old,new,iv)
        self.assertTrue(d['accepted'],d['reasons']);self.assertTrue(d['gains']['sar_related_f1']);self.assertFalse(d['gains']['pair_acc'])

    def test_ranking_regression_beyond_margin_is_rejected_even_with_detection_gain(self):
        old=_metrics(.75,.37,.20);new=_metrics(.70,.58,.13)
        iv={'pair_acc':[-.09,-.01],'sar_related_f1':[.148,.274],'mae':[-.098,-.037]}
        d=gate_pooled(old,new,iv)
        self.assertFalse(d['accepted']);self.assertIn('ranking_regression',d['reasons'])

    def test_gain_below_minimum_effect_is_not_a_gain(self):
        old=_metrics(.75,.37,.20);new=_metrics(.755,.375,.20)
        iv={'pair_acc':[.001,.01],'sar_related_f1':[.001,.01],'mae':[-.01,.01]}
        d=gate_pooled(old,new,iv)
        self.assertFalse(d['accepted']);self.assertIn('no_confirmed_gain',d['reasons'])

    def test_error_regression_is_rejected(self):
        old=_metrics(.75,.37,.13);new=_metrics(.80,.40,.17)
        iv={'pair_acc':[.02,.08],'sar_related_f1':[-.01,.07],'mae':[.01,.07]}
        d=gate_pooled(old,new,iv)
        self.assertFalse(d['accepted']);self.assertIn('error_regression',d['reasons'])

    def test_small_denominator_constraints_are_inactive_and_reported(self):
        old=_metrics(.75,.37,.20,cfa=.0,n_clean=6,gold=10,recall=.9);new=_metrics(.78,.58,.13,cfa=.33,n_clean=6,gold=10,recall=.5)
        iv={'pair_acc':[-.009,.06],'sar_related_f1':[.148,.274],'mae':[-.098,-.037]}
        d=gate_pooled(old,new,iv)
        self.assertTrue(d['accepted'],d['reasons'])
        self.assertIn('clean_false_alarm:too_few_clean_trajectories',d['inactive_constraints'])
        self.assertIn('recall:V-OT:too_few_reference_positives',d['inactive_constraints'])

    def test_large_denominator_constraints_are_active(self):
        old=_metrics(.75,.37,.20,cfa=.1,n_clean=40,gold=100,recall=.9);new=_metrics(.78,.58,.13,cfa=.3,n_clean=40,gold=100,recall=.5)
        iv={'pair_acc':[-.009,.06],'sar_related_f1':[.148,.274],'mae':[-.098,-.037]}
        d=gate_pooled(old,new,iv)
        self.assertFalse(d['accepted']);self.assertIn('clean_false_alarm_regression',d['reasons']);self.assertIn('recall_regression:V-OT',d['reasons'])

    def test_worst_group_regression_is_rejected(self):
        old=_metrics(.75,.37,.20);new=_metrics(.78,.58,.13)
        iv={'pair_acc':[-.009,.06],'sar_related_f1':[.148,.274],'mae':[-.098,-.037]}
        og={'a':{'mae':.2,'n':10},'b':{'mae':.1,'n':10}};ng={'a':{'mae':.25,'n':10},'b':{'mae':.05,'n':10}}
        self.assertIn('worst_group_regression',gate_pooled(old,new,iv,og,ng)['reasons'])
        ng={'a':{'mae':.15,'n':10},'b':{'mae':.1,'n':10}}
        self.assertTrue(gate_pooled(old,new,iv,og,ng)['accepted'])

    def test_pooling_keeps_task_identity_and_sums_pairs(self):
        rs=[{'traj_id':'a','task_id':'x','actions':[{'sequence_num':1}],'gold':[],'resource_tools':[],'domain':'d1'},
            {'traj_id':'b','task_id':'x','actions':[{'sequence_num':1}],'gold':[{'seq':1,'code':'V-OT'}],'resource_tools':[],'domain':'d1'}]
        reps=[{'a':[],'b':[]},{'a':[],'b':[{'seq':1,'code':'V-OT','severity':'high'}]},{'a':[],'b':[]}]
        pooled,preds=pool_replicates(rs,reps)
        self.assertEqual(len(pooled),6);self.assertEqual(len({r['traj_id'] for r in pooled}),6);self.assertEqual({r['task_id'] for r in pooled},{'x'})
        m=evaluate(pooled,preds,protocol='aligned_v2')
        # 6 trajectories of one task: 15 pairs, 9 of which have different gold scores (3 clean x 3 violated)
        self.assertEqual(m['n_pairs'],9);self.assertEqual(m['n_clean'],3)
        self.assertEqual(set(group_mae(pooled,preds,protocol='aligned_v2')),{'d1'})
        with self.assertRaises(ValueError):pool_replicates(rs,[{'a':[]}])

    def test_pooled_summary_reports_inactive_constraints(self):
        rs=[{'traj_id':'a','task_id':'x','actions':[{'sequence_num':1}],'gold':[],'resource_tools':[],'domain':'d1'},
            {'traj_id':'b','task_id':'x','actions':[{'sequence_num':1}],'gold':[{'seq':1,'code':'V-OT'}],'resource_tools':[],'domain':'d1'}]
        reps=[{'a':[],'b':[]}]*3
        s=pooled_summary(rs,reps,'aligned_v2')
        self.assertEqual(s['aggregation'],'pooled_replicates');self.assertEqual(s['n_trajectories'],6);self.assertEqual(s['replicate_count'],3)
        self.assertIn('clean_false_alarm:too_few_clean_trajectories',s['inactive_constraints'])
        self.assertIn('recall:V-OT:too_few_reference_positives',s['inactive_constraints'])

    def test_pooled_selection_ignores_small_denominator_false_alarm_but_not_large(self):
        old=_metrics(.60,.44,.16,cfa=.0,n_clean=6);seed=_metrics(.61,.50,.14,cfa=1.,n_clean=6)
        self.assertTrue(selection_safe_pooled(old,seed))
        self.assertEqual(select_candidate({'incumbent':old,'proposal_seed':seed},'incumbent',gate_protocol='pooled_v3'),'proposal_seed')
        self.assertEqual(select_candidate({'incumbent':old,'proposal_seed':seed},'incumbent'),'incumbent')
        big_old=_metrics(.60,.44,.16,cfa=.0,n_clean=60);big_seed=_metrics(.61,.50,.14,cfa=.5,n_clean=60)
        self.assertFalse(selection_safe_pooled(big_old,big_seed))
        with self.assertRaises(ValueError):select_candidate({'incumbent':old},'incumbent',gate_protocol='nope')

    def test_optimizer_pooled_protocol_writes_single_pooled_decision(self):
        # Three replicates, 2 propose + 2 select + 2 confirm tasks x 2 trajectories.
        def rec(tid,tr,gold):
            return {'traj_id':f'{tid}-{tr}','task_id':tid,'domain':'d','visible_task':'goal ## Tools available x','resource_tools':[],
                    'actions':[{'sequence_num':1,'content':'a'},{'sequence_num':2,'content':'forbidden tool' if gold else 'ok'}],'gold':gold,'annotated_rules':'r'}
        records=[]
        for tid in ('p1','p2','s1','s2','c1','c2'):
            records+=[rec(tid,0,[]),rec(tid,1,[{'seq':2,'code':'V-OT'}])]
        split={'propose':['p1','p2'],'select':['s1','s2'],'confirm':['c1','c2'],'train_u':[],'test':[]}
        lib=Library(rules=[{'rule_id':'R1','text':'Specialists use only assigned capabilities.'}])
        client=Mock();client.config={'concurrency':1}
        def call(instructions,user,schema,**kw):
            if schema is Library:return lib
            if schema is Equivalence:return Equivalence(equivalent=True,changed_rule_ids=[],explanation='')
            payload=json.loads(user)
            # Guided judge is right on violated trajectories; unguided judge always says clean.
            violated=any(a['content']=='forbidden tool' for a in payload['actions']) and payload['guidance']
            return AlignedVerdict(violations=[{'seq':2,'code':'V-OT','severity':'high','reason':'x'}] if violated else [])
        client.call.side_effect=call
        cfg={'generator_model':'g','judge_model':'j','judge_replicates':3,'score_protocol':'aligned_v2','baseline_start':True,
             'candidate_names':['proposal_seed'],'gate_protocol':'pooled_v3','bootstrap_samples':50,'min_confirm_pairs':1,
             'gate_parameters':{'min_pairs':1,'min_gold':1,'min_clean':1}}
        with TemporaryDirectory() as d:
            art=optimize(records,split,cfg,client,Path(d),0,'policy')
            self.assertTrue((Path(d)/'confirm_pooled.json').exists())
            pooled=json.loads((Path(d)/'confirm_pooled.json').read_text())
            self.assertEqual(pooled['decision']['protocol'],'pooled_v3')
            self.assertEqual(pooled['decision']['replicates_pooled'],3)
            self.assertEqual(json.loads((Path(d)/'selection.json').read_text())['gate_protocol'],'pooled_v3')
            self.assertEqual(json.loads((Path(d)/'selection.json').read_text())['metrics']['incumbent']['aggregation'],'pooled_replicates')
            result=json.loads((Path(d)/'result.json').read_text())
            self.assertEqual(result['decision']['accepted'],pooled['decision']['accepted'])
            if result['decision']['accepted']:self.assertTrue(art['guidance'])
            else:self.assertEqual(art['guidance'],'')

    def test_repair_draws_enter_selection_as_separate_candidates(self):
        def rec(tid,tr,gold):
            return {'traj_id':f'{tid}-{tr}','task_id':tid,'domain':'d','visible_task':'goal ## Tools available x','resource_tools':[],
                    'actions':[{'sequence_num':1,'content':'a'},{'sequence_num':2,'content':'forbidden tool' if gold else 'ok'}],'gold':gold,'annotated_rules':'r'}
        records=[]
        for tid in ('p1','p2','s1','s2','c1','c2'):
            records+=[rec(tid,0,[]),rec(tid,1,[{'seq':2,'code':'V-OT'}])]
        split={'propose':['p1','p2'],'select':['s1','s2'],'confirm':['c1','c2'],'train_u':[],'test':[]}
        client=Mock();client.config={'concurrency':1};salts=[]
        def call(instructions,user,schema,**kw):
            if schema is Library:
                salts.append(kw.get('salt',''))
                # Each proposer draw returns a different wording, so draws stay distinct candidates.
                return Library(rules=[{'rule_id':'R1','text':f"Specialists use only assigned capabilities (draw {len(salts)})."}])
            payload=json.loads(user)
            violated=any(a['content']=='forbidden tool' for a in payload['actions']) and payload['guidance']
            return AlignedVerdict(violations=[{'seq':2,'code':'V-OT','severity':'high','reason':'x'}] if violated else [])
        client.call.side_effect=call
        base={'generator_model':'g','judge_model':'j','judge_replicates':3,'score_protocol':'aligned_v2','baseline_start':True,
              'candidate_names':['error_repair'],'gate_protocol':'pooled_v3','bootstrap_samples':20,'min_confirm_pairs':1,
              'gate_parameters':{'min_pairs':1,'min_gold':1,'min_clean':1}}
        with TemporaryDirectory() as d:
            optimize(records,split,{**base,'repair_draws':3},client,Path(d),0,'policy')
            sel=json.loads((Path(d)/'selection.json').read_text())
            self.assertEqual(sel['repair_draws'],3)
            self.assertEqual(sorted(k for k in sel['metrics'] if k.startswith('error_repair')),['error_repair','error_repair_d1','error_repair_d2'])
            self.assertTrue((Path(d)/'error_repair_d2.json').exists())
            self.assertEqual(json.loads((Path(d)/'error_repair_d1.json').read_text())['draw'],1)
            repair_salts=[s for s in salts if 'draw-' in s]
            self.assertEqual(len(repair_salts),2)  # draw 0 keeps the historical salt
            self.assertIn(sel['winner'],sel['metrics'])
        salts.clear()
        with TemporaryDirectory() as d:
            optimize(records,split,base,client,Path(d),0,'policy')  # default: one draw, historical file names
            sel=json.loads((Path(d)/'selection.json').read_text())
            self.assertEqual(sel['repair_draws'],1)
            self.assertFalse((Path(d)/'error_repair_d1.json').exists())
            self.assertNotIn('draw',json.loads((Path(d)/'error_repair.json').read_text()))
            self.assertFalse(any('draw-' in s for s in salts))
        with TemporaryDirectory() as d,self.assertRaises(ValueError):
            optimize(records,split,{**base,'repair_draws':0},client,Path(d),0,'policy')

    def test_paid_requirements_count_repair_draws_and_reward_replicates(self):
        from guidance_rl.training import paid_stage_requirements
        base={'api':{'max_transport_retries':0},'optimization':{'judge_model':'j','generator_model':'g','score_protocol':'aligned_v2','baseline_start':True,
              'candidate_names':['proposal_seed','error_repair'],'judge_replicates':3}}
        opt={'stage':'optimization_07','mode':'optimization','start':7,'end':7}
        self.assertEqual(paid_stage_requirements(base,opt)['logical_calls'],2642)
        three={**base,'optimization':{**base['optimization'],'repair_draws':3}}
        self.assertEqual(paid_stage_requirements(three,opt)['logical_calls'],2642+2*(10*8*3)+2)
        train={'stage':'train_07','mode':'train','start':1,'end':7}
        self.assertEqual(paid_stage_requirements(base,train)['logical_calls'],6*24*8)
        self.assertEqual(paid_stage_requirements({**base,'reward_judge_replicates':2},train)['logical_calls'],2*6*24*8)
        with self.assertRaises(ValueError):paid_stage_requirements({**base,'reward_judge_replicates':0},train)

    def test_recall_constraint_codes_restrict_the_veto_and_report_inactive(self):
        from guidance_rl.pipeline import selection_safe_pooled
        def m(recall_or,recall_ot,**kw):
            return {'score_protocol':'aligned_v2','pair_acc':.7,'sar_related_f1':.5,'mae':.1,'n_pairs':100,'n_clean':30,'clean_false_alarm':.1,
                    'per_code':{'V-OR':{'gold':500,'recall':recall_or},'V-OT':{'gold':500,'recall':recall_ot},'V-IC':{'gold':0,'recall':None},'V-ID':{'gold':10,'recall':.5}},**kw}
        old=m(.30,.44);new=m(.21,.52,pair_acc=.76,sar_related_f1=.57,mae=.08)
        iv={'pair_acc':[.02,.11],'sar_related_f1':[.04,.11],'mae':[-.05,-.01]}
        d=gate_pooled(old,new,iv)
        self.assertFalse(d['accepted']);self.assertIn('recall_regression:V-OR',d['reasons'])
        d2=gate_pooled(old,new,iv,recall_constraint_codes=['V-OT','V-IC','V-ID'])
        self.assertTrue(d2['accepted']);self.assertIn('recall:V-OR:not_a_reward_code',d2['inactive_constraints'])
        worse=m(.30,.30,pair_acc=.76,sar_related_f1=.57,mae=.08)
        self.assertIn('recall_regression:V-OT',gate_pooled(old,worse,iv,recall_constraint_codes=['V-OT','V-IC','V-ID'])['reasons'])
        self.assertFalse(selection_safe_pooled(old,new));self.assertTrue(selection_safe_pooled(old,new,recall_codes=['V-OT','V-IC','V-ID']))

    def test_confirmed_recall_veto_needs_an_interval_below_zero(self):
        from guidance_rl.core import paired_interval
        def m(recall_id,**kw):
            return {'score_protocol':'aligned_v2','pair_acc':.7,'sar_related_f1':.5,'mae':.1,'n_pairs':100,'n_clean':30,'clean_false_alarm':.1,
                    'per_code':{'V-OT':{'gold':500,'recall':.5},'V-ID':{'gold':114,'recall':recall_id}},**kw}
        old=m(.55);new=m(.47,pair_acc=.76,sar_related_f1=.57,mae=.08)
        iv={'pair_acc':[.02,.11],'sar_related_f1':[.04,.11],'mae':[-.05,-.01]}
        self.assertIn('recall_regression:V-ID',gate_pooled(old,new,iv)['reasons'])
        # confirmed mode without an interval falls back to the point rule
        self.assertIn('recall_regression:V-ID',gate_pooled(old,new,iv,recall_veto='confirmed')['reasons'])
        noisy=gate_pooled(old,new,{**iv,'recall:V-ID':[-.21,.06]},recall_veto='confirmed')
        self.assertTrue(noisy['accepted']);self.assertIn('recall:V-ID:drop_not_confirmed',noisy['inactive_constraints'])
        real=gate_pooled(old,new,{**iv,'recall:V-ID':[-.20,-.01]},recall_veto='confirmed')
        self.assertIn('recall_regression:V-ID',real['reasons'])
        with self.assertRaises(ValueError):gate_pooled(old,new,iv,recall_veto='sometimes')
        self.assertFalse(selection_safe_pooled(old,new));self.assertTrue(selection_safe_pooled(old,new,recall_veto='confirmed'))
        self.assertFalse(selection_safe_pooled(old,m(.47,sar_related_f1=.40),recall_veto='confirmed'))
        # paired_interval addresses a per-code recall
        recs=[{'traj_id':f't{i}','task_id':f'k{i%4}','actions':[{'sequence_num':1}],'resource_tools':[],'gold':[{'seq':1,'code':'V-ID'}]} for i in range(8)]
        hit={r['traj_id']:[{'seq':1,'code':'V-ID','severity':'high'}] for r in recs};miss={r['traj_id']:[] for r in recs}
        self.assertEqual(paired_interval(recs,miss,hit,metric='recall:V-ID',n_boot=50,protocol='aligned_v2'),[-1.0,-1.0])

    def test_balanced_feedback_gives_misses_their_turn(self):
        from guidance_rl.optimize import _feedback,_feedback_balanced
        recs=[];preds={}
        for i in range(12):
            r={'traj_id':f't{i}','task_id':f'k{i}','domain':f'd{i%6}','visible_task':'goal ## Tools available x',
               'actions':[{'sequence_num':1,'content':'a'},{'sequence_num':2,'content':'b'}],'gold':[{'seq':1,'code':'V-OT'}]}
            recs.append(r);preds[r['traj_id']]=[{'seq':2,'code':c,'severity':'high'} for c in ('V-IC','V-ID','V-OR')]
        direction=lambda fb:[e['stratum']['direction'] for e in fb]
        old=_feedback(recs,preds);new=_feedback_balanced(recs,preds)
        self.assertEqual(direction(old).count('miss'),0)  # every false-positive stratum sorts before every miss stratum
        self.assertEqual(len(new),12);self.assertEqual(direction(new).count('miss'),3)
        self.assertEqual(new[0]['stratum'],{'direction':'miss','code':'V-OT','domain':'d0'})
        self.assertEqual(len({e['task_id'] for e in new}),12)  # still one example per task
        self.assertEqual(sorted({e['stratum']['code'] for e in new}),['V-IC','V-ID','V-OR','V-OT'])
        self.assertEqual(len({e['stratum']['domain'] for e in new if e['stratum']['direction']=='miss'}),3)  # domains rotate
        self.assertEqual(_feedback_balanced(recs,preds),new)  # deterministic
        self.assertEqual(len(_feedback_balanced(recs,preds,max_examples=5)),5)
        self.assertEqual(_feedback_balanced(recs,{r['traj_id']:[{'seq':1,'code':'V-OT','severity':'high'}] for r in recs}),[])

    def test_optimize_accepts_balanced_feedback_and_confirmed_recall_veto(self):
        def rec(tid,tr,gold):
            return {'traj_id':f'{tid}-{tr}','task_id':tid,'domain':'d','visible_task':'goal ## Tools available x','resource_tools':[],
                    'actions':[{'sequence_num':1,'content':'a'},{'sequence_num':2,'content':'forbidden tool' if gold else 'ok'}],'gold':gold,'annotated_rules':'r'}
        records=[]
        for tid in ('p1','p2','s1','s2','c1','c2'):
            records+=[rec(tid,0,[]),rec(tid,1,[{'seq':2,'code':'V-OT'}])]
        split={'propose':['p1','p2'],'select':['s1','s2'],'confirm':['c1','c2'],'train_u':[],'test':[]}
        client=Mock();client.config={'concurrency':1}
        def call(instructions,user,schema,**kw):
            if schema is Library:return Library(rules=[{'rule_id':'R1','text':'Specialists use only assigned capabilities.'}])
            payload=json.loads(user)
            violated=any(a['content']=='forbidden tool' for a in payload['actions']) and payload['guidance']
            return AlignedVerdict(violations=[{'seq':2,'code':'V-OT','severity':'high','reason':'x'}] if violated else [])
        client.call.side_effect=call
        base={'generator_model':'g','judge_model':'j','judge_replicates':3,'score_protocol':'aligned_v2','baseline_start':True,
              'candidate_names':['error_repair'],'gate_protocol':'pooled_v3','bootstrap_samples':20,'min_confirm_pairs':1,
              'gate_parameters':{'min_pairs':1,'min_gold':1,'min_clean':1,'recall_veto':'confirmed'},'feedback_selection':'balanced_v2'}
        with TemporaryDirectory() as d:
            optimize(records,split,base,client,Path(d),0,'policy')
            fb=json.loads((Path(d)/'proposal_feedback.json').read_text())
            self.assertEqual(fb['feedback_selection'],'balanced_v2');self.assertTrue(all(e['stratum']['direction']=='miss' for e in fb['examples']))
            cp=Path(d)/'confirm_pooled.json'
            if cp.exists():
                dec=json.loads(cp.read_text())['against_incumbent']['decision']
                self.assertEqual(dec['parameters']['recall_veto'],'confirmed');self.assertIn('recall:V-OT',dec['intervals'])
        with TemporaryDirectory() as d,self.assertRaises(ValueError):
            optimize(records,split,{**base,'feedback_selection':'everything'},client,Path(d),0,'policy')

if __name__=='__main__':unittest.main()
