"""Offline preflight: adaptive fresh480 orchestration, not scientific efficacy."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from guidance_rl.optimize import optimize, AlignedVerdict
from guidance_rl.tests import test_feedback_identity as feedback_fixtures
from guidance_rl.tests import test_phase_budget as budget_fixtures
from guidance_rl.phase_budget import check_phase_budget, PhaseBudgetError
from guidance_rl.training import execution_plan, paid_stage_requirements


class AdaptivePreflight(unittest.TestCase):
    def inputs(self):
        helper=feedback_fixtures.FeedbackIdentityTests(); _,_,cfg=helper.inputs()
        cfg.update(stage='adaptive28-offline-preflight',max_rounds=4)
        split={'propose':[f'p{i}' for i in range(30)],
               'select':[f's{i}' for i in range(10)],
               'confirm':[f'c{i}' for i in range(20)],'train_u':['u'],'test':['t']}
        rows=[]
        for role in ('propose','select','confirm'):
            for tid in split[role]:
                for i in range(8):
                    rows.append({'task_id':tid,'traj_id':f'{tid}-{i}',
                        'domain':'d','visible_task':tid,'resource_tools':[],
                        'actions':[{'sequence_num':1,'surface':'tool_call',
                                    'tool_name':'forbidden' if i%2 else 'safe'},
                                   {'sequence_num':2,'surface':'communication','content':'routine status'}],
                        'gold':[{'seq':1,'code':'V-OT','severity':'high'}] if i%2 else [],
                        **({'annotated_rules':'PROPOSE_ONLY_SECRET'} if role=='propose' else {})})
        return rows,split,cfg,helper.incumbent()

    def test_three_fresh480_rounds_match_2642_bound_and_retain_on_gate_rejection(self):
        rows,split,cfg,current=self.inputs(); namespaces=[]
        c={'final_step':28,'arm':'adaptive','api':{'max_transport_retries':0},'optimization':cfg}
        stages=[s for s in execution_plan(c) if s['mode']=='optimization']
        self.assertEqual([s['round'] for s in stages],[1,2,3])
        with TemporaryDirectory() as tmp:
            for stage in stages:
                api=feedback_fixtures.DistinctJudge();out=Path(tmp)/stage['stage']
                # Force confirmation to exercise its maximum-size branch. This
                # is a call-bound test, not evidence that any candidate wins.
                with patch('guidance_rl.optimize.select_candidate',return_value='error_repair'), \
                     patch('guidance_rl.optimize.paired_interval',return_value={}), \
                     patch('guidance_rl.optimize.gate',return_value={'accepted':False,'reasons':['offline forced rejection']}):
                    artifact=optimize(rows,split,cfg,api,out,stage['round'],f'checkpoint-{stage["end"]}',current)
                self.assertEqual(len(api.calls),2642)
                self.assertEqual(paid_stage_requirements(c,stage)['required_calls'],2642)
                self.assertEqual(artifact['guidance'],current['guidance'])
                self.assertFalse(artifact['provenance']['gate']['accepted'])
                self.assertEqual(sum(schema is AlignedVerdict for schema,_,_ in api.calls),2640)
                for schema,payload,_ in api.calls:
                    if schema is AlignedVerdict:
                        self.assertEqual(set(payload),{'task','guidance','actions'})
                        self.assertNotIn('PROPOSE_ONLY_SECRET',json.dumps(payload))
                namespaces.append({kw['salt'] for _,_,kw in api.calls})
                current=artifact
        self.assertTrue(all(not a&b for i,a in enumerate(namespaces) for b in namespaces[i+1:]))
        c['api']['max_transport_retries']=1
        self.assertEqual(paid_stage_requirements(c,stages[0])['required_calls'],5284)

    def test_reviewed_long_run_age_policy_does_not_rewrite_evidence(self):
        with TemporaryDirectory() as tmp:
            api,plan,out=budget_fixtures.PhaseBudgetTests().setup_case(tmp)
            before=Path(plan['cost_audit']).read_bytes()
            observed=json.loads(before)['observed_at']
            with patch('guidance_rl.phase_budget.time.time',return_value=observed+72*3600):
                with self.assertRaises(PhaseBudgetError):
                    check_phase_budget(api,plan,'train_01_smoke',384,out,protocol='aligned_v2',models=['judge'])
                plan['max_cost_age_hours']=120
                decision=check_phase_budget(api,plan,'train_01_smoke',384,out,protocol='aligned_v2',models=['judge'])
                self.assertTrue(decision['accepted'])
                self.assertEqual(decision['unit_cost_p95'],.01)
            self.assertEqual(Path(plan['cost_audit']).read_bytes(),before)


if __name__=='__main__':unittest.main()
