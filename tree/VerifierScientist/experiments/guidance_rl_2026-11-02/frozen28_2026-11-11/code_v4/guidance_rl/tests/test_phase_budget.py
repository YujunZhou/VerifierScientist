import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest

EXISTS=importlib.util.find_spec('guidance_rl.phase_budget') is not None
if EXISTS:
    from guidance_rl.phase_budget import check_phase_budget,check_execution_budget,PhaseBudgetError


class PhaseBudgetTests(unittest.TestCase):
    def test_module_implemented(self):self.assertTrue(EXISTS,'phase budget precheck is missing')

    def setup_case(self,d):
        root=Path(d);ledger=root/'budget.json'
        ledger.write_text(json.dumps({'spent':1.,'reserved':{'running':.5},'calls':5}))
        audit=root/'cost.json'
        audit.write_text(json.dumps({'schema_version':1,'observed_at':time.time(),'sample_count':100,
            'unit_cost_p95':.01,'models':['judge'],'score_protocol':'legacy_v1',
            'provenance':'Historical successful judge API usage; proxy for new policy.'}))
        api={'budget_file':str(ledger),'max_usd':10.,'max_transport_retries':1}
        plan={'cost_audit':str(audit),'cost_audit_sha256':hashlib.sha256(audit.read_bytes()).hexdigest(),
            'max_cost_age_hours':48,'min_cost_samples':30,'allow_protocol_proxy':True,
            'stages':{'train_01_smoke':{'planned_calls':384,'margin_usd':.2,'reserve_after_usd':1.}}}
        return api,plan,root/'decision.json'

    @unittest.skipUnless(EXISTS,'pending')
    def test_cost_plus_inflight_and_future_reserve_is_recorded(self):
        with TemporaryDirectory() as d:
            api,plan,out=self.setup_case(d)
            decision=check_phase_budget(api,plan,'train_01_smoke',384,out,protocol='aligned_v2',models=['judge'])
            self.assertTrue(decision['accepted']);self.assertTrue(decision['protocol_proxy'])
            self.assertAlmostEqual(decision['estimated_stage_usd'],4.04)
            self.assertAlmostEqual(decision['total_required_usd'],6.54)
            self.assertEqual(json.loads(out.read_text()),decision)
            self.assertEqual(json.loads(Path(api['budget_file']).read_text())['spent'],1.)

    @unittest.skipUnless(EXISTS,'pending')
    def test_rejects_insufficient_budget_and_preserves_decision(self):
        with TemporaryDirectory() as d:
            api,plan,out=self.setup_case(d);api['max_usd']=5.
            with self.assertRaises(PhaseBudgetError):check_phase_budget(api,plan,'train_01_smoke',384,out,protocol='aligned_v2',models=['judge'])
            self.assertFalse(json.loads(out.read_text())['accepted'])

    @unittest.skipUnless(EXISTS,'pending')
    def test_underestimated_attempts_and_missing_plan_fail_closed(self):
        with TemporaryDirectory() as d:
            api,plan,out=self.setup_case(d)
            for p in ({},plan):
                with self.assertRaises(PhaseBudgetError):check_phase_budget(api,p,'train_01_smoke',385,out,protocol='aligned_v2',models=['judge'])
                self.assertFalse(json.loads(out.read_text())['accepted'])

    @unittest.skipUnless(EXISTS,'pending')
    def test_stale_tampered_or_unacknowledged_proxy_cost_fails(self):
        with TemporaryDirectory() as d:
            api,plan,out=self.setup_case(d)
            plan['allow_protocol_proxy']=False
            with self.assertRaises(PhaseBudgetError):check_phase_budget(api,plan,'train_01_smoke',384,out,protocol='aligned_v2',models=['judge'])
            plan['allow_protocol_proxy']=True;plan['max_cost_age_hours']=0
            with self.assertRaises(PhaseBudgetError):check_phase_budget(api,plan,'train_01_smoke',384,out,protocol='aligned_v2',models=['judge'])
            plan['max_cost_age_hours']=48;Path(plan['cost_audit']).write_text('{}')
            with self.assertRaises(PhaseBudgetError):check_phase_budget(api,plan,'train_01_smoke',384,out,protocol='aligned_v2',models=['judge'])

    @unittest.skipUnless(EXISTS,'pending')
    def test_full_launch_plan_cannot_double_spend_same_remaining_money(self):
        with TemporaryDirectory() as d:
            api,plan,out=self.setup_case(d);api['max_usd']=7
            plan['stages']['train_07']={'planned_calls':384,'margin_usd':.2,'reserve_after_usd':0.}
            stages=[{'stage':'train_01_smoke','required_calls':384,'models':['judge']},
                    {'stage':'train_07','required_calls':384,'models':['judge']}]
            with self.assertRaises(PhaseBudgetError):check_execution_budget(api,plan,stages,out,protocol='aligned_v2')
            self.assertFalse(json.loads(out.read_text())['accepted'])


if __name__=='__main__':unittest.main()
