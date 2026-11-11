import unittest
from guidance_rl.reward import judge_with_transient_retry,transient_api_error,TRANSIENT_RETRY_DELAYS_S
import openai,httpx


def _status_error(cls,status):
    req=httpx.Request('POST','https://api.example/v1/responses')
    return cls('x',response=httpx.Response(status,request=req),body=None)


class TransientRetryTests(unittest.TestCase):
    def test_internal_server_error_retried_then_succeeds(self):
        calls=[];slept=[];log=[]
        def call():
            calls.append(1)
            if len(calls)<3:raise _status_error(openai.InternalServerError,503)
            return ['ok']
        out=judge_with_transient_retry(call,sleep=slept.append,log=log.append)
        self.assertEqual(out,['ok']);self.assertEqual(len(calls),3);self.assertEqual(slept,[10,30])
        self.assertEqual([r['retry_attempt'] for r in log],[1,2]);self.assertTrue(all(r['transient_error']=='InternalServerError' for r in log))
    def test_connection_and_rate_limit_are_transient(self):
        req=httpx.Request('POST','https://api.example/v1/responses')
        self.assertTrue(transient_api_error(openai.APIConnectionError(request=req)))
        self.assertTrue(transient_api_error(openai.APITimeoutError(request=req)))
        self.assertTrue(transient_api_error(_status_error(openai.RateLimitError,429)))
        self.assertTrue(transient_api_error(_status_error(openai.InternalServerError,502)))
    def test_client_errors_and_budget_never_retry(self):
        from guidance_rl.client import BudgetExceeded
        slept=[]
        for err in (_status_error(openai.BadRequestError,400),_status_error(openai.AuthenticationError,401),ValueError('x'),BudgetExceeded('cap')):
            calls=[]
            def call():
                calls.append(1);raise err
            with self.assertRaises(type(err)):judge_with_transient_retry(call,sleep=slept.append)
            self.assertEqual(len(calls),1)
        self.assertEqual(slept,[])
    def test_persistent_outage_bounded_and_reraised(self):
        calls=[];slept=[]
        def call():
            calls.append(1);raise _status_error(openai.InternalServerError,503)
        with self.assertRaises(openai.InternalServerError):judge_with_transient_retry(call,sleep=slept.append)
        self.assertEqual(len(calls),len(TRANSIENT_RETRY_DELAYS_S)+1);self.assertEqual(slept,list(TRANSIENT_RETRY_DELAYS_S))
    def test_slot_released_between_attempts(self):
        # The slot context is entered inside call(); after a failure it must be exited before sleeping.
        state={'held':False,'held_while_sleeping':False};calls=[]
        class Slot:
            def __enter__(self):state['held']=True
            def __exit__(self,*a):state['held']=False
        def call():
            with Slot():
                calls.append(1)
                if len(calls)<2:raise _status_error(openai.InternalServerError,500)
                return []
        def sleep(d):state['held_while_sleeping']|=state['held']
        self.assertEqual(judge_with_transient_retry(call,sleep=sleep),[]);self.assertFalse(state['held_while_sleeping'])
if __name__=='__main__':unittest.main()


class SkippableJudgeFailureTests(unittest.TestCase):
    def test_classification(self):
        from guidance_rl.reward import skippable_judge_failure
        from guidance_rl.client import BudgetExceeded
        import pydantic
        class M(pydantic.BaseModel):x:int
        try:M(x='no')
        except pydantic.ValidationError as e:verr=e
        self.assertEqual(skippable_judge_failure(ValueError('Input exceeds configured context guard; no silent truncation')),'input_exceeds_context_guard')
        self.assertEqual(skippable_judge_failure(_status_error(openai.BadRequestError,400)),'api_BadRequestError')
        self.assertEqual(skippable_judge_failure(_status_error(openai.InternalServerError,503)),'api_InternalServerError')
        self.assertEqual(skippable_judge_failure(verr),'structured_output_validation_error')
        self.assertEqual(skippable_judge_failure(RuntimeError('Model returned incomplete/refused structured output')),'incomplete_or_refused_structured_output')
        self.assertIsNone(skippable_judge_failure(BudgetExceeded('cap')))
        self.assertIsNone(skippable_judge_failure(ValueError('Segment guidance changed')))
        self.assertIsNone(skippable_judge_failure(KeyError('x')))
    def test_allowance_fails_closed(self):
        import json,tempfile
        from pathlib import Path
        from guidance_rl.reward import skipped_judgments_allowed,MIN_SKIPPED_ALLOWANCE
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(skipped_judgments_allowed(d))
            for i in range(MIN_SKIPPED_ALLOWANCE-1):(Path(d)/f'{i}.json').write_text(json.dumps({'judge_skipped':'x'}))
            self.assertTrue(skipped_judgments_allowed(d))
            (Path(d)/'last.json').write_text(json.dumps({'judge_skipped':'x'}))
            self.assertFalse(skipped_judgments_allowed(d))
            # 5% of a large attempt exceeds the absolute minimum
            for i in range(400):(Path(d)/f'ok{i}.json').write_text(json.dumps({'judge_skipped':None}))
            self.assertTrue(skipped_judgments_allowed(d))
    def test_compute_score_skips_oversized_and_scores_unsafe(self):
        import json,os,tempfile
        from pathlib import Path
        from unittest.mock import patch
        from guidance_rl.core import make_artifact
        from guidance_rl.reward import compute_score
        with tempfile.TemporaryDirectory() as d:
            a=make_artifact(guidance='',judge_model='m',judge_prompt='p',policy='base',score_protocol='aligned_v2')
            ap=Path(d)/'a.json';ap.write_text(json.dumps(a));audit=Path(d)/'audit'
            cp=Path(d)/'c.json';cp.write_text(json.dumps({'artifact':str(ap),'artifact_id':a['artifact_id'],'mode':'train','audit_dir':str(audit),'api':{'budget_file':str(Path(d)/'budget.json'),'concurrency':2}}))
            info={'guidance_final':{'text':'a','status':'tool_call','extraction_version':1},'guidance_rollout_id':'r'}
            r={'traj_id':'r','task_id':'t','actions':[],'resource_tools':[],'gold':[]}
            boom=ValueError('Input exceeds configured context guard; no silent truncation')
            with patch.dict(os.environ,{'GUIDANCE_REWARD_CONFIG':str(cp)}),patch('guidance_rl.reward.build_record',return_value=(r,{'sar_avg':1.,'tcr_det':.8})),patch('guidance_rl.reward.OpenAIClient'),patch('guidance_rl.reward.judge_record',side_effect=boom):
                out=compute_score(ground_truth='t',extra_info=info)
            self.assertEqual(out['verifier_sar'],0.);self.assertAlmostEqual(out['score'],.4);self.assertEqual(out['judge_skipped'],1.)
            saved=json.loads(next(audit.glob('*.json')).read_text())
            self.assertEqual(saved['status'],'complete');self.assertIsNone(saved['prediction']);self.assertEqual(saved['judge_skipped'],'input_exceeds_context_guard')
            # Budget exhaustion still propagates
            from guidance_rl.client import BudgetExceeded
            with patch.dict(os.environ,{'GUIDANCE_REWARD_CONFIG':str(cp)}),patch('guidance_rl.reward.build_record',return_value=(r,{'sar_avg':1.,'tcr_det':.8})),patch('guidance_rl.reward.OpenAIClient'),patch('guidance_rl.reward.judge_record',side_effect=BudgetExceeded('cap')):
                with self.assertRaises(BudgetExceeded):compute_score(ground_truth='t',extra_info=dict(info,guidance_rollout_id='r2'))
            # Unexpected ValueError still propagates
            with patch.dict(os.environ,{'GUIDANCE_REWARD_CONFIG':str(cp)}),patch('guidance_rl.reward.build_record',return_value=(dict(r,traj_id='r3'),{'sar_avg':1.,'tcr_det':.8})),patch('guidance_rl.reward.OpenAIClient'),patch('guidance_rl.reward.judge_record',side_effect=ValueError('Segment guidance changed')):
                with self.assertRaises(ValueError):compute_score(ground_truth='t',extra_info=dict(info,guidance_rollout_id='r3'))
