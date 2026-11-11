import unittest
from pathlib import Path
from unittest import mock
from unittest.mock import Mock
import openai,httpx
from guidance_rl import optimize as opt
from guidance_rl import transient
from guidance_rl.core import make_artifact


def _status_error(cls,status):
    req=httpx.Request('POST','https://api.example/v1/responses')
    return cls('x',response=httpx.Response(status,request=req),body=None)


def _ok():
    v=Mock();v.model_dump.return_value={'violations':[]};return v


class OptimizeTransientRetryTests(unittest.TestCase):
    def setUp(self):
        self.record={'traj_id':'x','task_id':'a','actions':[],'visible_task':'goal','gold':[],'resource_tools':[],'access_rules':''}
        self.artifact=make_artifact(guidance='',judge_model='m',judge_prompt='p',policy='base')
    def _no_sleep(self,slept):
        return mock.patch.object(opt,'judge_with_transient_retry',lambda call:transient.judge_with_transient_retry(call,sleep=slept.append))
    def test_judge_record_retries_cloudflare_520_then_succeeds(self):
        slept=[];c=Mock()
        c.call.side_effect=[_status_error(openai.InternalServerError,520),_status_error(openai.InternalServerError,502),_ok()]
        with self._no_sleep(slept):self.assertEqual(opt.judge_record(c,self.record,self.artifact),[])
        self.assertEqual(c.call.call_count,3);self.assertEqual(slept,[10,30])
    def test_judge_record_client_error_not_retried(self):
        slept=[];c=Mock();c.call.side_effect=_status_error(openai.BadRequestError,400)
        with self._no_sleep(slept),self.assertRaises(openai.BadRequestError):opt.judge_record(c,self.record,self.artifact)
        self.assertEqual(c.call.call_count,1);self.assertEqual(slept,[])
    def test_persistent_outage_bounded(self):
        slept=[];c=Mock();c.call.side_effect=_status_error(openai.InternalServerError,520)
        with self._no_sleep(slept),self.assertRaises(openai.InternalServerError):opt.judge_record(c,self.record,self.artifact)
        self.assertEqual(c.call.call_count,len(transient.TRANSIENT_RETRY_DELAYS_S)+1)
    def test_predict_recovers_inside_thread_pool(self):
        slept=[];c=Mock();c.config={'concurrency':2}
        c.call.side_effect=[_status_error(openai.InternalServerError,520),_ok(),_ok()]
        records=[dict(self.record,traj_id='x'),dict(self.record,traj_id='y')]
        with self._no_sleep(slept):self.assertEqual(opt.predict(c,records,self.artifact),{'x':[],'y':[]})
        self.assertEqual(c.call.call_count,3)
    def test_no_unretried_model_call_sites_remain(self):
        src=Path(opt.__file__).read_text()
        self.assertEqual(src.count('client.call('),1,'only the _call helper may call the client');self.assertEqual(src.count('=_call(client,'),4,'judge, seed, rewrite and equivalence-audit calls')
    def test_reward_reexports_shared_helpers(self):
        from guidance_rl import reward
        self.assertIs(reward.judge_with_transient_retry,transient.judge_with_transient_retry)
        self.assertIs(reward.transient_api_error,transient.transient_api_error)
if __name__=='__main__':unittest.main()
