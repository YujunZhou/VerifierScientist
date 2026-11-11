"""Offline transport failures exercise the real cache and budget ledgers."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import openai
from pydantic import BaseModel

from guidance_rl.client import BudgetExceeded, OpenAIClient


class Answer(BaseModel):
    seq: int


def response(parse_error=None, text=None):
    raw = Mock()
    raw.text = text if text is not None else json.dumps({
        'status': 'completed', 'usage': {'input_tokens': 10, 'output_tokens': 20}})
    if parse_error is not None:
        raw.parse.side_effect = parse_error
    else:
        raw.parse.return_value = SimpleNamespace(status='completed', output_parsed=Answer(seq=1))
    return raw


class RetryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = {'cache_dir': str(self.root/'cache'), 'budget_file': str(self.root/'budget.json'),
                    'max_usd': 1, 'max_calls': 10, 'max_output_tokens': 10,
                    'prices_verified': True, 'prices': {'m': {'input': 1, 'output': 1}},
                    'max_transport_retries': 1, 'transport_retry_delay_s': .01}
        self.request = httpx.Request('POST', 'https://api.openai.com/v1/responses')
        self.sleeps = []
        sleeper=patch('guidance_rl.client.time.sleep',side_effect=self.sleeps.append)
        sleeper.start();self.addCleanup(sleeper.stop)

    def client(self, events):
        client = OpenAIClient(self.cfg)
        client.client = Mock()
        client.client.responses.with_raw_response.parse.side_effect = events
        return client

    def call(self, client):
        return client.call('instructions', 'task', Answer, model='m')

    def logs(self, kind):
        return [json.loads(p.read_text()) for p in (self.root/'cache'/kind).glob('*.json')]

    def test_connection_retry_reserves_and_settles_each_attempt_then_caches_once(self):
        c = self.client([openai.APIConnectionError(request=self.request), response()])
        try:
            result=self.call(c)
        except openai.APIConnectionError:
            self.fail('One permitted transport retry should recover from the first connection error')
        self.assertEqual(result.seq, 1)
        self.assertEqual(c.client.responses.with_raw_response.parse.call_count, 2)
        ledger = c.budget.read()
        self.assertEqual(ledger['calls'], 2)
        self.assertEqual(ledger['reserved'], {})
        attempts = sorted(self.logs('attempts'), key=lambda x: x['attempt_index'])
        self.assertEqual([a['status'] for a in attempts], ['error', 'success'])
        self.assertNotEqual(attempts[0]['reservation'], attempts[1]['reservation'])
        self.assertEqual(attempts[0]['charged_usd'], attempts[0]['worst_usd'])
        self.assertAlmostEqual(ledger['spent'], attempts[0]['worst_usd']+.00003)
        self.assertEqual(len(self.sleeps), 1)
        self.assertEqual(len(self.logs('errors')), 1)
        self.assertEqual(len(list((self.root/'cache').glob('*/*.json'))), 5)
        self.call(c)
        self.assertEqual(c.budget.read(), ledger)
        self.assertEqual(c.client.responses.with_raw_response.parse.call_count, 2)
        for a in attempts:
            self.assertEqual(a['retry_policy']['max_transport_retries'], 1)
            self.assertEqual(a['retry_policy']['sdk_max_retries'], 0)
            self.assertEqual(len(a['source_fingerprint']), 64)

    def test_sequential_receipt_matches_exact_success_and_cache_reuse(self):
        import hashlib
        c=self.client([response()]);self.call(c)
        receipt=dict(c.last_receipt);path=Path(receipt['path'])
        self.assertEqual(receipt['sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(json.loads(path.read_text())['request'],c.request_spec('instructions','task',Answer,model='m'))
        self.call(c);self.assertEqual(receipt,c.last_receipt)
        self.assertEqual(c.client.responses.with_raw_response.parse.call_count,1)

    def test_timeout_stops_at_explicit_retry_bound_and_accounts_unknown_usage(self):
        c = self.client([openai.APITimeoutError(request=self.request),
                         openai.APITimeoutError(request=self.request), response()])
        with self.assertRaises(openai.APITimeoutError): self.call(c)
        self.assertEqual(c.client.responses.with_raw_response.parse.call_count, 2)
        self.assertEqual(c.budget.read()['calls'], 2)
        self.assertEqual(c.budget.read()['reserved'], {})
        self.assertEqual(len(self.logs('attempts')), 2)
        self.assertEqual(len(self.sleeps), 1)

    def test_retry_cannot_bypass_dollar_budget_or_sleep_after_denied_reservation(self):
        self.cfg['max_usd'] = .01
        c = self.client([openai.APIConnectionError(request=self.request), response()])
        with self.assertRaises(BudgetExceeded): self.call(c)
        self.assertEqual(c.client.responses.with_raw_response.parse.call_count, 1)
        self.assertEqual(c.budget.read()['calls'], 1)
        self.assertGreater(c.budget.read()['spent'], 0)
        self.assertEqual(self.sleeps, [])

    def test_default_policy_preserves_no_retry(self):
        del self.cfg['max_transport_retries']
        c = self.client([openai.APIConnectionError(request=self.request), response()])
        with self.assertRaises(openai.APIConnectionError): self.call(c)
        self.assertEqual(c.client.responses.with_raw_response.parse.call_count, 1)
        self.assertEqual(self.sleeps, [])

    def test_structured_parse_failure_is_not_retried_and_known_usage_is_charged(self):
        c = self.client([response(ValueError('invalid seq')), response()])
        with self.assertRaises(ValueError): self.call(c)
        self.assertEqual(c.client.responses.with_raw_response.parse.call_count, 1)
        self.assertAlmostEqual(c.budget.read()['spent'], .00003)
        self.assertEqual(c.budget.read()['reserved'], {})

    def test_invalid_json_is_not_retried_and_unknown_usage_is_charged_worst(self):
        c = self.client([response(text='invalid JSON'), response()])
        with self.assertRaises(json.JSONDecodeError): self.call(c)
        self.assertEqual(c.client.responses.with_raw_response.parse.call_count, 1)
        self.assertEqual(c.budget.read()['reserved'], {})
        self.assertGreater(c.budget.read()['spent'], 0)

    def test_auth_quota_and_budget_failures_are_not_retried_or_logged_with_secret(self):
        secret = 'do-not-log-the-key'
        for error in (openai.AuthenticationError(secret, response=httpx.Response(401, request=self.request), body=None),
                      openai.RateLimitError(secret, response=httpx.Response(429, request=self.request), body=None),
                      BudgetExceeded(secret)):
            with self.subTest(error=type(error).__name__):
                c = self.client([error, response()])
                with self.assertRaises(type(error)): self.call(c)
                self.assertEqual(c.client.responses.with_raw_response.parse.call_count, 1)
        for p in (self.root/'cache').rglob('*.json'):
            self.assertNotIn(secret, p.read_text())

    def test_legacy_cache_is_reused_under_new_retry_policy(self):
        c = self.client([response()]); self.call(c)
        cache = next(p for p in (self.root/'cache').glob('*/*.json') if p.parent.name not in {'raw','errors','attempts'})
        saved = json.loads(cache.read_text())
        saved.pop('attempt_provenance', None)
        cache.write_text(json.dumps(saved))
        self.cfg['max_transport_retries'] = 0
        reused = self.client([])
        self.assertEqual(self.call(reused).seq, 1)
        self.assertEqual(reused.client.responses.with_raw_response.parse.call_count, 0)
        reuse=self.logs('cache_reuse')[-1]
        self.assertFalse(reuse['original_attempt_provenance_present'])
        self.assertFalse(reuse['paid_attempt'])

    def test_sdk_retry_is_disabled_even_when_explicit_transport_retry_is_enabled(self):
        env=self.root/'.env';env.write_text('OPENAI_API_KEY=test-unit-key\n')
        self.cfg['env_file']=str(env)
        sdk=Mock();sdk.responses.with_raw_response.parse.return_value=response()
        with patch('openai.OpenAI',return_value=sdk) as factory:
            self.call(OpenAIClient(self.cfg,sleep=self.sleeps.append))
        self.assertEqual(factory.call_args.kwargs['max_retries'],0)
        self.assertEqual(sdk.responses.with_raw_response.parse.call_count,1)

    def test_usage_exceeding_reservation_stops_without_double_settling_or_retry(self):
        raw=response(text=json.dumps({'usage':{'input_tokens':1000000,'output_tokens':20}}))
        c=self.client([raw,response()])
        with self.assertRaises(BudgetExceeded):self.call(c)
        self.assertEqual(c.client.responses.with_raw_response.parse.call_count,1)
        self.assertEqual(c.budget.read()['reserved'],{})
        self.assertEqual(c.budget.read()['calls'],1)
        self.assertGreater(c.budget.read()['spent'],1)

    def test_invalid_retry_controls_are_rejected_without_budget_or_api_calls(self):
        for key,value in [('max_transport_retries',-1),('max_transport_retries',1.5),
                          ('max_transport_retries',True),('transport_retry_delay_s',float('nan')),
                          ('transport_retry_delay_s',6)]:
            with self.subTest(key=key,value=value):
                cfg={**self.cfg,key:value}
                with self.assertRaises(ValueError):OpenAIClient(cfg)
        self.assertFalse((self.root/'budget.json').exists())


if __name__ == '__main__': unittest.main()
