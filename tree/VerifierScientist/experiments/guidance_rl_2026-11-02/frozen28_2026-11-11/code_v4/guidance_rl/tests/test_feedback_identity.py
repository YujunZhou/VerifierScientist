"""N1 regression: repair feedback describes the actual rules being repaired."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from guidance_rl.core import digest, make_artifact, verify_artifact
from guidance_rl.optimize import (ALIGNED_JUDGE_PROMPT, AlignedVerdict, Equivalence,
                                  Library, optimize, render)


class DistinctJudge:
    config = {'concurrency': 1}

    def __init__(self):
        self.calls = []

    def call(self, instruction, user, schema, **kwargs):
        payload = json.loads(user)
        self.calls.append((schema, payload, kwargs))
        if schema is AlignedVerdict:
            pred = []
            # A fresh seed detects forbidden tools, but incorrectly reports an
            # ordinary status message as sensitive disclosure. The incumbent
            # misses forbidden tools instead. These are genuinely different
            # error populations, not just different artifact identifiers.
            if 'SEED_RULE' in payload['guidance']:
                for a in payload['actions']:
                    if a.get('tool_name') == 'forbidden':
                        pred.append({'seq': a['sequence_num'], 'code': 'V-OT',
                                     'severity': 'high', 'reason': 'forbidden capability'})
                    elif a.get('content') == 'routine status':
                        pred.append({'seq': a['sequence_num'], 'code': 'V-ID',
                                     'severity': 'high', 'reason': 'seed overgeneralization'})
            return AlignedVerdict(violations=pred)
        if schema is Equivalence:
            return Equivalence(equivalent=True, changed_rule_ids=[], explanation='offline fixture')
        if kwargs['tag'] == 'seed':
            return Library(rules=[{'rule_id': 'SEED', 'text': 'SEED_RULE: block forbidden tools.'}])
        if kwargs['tag'] == 'candidate:error_repair':
            # Return the same library; these tests do not assert improvement.
            return Library(rules=payload['rules'])
        raise AssertionError(kwargs)


class FeedbackIdentityTests(unittest.TestCase):
    def inputs(self, refresh=True):
        split = {'propose': ['p'], 'select': ['s'], 'confirm': ['c'], 'train_u': ['u'], 'test': ['t']}
        rows = []
        for tid in ('p', 's', 'c'):
            for i in (0, 1):
                rows.append({'task_id': tid, 'traj_id': tid + str(i), 'domain': 'd',
                    'visible_task': 'task ' + tid, 'resource_tools': [],
                    'actions': [{'sequence_num': 1, 'surface': 'tool_call',
                                 'tool_name': 'forbidden' if i else 'safe'},
                                {'sequence_num': 2, 'surface': 'communication', 'content': 'routine status'}],
                    'gold': [{'seq': 1, 'code': 'V-OT', 'severity': 'high'}] if i else [],
                    **({'annotated_rules': 'PROPOSAL_PRIVATE_RULES'} if tid == 'p' else {})})
        cfg = {'generator_model': 'm', 'judge_model': 'm', 'score_protocol': 'aligned_v2',
               'baseline_start': True, 'judge_replicates': 3, 'bootstrap_samples': 20,
               'candidate_names': (['proposal_seed'] if refresh else []) + ['error_repair']}
        return rows, split, cfg

    def incumbent(self):
        rules = [{'rule_id': 'OLD', 'text': 'OLD_RULE: existing accepted criterion.'}]
        return make_artifact(guidance=render(rules), judge_model='m', judge_prompt=ALIGNED_JUDGE_PROMPT,
                             policy='previous', score_protocol='aligned_v2', provenance={'rules': rules})

    def assert_binding(self, out, api, expected_kind, expected_guidance):
        record = json.loads((out / 'proposal_feedback.json').read_text())
        source = verify_artifact(record['source_artifact'])
        self.assertEqual(record['source_artifact_id'], source['artifact_id'])
        self.assertEqual(record['source_guidance_digest'], digest(source['guidance']))
        self.assertEqual(record['source_kind'], expected_kind)
        self.assertIn(expected_guidance, source['guidance'])
        self.assertEqual(set(record['predictions']), {'p0', 'p1'})
        payload = next(p for _, p, k in api.calls if k['tag'] == 'candidate:error_repair')
        self.assertEqual(render(payload['rules']), source['guidance'])
        self.assertEqual(payload['feedback_source_artifact_id'], source['artifact_id'])
        self.assertEqual(payload['proposal_errors'], record['examples'])
        self.assertTrue(record['examples'])
        self.assertTrue(all(e['task_id'] == 'p' for e in record['examples']))
        for schema, p, kwargs in api.calls:
            if schema is AlignedVerdict:
                self.assertNotIn('PROPOSAL_PRIVATE_RULES', json.dumps(p))
                if p['task'] == 'task p':
                    self.assertEqual(p['guidance'], source['guidance'])
        return record

    def test_empty_baseline_errors_do_not_repair_fresh_seed(self):
        rows, split, cfg = self.inputs(); api = DistinctJudge()
        with TemporaryDirectory() as tmp:
            out = Path(tmp); optimize(rows, split, cfg, api, out, 0, 'current')
            # First assert behavioral difference, so the pre-fix test fails on
            # incorrect feedback, not just a missing audit metadata field.
            payload = next(p for _, p, k in api.calls if k['tag'] == 'candidate:error_repair')
            self.assertTrue(all(e['false_positives'] == [[2, 'V-ID']] for e in payload['proposal_errors']))
            self.assert_binding(out, api, 'proposal_seed', 'SEED_RULE')

    def test_regenerated_seed_does_not_use_nonempty_incumbent_errors(self):
        rows, split, cfg = self.inputs(); api = DistinctJudge()
        with TemporaryDirectory() as tmp:
            out = Path(tmp); optimize(rows, split, cfg, api, out, 1, 'current', incumbent=self.incumbent())
            payload = next(p for _, p, k in api.calls if k['tag'] == 'candidate:error_repair')
            self.assertTrue(all(e['false_positives'] == [[2, 'V-ID']] for e in payload['proposal_errors']))
            self.assert_binding(out, api, 'proposal_seed', 'SEED_RULE')

    def test_no_seed_feedback_matches_incumbent_rules(self):
        rows, split, cfg = self.inputs(refresh=False); api = DistinctJudge(); incumbent = self.incumbent()
        with TemporaryDirectory() as tmp:
            out = Path(tmp); optimize(rows, split, cfg, api, out, 1, 'current', incumbent=incumbent)
            record = self.assert_binding(out, api, 'incumbent', 'OLD_RULE')
            self.assertEqual(record['source_artifact_id'], incumbent['artifact_id'])
            self.assertTrue(all(e['misses'] == [[1, 'V-OT']] for e in record['examples']))
            self.assertFalse(any(k['tag'] == 'seed' for _, _, k in api.calls))

    def test_old_partial_aligned_identity_is_rejected_before_calls(self):
        rows, split, cfg = self.inputs(); api = DistinctJudge()
        old_identity = digest({'records': rows, 'split': split, 'config': cfg,
                               'round': 0, 'policy': 'current', 'parent': None})
        with TemporaryDirectory() as tmp:
            out = Path(tmp); marker = out / 'input.json'
            marker.write_text(json.dumps({'identity': old_identity}))
            original = marker.read_bytes()
            with self.assertRaises(ValueError): optimize(rows, split, cfg, api, out, 0, 'current')
            self.assertEqual(marker.read_bytes(), original)
            self.assertEqual(api.calls, [])

    def test_same_feedback_protocol_completed_round_resumes_without_calls(self):
        rows, split, cfg = self.inputs(); api = DistinctJudge()
        with TemporaryDirectory() as tmp:
            out = Path(tmp); artifact = optimize(rows, split, cfg, api, out, 0, 'current')
            marker = json.loads((out / 'input.json').read_text())
            self.assertEqual(marker['proposal_feedback_protocol'], 'artifact_matched_v1')
            calls = len(api.calls)
            self.assertEqual(optimize(rows, split, cfg, api, out, 0, 'current'), artifact)
            self.assertEqual(len(api.calls), calls)
            self.assertTrue(all('artifact_matched_v1' in k['salt'] for _, _, k in api.calls))

    def test_incumbent_rules_must_reconstruct_actual_judged_guidance(self):
        rows, split, cfg = self.inputs(refresh=False); api = DistinctJudge()
        bad = make_artifact(guidance='Different text than its stored rules', judge_model='m',
            judge_prompt=ALIGNED_JUDGE_PROMPT, policy='previous', score_protocol='aligned_v2',
            provenance={'rules': [{'rule_id': 'OLD', 'text': 'OLD_RULE'}]})
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'feedback source guidance'):
                optimize(rows, split, cfg, api, Path(tmp), 1, 'current', incumbent=bad)
            self.assertEqual(api.calls, [])


if __name__ == '__main__': unittest.main()
