# Guidance prompt optimization + segmented Router RL

Primary review document: [Chinese plan, code analysis and evidence](../../VerifierScientist/docs/05_Guidance优化与完整训练框架_Claude审核_2026-11-02.md).

This package is additive: legacy runs, benchmark labels and old training scripts are not rewritten. It separates an explanatory experiment on historical guides from a reusable Train-R-only prompt optimizer. Router training requires a reviewed source/config/input manifest and an externally supplied approval file.

```bash
# From /groups/xzhang33/tguo2taicheng
export PYTHONPATH="$PWD/harnessaudit_verl"
python3 -m unittest discover -s harnessaudit_verl/guidance_rl/tests -v
python3 -m guidance_rl.cli preflight --config VerifierScientist/experiments/guidance_rl_2026-11-02/control/launch.json
```

`prepare` writes derived data; `diagnose` replays historical predictions without APIs. `controlled` and `optimize` call OpenAI with a shared persistent budget. `review-manifest` inventories source and inputs but **does not approve execution**. `run` requires `--approval` matching the exact manifest, four assigned GPUs, an initial verifier artifact and valid runtime configuration.

Reward = 0.5 predicted safety + 0.5 deterministic completion. Oracle rules produce private audit metrics and Train-R optimization labels only. No automatic channel muting or generated program checks are mixed into these guidance experiments. API failures do not become zero reward or an empty violation verdict.

One frozen worker GPU; three Router GPUs; cumulative training endpoints 7, 14, 21, 28. Train-R calibration uses eight sampled trajectories per task. Test-60 is evaluated separately, never used for guidance selection. Fixed-guidance and adaptive arms are supported via separate configs/run directories; only the adaptive job is queued at present.

GPU training and optimizer-state restoration require on-node validation. CPU unit tests do not establish CUDA compatibility or the method's effectiveness. The review document records the current validation status and limitations.

The user explicitly authorized continued execution on 2026-11-03 within a shared USD 10 API cap. `control/finalize_authorized_launch.py` can materialize this existing authorization only after frozen code/config and all 30 restored fixture environments pass preflight. Missing fixtures or altered code block launch; no task is dropped. `HA_FIXTURES_ROOT` selects new experiment-owned copies; legacy paths remain the default. The fixture launcher isolates test dependencies and disables concurrent mutation of shared environments.

## Gate protocols (added 2026-11-08)

`optimization.gate_protocol` selects how candidate guidance is admitted. `per_replicate_v2` (default, unchanged) requires every judge replicate on its own to show a ranking-accuracy interval above zero and applies the false-alarm and per-code recall constraints regardless of denominator size. `pooled_v3` pools the replicates as extra trajectories of the same task before the task-level paired bootstrap, accepts a candidate when it shows a confirmed gain (interval above zero, effect at least `min_effect`) on ranking accuracy or on the reward-related detection F1 while being non-inferior (margin `noninferiority`) on the other, and additionally requires no error regression (`mae_tolerance`), no worst-domain error regression (`worst_group_tolerance`), no per-code recall regression on codes with at least `min_gold` pooled reference positives, and no false-alarm regression when at least `min_clean` pooled clean trajectories exist. Inactive constraints are reported in the decision. Parameters live in `optimization.gate_parameters`; call counts per round are identical for both protocols. The 2026-11-08 offline re-judgment of the adaptive28 rounds is in `adaptive28_2026-11-04/offline_gate_review_2026-11-08/`.

## Repair draws (added 2026-11-09)

`optimization.repair_draws` (integer 1-8, default 1 = historical behaviour) asks the proposer for that many
independent `error_repair` rewrites of the same feedback payload. Draw 0 keeps the historical candidate name
and generation salt; draws `i>=1` are named `error_repair_d{i}`, use salt `<generation_salt>:draw-{i}` and
are written to `error_repair_d{i}.json` with `draw`/`repair_draws` fields. All draws are judged on the select
split like any other candidate and compete in `select_candidate`; identical draws are deduplicated. The
confirm stage still judges only the single selection winner. `selection.json` records `repair_draws`.

Motivation: on the step-21 confirm set, three draws of one identical repair prompt (same rules, same 12
feedback examples) spanned detection F1 0.41-0.55 and ranking accuracy 0.70-0.76 against the same
trajectories, i.e. a single draw made each round a lottery
(`offline_gate_review_2026-11-08/design_experiments_2026-11-09/design_api_results.json`). Cost per extra
draw: 240 judge calls (80 select trajectories x 3 replicates). Tests: `tests/test_pooled_gate.py`
(`test_repair_draws_enter_selection_as_separate_candidates`).

## Reward judge replicates (added 2026-11-09)

Top-level training config key `reward_judge_replicates` (integer 1-5, default 1 = historical behaviour) is copied
into the per-attempt reward config for `mode: train`. The reward then judges every trajectory that many times
(`judge_record(..., rep)` with rep 0..k-1, independent cache keys) and uses the **mean** safety score as
`verifier_sar`. The audit record keeps `prediction` = replicate 0 for compatibility and adds
`predictions_all_replicates`; the reward dict adds `judge_replicates` and `verifier_sar_replicates`. A skipped
judgment (transient failure allowance) still scores the trajectory as unsafe. `paid_stage_requirements` multiplies
the planned train-stage calls by this count, and counts extra `repair_draws` in optimization stages.

Evidence: on the step-14 and step-21 confirm sets, averaging three judgments raised the safety score's ranking
accuracy against the oracle by 3-4 points (0.70 -> 0.74 without guidance, 0.75 -> 0.79 with the round-1 library)
and lowered its error (`offline_gate_review_2026-11-08/design_experiments_2026-11-09/offline_design_analysis.json`).
Left at 1 in the v6 draft so the next arms stay comparable with the November no-guidance run; 2 is the suggested ablation.

## Recall veto by code (added 2026-11-09)

`gate_parameters.recall_constraint_codes` (list of violation codes, default `null` = all codes) limits the
per-code recall veto in `gate_pooled` and in pooled selection to the listed codes; the others are reported under
`inactive_constraints` as `recall:<code>:not_a_reward_code`. The v6 draft lists `V-OT`, `V-IC`, `V-ID`, i.e. the
codes that enter the safety score and the safety-related F1 (`V-OR` is excluded from both by design). Selection
now also receives `min_gold`/`min_clean` from `gate_parameters` instead of hard-coded 30/20. Record:
`fix_2026-11-09_recall_codes.json`.

## Audit changes (added 2026-11-09, second pass)

Three opt-in settings; defaults reproduce the earlier behaviour. Record: `fix_2026-11-09_audit_recall_veto_feedback_balance.json`.

- `gate_parameters.recall_veto`: `'point'` (default) vetoes when a code's pooled recall drops by more than
  `recall_tolerance`; `'confirmed'` also requires the task-bootstrap interval of the recall difference
  (`paired_interval(..., metric='recall:<code>')`) to lie below zero. Without it, a library compared with a second
  judging of itself was vetoed on V-ID (38 positives) in a third of comparisons. In `'confirmed'` mode pooled
  selection does not apply the per-code recall check; the confirmation gate does, once, with intervals.
- `optimization.feedback_selection`: `'stratified_v1'` (default) or `'balanced_v2'`. The default walks strata in
  sorted order, so false-positive strata fill the 12-example cap before miss strata are reached (8-10 false
  positives per round in every stored round). `'balanced_v2'` lets (direction, code) pairs take turns, misses
  first, rotating domains.
- The v6 draft uses both, keeps the recall veto on **all** codes (the restriction to V-OT/V-IC/V-ID did not hold
  up out of sample), and lists only `error_repair` in `candidate_names`: with `proposal_seed` listed and a
  non-empty incumbent, every round proposes a fresh library and repairs that one instead of the incumbent.
