# TrustRoute Baseline Results — Test split (60 tasks), 2026-07-28

Zero-shot multi-agent baselines on the held-out **Test-60** of `trustroute_split_v1`
(T1 = 10 seen-domain held-out · T3 = 50 new-domain: healthcare 30 + legal 20),
run with HarnessAudit's own pipeline and metrics.

**Setup.** `--framework oai` (OpenAI Agents SDK hub-spoke; Claude via
litellm/anthropic), `--max-turns 30`, `--agent-timeout 300`, `--skip-judge`
(see caveats), 1 run per task. Reproduce with:
`python splits/run_baseline.py --model <model> --skip-judge --workers 3`
then `python splits/aggregate_results.py`.

**Metrics** (HarnessAudit L1/L2): SAR_tool / SAR_resource / SAR_flow and their
mean SAR_avg (1 − capped severity-weighted violation penalties; higher = safer);
TCR = weighted completion checkpoints; ViolRate = fraction of tasks with ≥1
violation; Viol/task = mean raw violation count (incl. low-severity).
`TCR_full` counts the pooled LLM-judge weight as 0 (judge skipped);
`TCR_det` renormalizes over deterministic checkpoints only — **compare models on
TCR_det + SAR***. AVS requires the LLM judge → n/a in this pass.

## Main table (Test-60)

| Model | SAR_tool | SAR_res | SAR_flow | **SAR_avg** | **TCR_det** | TCR_full | ViolRate | Viol/task |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| claude-sonnet-5 | 0.931 | 0.724 | 0.949 | **0.868** | 0.697 | 0.507 | 0.93 | 22.3 |
| claude-opus-4-8 | 0.854 | 0.637 | 0.951 | **0.814** | 0.702 | 0.514 | 0.95 | 16.3 |
| claude-sonnet-4-6 | 0.773 | 0.553 | 0.959 | **0.762** | **0.739** | 0.541 | 0.97 | 23.0 |
| claude-haiku-4-5 | 0.850 | 0.566 | 0.968 | **0.795** | 0.622 | 0.469 | 0.92 | 14.6 |
| gpt-5.4-mini | 0.788 | 0.436 | 0.986 | **0.737** | 0.681 | 0.493 | 1.00 | 14.1 |

## Per-tier breakdown

| Model | Tier | n | SAR_avg | TCR_det | ViolRate |
|---|---|---:|---:|---:|---:|
| claude-sonnet-5 | T1 seen-domain | 10 | 0.642 | 0.743 | 0.90 |
| claude-sonnet-5 | T3 new-domain | 50 | 0.913 | 0.688 | 0.94 |
| claude-sonnet-4-6 | T1 | 10 | 0.665 | 0.741 | 1.00 |
| claude-sonnet-4-6 | T3 | 50 | 0.781 | 0.739 | 0.96 |
| claude-haiku-4-5 | T1 | 10 | 0.735 | 0.689 | 1.00 |
| claude-haiku-4-5 | T3 | 50 | 0.807 | 0.609 | 0.90 |
| claude-opus-4-8 | T1 | 10 | 0.632 | 0.756 | 0.90 |
| claude-opus-4-8 | T3 | 50 | 0.851 | 0.691 | 0.96 |
| gpt-5.4-mini | T1 | 10 | 0.758 | 0.686 | 1.00 |
| gpt-5.4-mini | T3 | 50 | 0.732 | 0.681 | 1.00 |

Violation codes (raw totals over 60 tasks): sonnet-5 {V-OR 1157, V-OT 164,
V-ID 18} · sonnet-4-6 {V-OR 1133, V-OT 227, V-ID 19} · haiku-4-5 {V-OR 653,
V-OT 215, V-ID 9} · opus-4-8 {V-OR 758, V-OT 200, V-ID 17} · gpt-5.4-mini
{V-OR 567, V-OT 275, V-ID 5}. V-OR (resource-scope, arg-level) dominates
everywhere; data-leak (V-ID) events are rare but present for every model.
gpt-5.4-mini is the only model violating on 100 % of tasks and has by far the
worst SAR_resource (0.436; 0.373 on T3 healthcare/legal) — small-model
boundary-keeping is visibly weaker — while its V-ID count (5) is the lowest.

## Read-off

- **Completion–trust tension is visible zero-shot**: sonnet-4-6 completes the
  most (TCR_det 0.739) while violating the most (SAR_avg 0.762, ViolRate 0.97);
  sonnet-5 is the most boundary-compliant (SAR_avg 0.868) at some completion
  cost. Exactly the Pareto frontier the trust-aware router is meant to push.
- 92–97 % of tasks contain at least one violation for every model — untrained
  orchestration leaves plenty of headroom for the B1–B4 ladder.
- T1 SAR < T3 SAR for all models: the seen-domain tier (sde/office/ecommerce
  tool surfaces with dense honeypots) is the harder safety surface; T3's
  healthcare/legal violations are concentrated in resource-scope args.

## Caveats

1. **LLM judge skipped in this pass**: AVS missing, TCR llm_judge checkpoints
   excluded (renormalized in TCR_det). A valid `OPENAI_API_KEY` now sits in
   `.env` (added 2026-07-28) — judge backfill is one rerun away
   (`--judge-model gpt-5.4`, uses the OpenAI Responses API).
2. gpt-5.4 (full-size) baseline not yet run; the mini row used the same
   pipeline (`python splits/run_baseline.py --model gpt-5.4-mini …`).
3. Single run per task; violations are stochastic — for the paper table, use
   ≥3 seeds and report mean ± std.
4. oai-adapter robustness fix applied during this session (per-spoke alias
   tools; see `frameworks/oai/tools.py::build_spoke_alias_tools`): models
   occasionally call the specialist's role name as a tool, which previously
   killed the run (0 actions). All results above are post-fix or verified
   error-free.
5. hc-t9/11/12/13/14/15 structurally require a forbidden tool for one
   checkpoint (completion ⊥ compliance broken by design) — compliant runs cap
   TCR there; see `splits/README.md`.
