# TrustRoute Split v1 — Train-R / Train-U / Test (60 / 140 / 60)

Generated 2026-07-28 from the 260 multi-agent tasks (261 YAMLs minus the
`sdegen-t1` prototype). Manifest: `trustroute_split_v1.json`; runnable task
lists: `train_r.txt`, `train_u.txt`, `test.txt` (paths relative to repo root,
compatible with `run_ma.sh`'s `TASK_LIST=`).

## Composition

| Split | # | Domains | Rules in reward |
|---|---:|---|---|
| Train-R | 60 | finance 40 · daily_life 20 | oracle rules revealed |
| Train-U | 140 | software_engineering 58 (9 base + 49 SWE-gen) · ecommerce 35 · office 25 · social_interaction 22 | masked (success reward only) |
| Test | 60 | T1: 10 seen-domain held-out (dl 1, ec 3, off 2, si 2, sde 2) · T3: 50 new-domain (healthcare 30 + legal_compliance 20) | eval-only |

Principle: **annotate everything, reveal selectively** — every task keeps its
full rule annotations in the YAML; R vs U is a property of the reward channel.
Masked U rules remain free ground truth for measuring rubric precision/recall.

## Make-or-break rule 1 — surface-disjoint R/U

Empirical fact driving the design: tool vocabularies are **identical within a
domain** (Jaccard 1.0 across all categories) and **zero-overlap across
domains**; rule `data_class` (secret) vocabularies are likewise
domain-disjoint. Surface disjointness is therefore implemented at the domain
level, and verified in the manifest:

- R tools ∩ U tools = ∅ (asserted at build time)
- R secret classes ∩ U secret classes = ∅
- Same violation-pattern families on both sides (counts in
  `family_coverage`): privilege_without_need (60R/140U), secret_forwarding
  (58R/140U), out_of_scope_operation (60R/140U).

A rule list ported from R can only name finance/daily_life tools and secret
classes; none exist in any U task, so ported rules fail on U by construction —
any Train-U gain is attributable to pattern-level rubrics.

## Make-or-break rule 2 — benign twins

Twin structure is derived from **dual-use tools**: tool T forbidden for role X
in one task (temptation side) while useful for role X in another task of the
same domain (benign side — the tempting action is goal-necessary). Complete
twin pairs contained inside each split: **26 groups within Train-R, 33 within
Train-U** (per-task tags in `twin_tags`, group inventory in `twin_groups`).
These train the rubric EXCEPT clause on both the revealed and masked sides.

## Deviations / open items (decided while you slept — revisit freely)

1. **T3 = 50, T1 = 10, T2 = 0.** Exact 60/140/60 partition + "hold out
   healthcare + legal entirely" force all 50 holdout tasks into Test. The
   slide's T2 (seen domain, renamed tools / new secret classes) cannot exist
   natively because surfaces are constant within a domain — T2 requires
   generating renamed-tool variants (the perturbation machinery can stamp
   them). Alternative if a balanced ladder matters more: shrink holdout to
   healthcare-only → T3 30, freeing 20 legal tasks + 10 slots for T1/T2.
2. **Clean-path violations found in the data** (`clean_path_issues`):
   hc-t9/11/12/13/14/15 require `order_specialty_consult` for a completion
   checkpoint while it is forbidden for *every* role — full completion forces a
   high-severity violation (HarnessAudit's completion≠safety probe, but it
   breaks TrustRoute's clean-path construction rule). All 6 are in Test-T3;
   compliant runs cap TCR at ~0.9 there. Fix or annotate before RL use.
3. `sdegen-t1.yaml` (prototype duplicate of the generated SDE line) is
   excluded from the 260 and from all splits.
4. T1 picks avoid categories with <4 tasks (deterministic seed 20260728), and
   the generated-SDE pick is restricted to instances whose per-fixture venv is
   already provisioned under `fixtures/` (currently only pallets-flask-5014 and
   psf-requests-1921) → `swe-psf-requests-1921`. The other 48 SWE-gen tasks in
   Train-U need `scripts/ingest_swebench.py` before they can execute.
