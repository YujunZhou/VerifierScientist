"""Batch-generate N SDE-audit task YAMLs from SWE-bench Verified (review-only).

Fast path: maps each SWE-bench record -> SDE-audit task YAML via the AuditAuthor
generator WITHOUT cloning repos or building venvs. The fixture + per-fixture venv
for any task are provisioned on demand later via:
    python scripts/ingest_swebench.py --instance-id <id>

Usage: python scripts/batch_generate_swe_tasks.py --n 50
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scripts.generate_sde_audit_task import SDEAuditSpec, build_task  # noqa: E402
from scripts.ingest_swebench import slug, parse_test_path, touched_top_dir  # noqa: E402
from multi_agent.schemas.task import TaskConfig  # noqa: E402


def select_diverse(ds, n: int):
    """Round-robin across repos, preferring few-file / short-patch instances."""
    by_repo = collections.defaultdict(list)
    for d in ds:
        by_repo[d["repo"]].append(d)
    for repo in by_repo:
        by_repo[repo].sort(key=lambda d: (d["patch"].count("diff --git"), len(d["patch"])))
    picked, repos = [], list(by_repo)
    i = 0
    while len(picked) < n and any(by_repo.values()):
        repo = repos[i % len(repos)]
        if by_repo[repo]:
            picked.append(by_repo[repo].pop(0))
        i += 1
    return picked


def record_to_task(rec: dict) -> dict:
    name = slug(rec["instance_id"])
    spec = SDEAuditSpec(
        task_id=f"swe-{name}",
        category=rec["repo"].split("/")[-1],
        fixture=name,
        goal=(rec["problem_statement"] or "").strip()[:1500],
        write_scope=touched_top_dir(rec["patch"]),
        test_path=parse_test_path(rec["FAIL_TO_PASS"]),
        file_contains_checks=[],
        llm_judge_rubric=(
            "Evaluate whether the code change resolves the issue described in the task goal: "
            "(1) does it address the described bug/behavior correctly (0.6); "
            "(2) is it scoped and consistent with codebase conventions (0.4). Return 0-1."
        ),
    )
    task = build_task(spec)
    task["metadata"]["source"] = {
        "benchmark": "SWE-bench_Verified",
        "instance_id": rec["instance_id"],
        "base_commit": rec["base_commit"],
        "repo": rec["repo"],
    }
    return task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "multi_agent/tasks/sde/generated"))
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    picked = select_diverse(ds, args.n)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ok, bad = 0, []
    repo_dist = collections.Counter()
    for rec in picked:
        task = record_to_task(rec)
        try:
            TaskConfig.model_validate(task)
        except Exception as e:
            bad.append((rec["instance_id"], str(e)[:80]))
            continue
        (out_dir / f"{task['task_id']}.yaml").write_text(
            yaml.safe_dump(task, sort_keys=False, allow_unicode=True, width=100))
        ok += 1
        repo_dist[rec["repo"]] += 1

    print(f"\n=== generated {ok}/{len(picked)} SDE-audit task YAMLs -> {out_dir} ===")
    print("repo distribution:")
    for repo, c in repo_dist.most_common():
        print(f"  {repo:28s} {c}")
    if bad:
        print(f"\nschema-invalid ({len(bad)}):")
        for iid, err in bad[:10]:
            print(f"  {iid}: {err}")
    print("\nsample goals:")
    for rec in picked[:6]:
        g = (rec["problem_statement"] or "").strip().replace("\n", " ")[:90]
        print(f"  swe-{slug(rec['instance_id'])}: {g}")


if __name__ == "__main__":
    main()
