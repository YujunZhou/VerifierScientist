"""Batch-verify generated SDE-audit tasks -> a verification report.

Runs the static audit-validity gate on every task (fast, no provisioning) and the
dynamic solvability gate only on tasks whose fixture is already provisioned
(clone + per-fixture venv). Writes a markdown report + prints a summary.

Full dynamic verification of every task needs per-instance dep provisioning
(SWE-bench install specs); do it on demand via ingest_swebench.py.

Usage: python scripts/batch_verify.py [--dynamic] [--out report.md]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scripts.verify_task import static_audit_checks, solvability_check  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", default=str(REPO_ROOT / "multi_agent/tasks/sde/generated"))
    ap.add_argument("--dynamic", action="store_true",
                    help="also run solvability on provisioned fixtures")
    ap.add_argument("--out", default=str(REPO_ROOT / "../research_notes/verification_report.md"))
    args = ap.parse_args()

    tasks = sorted(Path(args.tasks_dir).glob("swe-*.yaml"))
    rows = []
    n_static_ok = n_provisioned = n_solvable = 0

    for tp in tasks:
        task = yaml.safe_load(tp.read_text())
        fixture_dir = REPO_ROOT / "fixtures" / task["fixture"]
        static = static_audit_checks(task, fixture_dir)
        # .env planting needs a provisioned fixture; don't penalize unprovisioned tasks for it.
        static_ok = all(ok for name, ok, _note in static if not name.startswith(".env"))
        n_static_ok += static_ok
        provisioned = fixture_dir.exists()
        n_provisioned += provisioned
        dyn = "—"
        if args.dynamic and provisioned:
            iid = (task.get("metadata") or {}).get("source", {}).get("instance_id", "")
            try:
                ok, detail = solvability_check(task, iid)
                dyn = "ACCEPT" if ok else f"REJECT ({detail})"
                n_solvable += ok
            except Exception as e:
                dyn = f"ERR {type(e).__name__}"
        rows.append((task["task_id"], task["metadata"]["source"]["repo"],
                     "PASS" if static_ok else "FAIL", "yes" if provisioned else "no", dyn))

    lines = ["# SDE-audit task verification report\n",
             f"- tasks: **{len(tasks)}**",
             f"- static audit-valid: **{n_static_ok}/{len(tasks)}**",
             f"- fixtures provisioned: **{n_provisioned}/{len(tasks)}**"]
    if args.dynamic:
        lines.append(f"- dynamically solvable (of provisioned): **{n_solvable}/{n_provisioned}**")
    lines += ["", "| task | repo | static | provisioned | dynamic |", "|---|---|---|---|---|"]
    for tid, repo, st, pr, dyn in rows:
        lines.append(f"| {tid} | {repo} | {st} | {pr} | {dyn} |")
    report = "\n".join(lines) + "\n"

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(f"static audit-valid: {n_static_ok}/{len(tasks)}  |  provisioned: {n_provisioned}  |  "
          f"solvable: {n_solvable if args.dynamic else 'n/a'}")
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
