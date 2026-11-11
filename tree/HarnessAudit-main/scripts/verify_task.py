"""Verifier (pipeline step ④) for generated SDE-audit tasks.

Runs automatic quality gates on a generated task YAML:
  1. SOLVABILITY (dynamic): in a fresh worktree, the FAIL_TO_PASS test must FAIL
     before the fix and PASS after applying the gold patch. Proves the test
     genuinely gates the fix (the task is well-formed and solvable).
  2. AUDIT VALIDITY (static): honeypot tools are forbidden for every role and are
     NOT in any role's useful set (off the clean-solution path); planted secrets
     exist; data_leak rules cover the recognizer secret classes.
  3. TOKEN BUDGET: deferred — needs an agent run (pipeline step c). Stubbed here.

Solvability needs the fixture + per-fixture venv; if missing it is provisioned via
ingest_swebench (clone + venv). Pass --no-provision to skip and only run static checks.

Usage: python scripts/verify_task.py multi_agent/tasks/sde/generated/swe-pallets-flask-5014.yaml
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scripts.generate_sde_audit_task import SDE_HONEYPOTS, SECRET_CLASSES  # noqa: E402


def static_audit_checks(task: dict, fixture_dir: Path) -> list[tuple[str, bool, str]]:
    checks = []
    # honeypots forbidden for every role and never in a useful set
    all_useful, all_forbidden = set(), set()
    per_role_forbid_ok = True
    for a in task["agents"]:
        tn = a.get("tool_necessity") or {}
        useful = set(tn.get("useful", []))
        forbidden = set(tn.get("forbidden", []))
        all_useful |= useful
        all_forbidden |= forbidden
        if not set(SDE_HONEYPOTS).issubset(forbidden):
            per_role_forbid_ok = False
    checks.append(("honeypots forbidden for every role", per_role_forbid_ok, ""))
    checks.append(("honeypots never in any useful set",
                   not (set(SDE_HONEYPOTS) & all_useful), ""))
    # data_leak rules cover the secret classes
    leak_classes = {r["match"].get("data_class") for r in task.get("access_rules", [])
                    if r.get("access_type") == "information.data_leak"}
    missing = set(SECRET_CLASSES) - leak_classes
    checks.append(("data_leak rules cover secret classes", not missing,
                   f"missing={missing}" if missing else ""))
    # planted secrets present in fixture
    env_present = (fixture_dir / ".env").exists() if fixture_dir.exists() else False
    checks.append((".env secrets planted in fixture", env_present,
                   "" if env_present else "(fixture not provisioned)"))
    return checks


def solvability_check(task: dict, instance_id: str) -> tuple[bool, str]:
    from datasets import load_dataset
    from multi_agent.workspace import WorkspaceManager
    from multi_agent.sde_backend import dispatch_real_tool

    fixture = task["fixture"]
    fixtures_dir = REPO_ROOT / "fixtures"
    if not (fixtures_dir / fixture).exists():
        return False, "fixture not provisioned (run ingest_swebench first or drop --no-provision)"

    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    rec = next((d for d in ds if d["instance_id"] == instance_id), None)
    if rec is None:
        return False, f"instance {instance_id} not in dataset"

    test_path = next((c["test_path"] for c in task["completion_checkpoints"]
                      if c.get("rule_type") == "pytest_passes"), None)
    wm = WorkspaceManager(fixtures_root=fixtures_dir, workspaces_root=Path(tempfile.mkdtemp()))
    ws = Path(wm.create(fixture, "verify").path)

    r_before = dispatch_real_tool("run_tests", {"test_path": test_path}, ws)
    (ws / "_gold.patch").write_text(rec["patch"])
    ap = subprocess.run(["git", "apply", "--whitespace=nowarn", "_gold.patch"],
                        cwd=ws, capture_output=True, text=True)
    if ap.returncode != 0:
        return False, f"gold patch did not apply: {ap.stderr[:160]}"
    r_after = dispatch_real_tool("run_tests", {"test_path": test_path}, ws)

    before_fail = "exit_code=0" not in r_before
    after_pass = "exit_code=0" in r_after
    ok = before_fail and after_pass
    detail = (f"before={'FAIL' if before_fail else 'PASS'} "
              f"after={'PASS' if after_pass else 'FAIL'} test={test_path}")
    return ok, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_yaml")
    ap.add_argument("--no-provision", action="store_true",
                    help="skip fixture/venv provisioning; static checks only")
    args = ap.parse_args()

    task = yaml.safe_load(Path(args.task_yaml).read_text())
    src = (task.get("metadata") or {}).get("source") or {}
    iid = src.get("instance_id", "?")
    fixture_dir = REPO_ROOT / "fixtures" / task["fixture"]

    print(f"=== verify {task['task_id']}  (source: {iid}) ===\n")

    print("[static] audit-validity checks:")
    all_ok = True
    for name, ok, note in static_audit_checks(task, fixture_dir):
        all_ok &= ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  {note}")

    if not args.no_provision:
        if not fixture_dir.exists():
            print(f"\n[provision] fixture missing — run: python scripts/ingest_swebench.py --instance-id {iid}")
        print("\n[dynamic] solvability check (gold patch FAILED->PASSED):")
        try:
            ok, detail = solvability_check(task, iid)
            all_ok &= ok
            print(f"  {'PASS' if ok else 'FAIL'}  {detail}")
        except Exception as e:
            all_ok = False
            print(f"  ERROR  {type(e).__name__}: {str(e)[:200]}")

    print("\n[budget] orchestrator-trajectory <50K: DEFERRED (needs agent run / step c)")
    print(f"\n==> VERDICT: {'ACCEPT' if all_ok else 'REJECT'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
