"""Check that each provisioned SWE fixture behaves correctly at RL time.

Uses exactly the runtime path of an episode: a git worktree from HarnessAudit's
WorkspaceManager, then the task YAML's own pytest_passes checkpoint run through
sde_backend.run_tests. A fixture is USABLE iff
  * at base, the checkpoint test does NOT pass (the task is not pre-solved), and
  * after applying the SWE-bench gold patch to the worktree, it DOES pass
    (the task is solvable and the agent's edits are what gets tested),
with every run_tests call under the 300 s timeout. With --concurrency N, the
gold check runs on N simultaneous worktrees of the same fixture (GRPO launches
n rollouts of one task at once).

Run with the verl08 python (the env the tools run in):
    python validate.py [--only <instance_id> ...] [--workers 8] [--concurrency 1]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

import yaml

G = Path("/groups/xzhang33/tguo2taicheng")
HA = G / "HarnessAudit-main"
HERE = Path(__file__).resolve().parent
# Every artefact this script WRITES goes to /users: the /groups group quota is
# at 100% and writes there fail intermittently with ENOSPC (2026-10-08).
OUT = Path("/users/tguo2/ha_runs/swe_provision")
sys.path.insert(0, str(HA))
from multi_agent.sde_backend import dispatch_real_tool  # noqa: E402
from multi_agent.workspace import WorkspaceManager  # noqa: E402

WS_ROOT = Path("/tmp/tguo2-swe-validate-ws")


def pytest_checks(task_yaml: Path) -> list[str]:
    y = yaml.safe_load(task_yaml.read_text())
    return [c["test_path"] for c in y.get("completion_checkpoints", [])
            if c.get("rule_type") == "pytest_passes"]


def run(ws_path: Path, test_path: str) -> tuple[int | None, float, str]:
    t = time.time()
    out = dispatch_real_tool("run_tests", {"test_path": test_path}, ws_path)
    m = re.search(r"exit_code=(-?\d+)", out)
    return (int(m.group(1)) if m else None), time.time() - t, out


def gold_run(wm, fixture, patch, tests):
    ws = wm.create(fixture, f"val-{uuid.uuid4().hex[:8]}")
    try:
        p = ws.path / "_gold.diff"
        p.write_text(patch)
        rc = subprocess.run(["git", "apply", "--whitespace=nowarn", "_gold.diff"], cwd=ws.path,
                            capture_output=True, text=True)
        p.unlink()
        if rc.returncode != 0:
            return [("apply-failed", 0.0, rc.stderr[-500:])]
        return [run(ws.path, t) for t in tests]
    finally:
        wm.destroy(ws)


def validate(rec: dict, concurrency: int) -> dict:
    try:
        return _validate(rec, concurrency)
    except Exception as e:  # noqa: BLE001 -- one bad fixture must not stop the sweep
        return {"instance_id": rec["instance_id"], "fixture": rec["fixture"], "usable": False,
                "reason": f"{type(e).__name__}: {str(e)[:300]}"}


def _validate(rec: dict, concurrency: int) -> dict:
    fixture = rec["fixture"]
    res = {"instance_id": rec["instance_id"], "fixture": fixture, "usable": False}
    st = OUT / "status" / f"{fixture}.json"
    if not st.exists() or not json.loads(st.read_text())["ok"]:
        res["reason"] = "provision failed"
        return res
    tests = pytest_checks(HA / rec["task_yaml"])
    res["tests"] = tests
    wm = WorkspaceManager(fixtures_root=HA / "fixtures", workspaces_root=WS_ROOT)
    ws = wm.create(fixture, f"val-{uuid.uuid4().hex[:8]}")
    try:
        base = [run(ws.path, t) for t in tests]
    finally:
        wm.destroy(ws)
    with cf.ThreadPoolExecutor(concurrency) as ex:
        golds = list(ex.map(lambda _: gold_run(wm, fixture, rec["patch"], tests), range(concurrency)))
    res["base"] = [(rc, round(s)) for rc, s, _ in base]
    res["gold"] = [[(rc, round(s)) for rc, s, _ in g] for g in golds]
    res["base_tail"] = base[0][2][-600:] if base else ""
    res["gold_tail"] = golds[0][0][2][-600:] if golds and golds[0] else ""
    slow = max([s for _, s, _ in base] + [s for g in golds for _, s, _ in g], default=0)
    if not tests:
        res["reason"] = "no pytest_passes checkpoint"
    elif any(rc == 0 for rc, _, _ in base):
        res["reason"] = "test already passes at base"
    elif not all(rc == 0 for g in golds for rc, _, _ in g):
        res["reason"] = "gold patch does not make the test pass"
    elif slow > 240:
        res["reason"] = f"too slow ({slow:.0f}s)"
    else:
        res["usable"] = True
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--out", default=str(OUT / "validation.json"))
    args = ap.parse_args()
    recs = json.loads((HERE / "instances.json").read_text())
    if args.only:
        recs = [r for r in recs if r["instance_id"] in args.only]
    out = []
    with cf.ThreadPoolExecutor(args.workers) as ex:
        for r in ex.map(lambda rec: validate(rec, args.concurrency), recs):
            out.append(r)
            print(f"{'USABLE' if r['usable'] else 'REJECT'} {r['instance_id']} "
                  f"base={r.get('base')} gold={r.get('gold')} {r.get('reason', '')}", flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"{sum(r['usable'] for r in out)}/{len(out)} usable")


if __name__ == "__main__":
    main()
