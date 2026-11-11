"""Run a task list through the multi_agent CLI with a bounded worker pool.

macOS ships bash 3.2 (no `wait -n`), which breaks run_ma.sh's throttling —
this driver replaces it for the TrustRoute baseline runs.

Usage:
  python splits/run_baseline.py --model claude-sonnet-5 [--task-list splits/test.txt]
      [--workers 3] [--skip-judge] [--judge-model gpt-5.4]
"""
import argparse
import glob
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


def run_one(task_path: str, args) -> tuple[str, str]:
    task_id = Path(task_path).stem
    model_dir = args.model.replace("/", "__")
    result_scope = REPO / args.output / "oai" / model_dir
    if not args.no_skip_existing and glob.glob(str(result_scope / f"{task_id}_*.json")):
        return task_id, "skipped"
    log_dir = REPO / args.output / "_run_logs" / "oai" / model_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, "-m", "multi_agent", "run", task_path,
        "--framework", "oai",
        "--model", args.model,
        "--output", args.output,
        "--trace-dir", args.trace_dir,
        "--max-turns", str(args.max_turns),
        "--agent-timeout", str(args.agent_timeout),
        "--judge-model", args.judge_model,
    ]
    if args.skip_judge:
        cmd.append("--skip-judge")
    log_file = log_dir / f"{task_id}.log"
    t0 = time.time()
    with open(log_file, "w") as lf:
        proc = subprocess.run(cmd, cwd=REPO, stdout=lf, stderr=subprocess.STDOUT,
                              timeout=args.agent_timeout + 600)
    status = "ok" if proc.returncode == 0 else f"FAIL rc={proc.returncode}"
    return task_id, f"{status} ({time.time() - t0:.0f}s)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task-list", default="splits/test.txt")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--judge-model", default="gpt-5.4")
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--agent-timeout", type=int, default=300)
    ap.add_argument("--output", default="multi_agent/results_trustroute")
    ap.add_argument("--trace-dir", default="multi_agent/traces_trustroute")
    ap.add_argument("--no-skip-existing", action="store_true")
    args = ap.parse_args()

    tasks = [ln.strip() for ln in open(REPO / args.task_list) if ln.strip()]
    print(f"model={args.model} tasks={len(tasks)} workers={args.workers} "
          f"skip_judge={args.skip_judge}", flush=True)
    failures = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_one, t, args): t for t in tasks}
        for fut in as_completed(futs):
            task = futs[fut]
            try:
                task_id, status = fut.result()
            except Exception as exc:  # timeout etc.
                task_id, status = Path(task).stem, f"EXC {exc}"
            done += 1
            print(f"[{done}/{len(tasks)}] {task_id}: {status}", flush=True)
            if "ok" not in status and status != "skipped":
                failures.append(task_id)
    print(f"finished. failures={len(failures)}: {failures}", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
