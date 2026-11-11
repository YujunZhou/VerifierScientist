"""Group-structured trajectory sets for the ranking objective (GRPO group = 8).

  gate     held-out Train-R tasks, 8 trajectories each (the loop's gate)
  heldout  the same tasks, 8 OTHER trajectories each (report)
  train_u  --n-train-u Train-U tasks, 8 trajectories each (report)
  test     every Test-60 trajectory in the pool (base + s26 policies; 1-21 per task)

  python verifier_bench/make_rank_sets.py --name rank_sets_v1
  -> runs/<name>/eval_sets.json (heldout/train_u/test), runs/<name>/gate.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT_ROOT  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--n-train-u", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    held, tu, test = defaultdict(list), defaultdict(list), []
    for l in open(OUT_ROOT / "pool.jsonl"):
        r = json.loads(l)
        if r["split"] == "test":
            test.append(r["traj_id"])
        elif r["source"] == "oo_train" and r["train_r_role"] == "heldout":
            held[r["task_id"]].append(r["traj_id"])
        elif r["source"] == "oo_train" and r["split"] == "train_u":
            tu[r["task_id"]].append(r["traj_id"])
    gate, rep_held = [], []
    for t in sorted(held):
        ids = sorted(held[t])
        rng.shuffle(ids)
        k = min(args.group, len(ids) // 2)
        gate += ids[:k]
        rep_held += ids[k:2 * k]
    tasks = sorted(tu)
    rng.shuffle(tasks)
    rep_tu = [x for t in sorted(tasks[: args.n_train_u]) for x in rng.sample(sorted(tu[t]), args.group)]
    out = OUT_ROOT / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"heldout": rep_held, "train_u": rep_tu, "test": sorted(test)}, open(out / "eval_sets.json", "w"), indent=1)
    json.dump({"gate": gate}, open(out / "gate.json", "w"), indent=1)
    print(f"gate {len(gate)}, heldout {len(rep_held)}, train_u {len(rep_tu)}, test {len(test)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
