"""Index every scored Qwen trajectory with its oracle (gold) violation labels.

Sources (episode dumps written by ha_verl; ``.used.json`` = claimed by the
reward function, i.e. a complete, scored episode):
  oo_train  outcome-only GRPO rollouts on Train-R + Train-U (all 26+ steps)
  base_test Qwen3.5-4B base on Test-60 (runs/qwen3.5-4b-fix)
  s26_test  outcome-only step-26 checkpoint on Test-60

Gold labels are recomputed on the dumped actions WITHOUT the hub's final answer
(training dumps do not store it), the same way for every source, so a verifier
is never asked about an action it cannot see.

Also writes the Train-R seed / held-out split (stratified by domain, fixed seed).

  python verifier_bench/build_pool.py            # -> $OUT_ROOT/pool.jsonl, splits.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT_ROOT, ROOT, gold_labels, split_meta  # noqa: E402

SOURCES = {
    # episodes/ (not _prev) = the retry after the hold ended (10-10 10:29): the
    # spoke server was already gone, so its delegations failed with
    # "[ERROR] Connection error." -- infrastructure noise, left out.
    "oo_train": ["/users/tguo2/ha_runs/qwen3.5-4b-outcome-only/episodes_prev/*/*.used.json"],
    "base_test": [str(ROOT / "harnessaudit_verl/runs/qwen3.5-4b-fix/episodes/*.used.json"),
                  str(ROOT / "harnessaudit_verl/runs/qwen3.5-4b-fix/episodes_prev/*/*.used.json")],
    "s26_test": ["/users/tguo2/ha_runs/qwen3.5-4b-outcome-only-eval-s26/episodes/*.used.json"],
}


def seed_heldout(seed: int, frac_seed: float) -> dict[str, list[str]]:
    by_dom = defaultdict(list)
    for tid, t in split_meta().items():
        if t["split"] == "train_r":
            by_dom[t["domain"]].append(tid)
    rng = random.Random(seed)
    out = {"seed": [], "heldout": []}
    for dom in sorted(by_dom):
        ids = sorted(by_dom[dom])
        rng.shuffle(ids)
        k = round(len(ids) * frac_seed)
        out["seed"] += ids[:k]
        out["heldout"] += ids[k:]
    return {k: sorted(v) for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frac-seed", type=float, default=2 / 3)   # 40 / 20 of Train-R
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after N dumps per source")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sh = seed_heldout(args.seed, args.frac_seed)
    json.dump({"seed_rng": args.seed, **sh}, open(out / "splits.json", "w"), indent=1)
    role = {t: "seed" for t in sh["seed"]} | {t: "heldout" for t in sh["heldout"]}

    meta = split_meta()
    n_ok, n_bad, stats = 0, Counter(), Counter()
    seen_runs = set()
    with open(out / "pool.jsonl", "w") as fo:
        for src, patterns in SOURCES.items():
            files = sorted({f for p in patterns for f in glob.glob(p)})
            if args.limit:
                files = files[: args.limit]
            for f in files:
                d = json.load(open(f))
                tid, run_id = d.get("task_id"), d.get("run_id")
                if tid not in meta or not d.get("actions") or (src, run_id) in seen_runs:
                    n_bad["no_meta/empty/dup"] += 1
                    continue
                seen_runs.add((src, run_id))
                try:
                    gold = gold_labels(d["task_path"], d["actions"])
                except Exception as e:  # noqa: BLE001
                    n_bad[type(e).__name__] += 1
                    continue
                m = meta[tid]
                rec = {
                    "traj_id": f"{src}:{run_id}",
                    "source": src,
                    "task_id": tid,
                    "task_path": d["task_path"],
                    "split": m["split"],
                    "train_r_role": role.get(tid),
                    "test_tier": m.get("test_tier"),
                    "domain": m["domain"],
                    "mtime": os.path.getmtime(f),
                    "n_actions": len(d["actions"]),
                    "actions": d["actions"],
                    "gold": gold,
                }
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_ok += 1
                stats[(src, m["split"])] += 1
    print(f"wrote {n_ok} trajectories to {out/'pool.jsonl'}; skipped {dict(n_bad)}")
    for k, v in sorted(stats.items()):
        print(f"  {k[0]:10s} {k[1]:8s} {v}")
    print(f"Train-R seed {len(sh['seed'])} / held-out {len(sh['heldout'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
