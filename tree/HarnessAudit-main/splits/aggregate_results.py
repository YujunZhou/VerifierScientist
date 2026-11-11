"""Aggregate per-task result JSONs into the TrustRoute baseline table.

Reads multi_agent/results_trustroute/oai/<model>/<task>_*.json, joins with
splits/trustroute_split_v1.json for tier labels, and prints a markdown table
(overall + per test tier).

TCR note: runs executed with --skip-judge score the pooled llm_judge portion
as 0. We therefore report:
  - tcr_full   : TCR over the full checkpoint weight (llm part counted as 0)
  - tcr_det    : deterministic checkpoints only, renormalized by their weight
"""
import glob
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "multi_agent" / "results_trustroute" / "oai"

manifest = json.load(open(REPO / "splits" / "trustroute_split_v1.json"))
tier_of = {t["task_id"]: t["test_tier"] for t in manifest["tasks"] if t["split"] == "test"}
domain_of = {t["task_id"]: t["domain"] for t in manifest["tasks"]}
test_ids = set(tier_of)


def load_model(model_dir: Path) -> dict[str, dict]:
    """One report per task: prefer error-free runs, then the most recent mtime."""
    candidates = defaultdict(list)
    for f in glob.glob(str(model_dir / "*.json")):
        r = json.load(open(f))
        tid = r.get("task_id")
        if tid in test_ids:
            candidates[tid].append((not r.get("error"), Path(f).stat().st_mtime, len(candidates[tid]), r))
    return {tid: max(runs)[3] for tid, runs in candidates.items()}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def summarize(rows: list[dict]) -> dict:
    det_scores = []
    for r in rows:
        w = r.get("deterministic_completion_weight") or 0.0
        s = r.get("deterministic_completion_score")
        if s is not None and w > 0:
            det_scores.append(s)
    return {
        "n": len(rows),
        "sar_tool": mean([r.get("sar_tool") for r in rows]),
        "sar_resource": mean([r.get("sar_resource") for r in rows]),
        "sar_flow": mean([r.get("sar_flow") for r in rows]),
        "sar_avg": mean([r.get("sar_avg") for r in rows]),
        "tcr_full": mean([r.get("tcr") for r in rows]),
        "tcr_det": mean(det_scores),
        "avs": mean([r.get("avs") for r in rows]),
        "viol_rate": mean([1.0 if (r.get("n_violations") or 0) > 0 else 0.0 for r in rows]),
        "viol_per_task": mean([float(r.get("n_violations") or 0) for r in rows]),
        "errors": sum(1 for r in rows if r.get("error")),
    }


def main() -> None:
    models = sorted(d.name for d in RESULTS.iterdir() if d.is_dir() and not d.name.startswith("_"))
    print(f"test tasks expected: {len(test_ids)}\n")

    header = ("| Model | Split | n | SAR_tool | SAR_res | SAR_flow | SAR_avg | "
              "TCR_full | TCR_det | AVS | ViolRate | Viol/task |")
    sep = "|" + "---|" * 12
    print(header); print(sep)
    for m in models:
        per_task = load_model(RESULTS / m)
        missing = test_ids - set(per_task)
        groups = defaultdict(list)
        for tid, r in per_task.items():
            groups["ALL"].append(r)
            groups[tier_of[tid]].append(r)
        for g in ["ALL", "T1_seen_domain_seen_surface", "T3_new_domain"]:
            if g not in groups:
                continue
            s = summarize(groups[g])
            label = {"ALL": "Test-60", "T1_seen_domain_seen_surface": "T1 (10)",
                     "T3_new_domain": "T3 (50)"}[g]
            print(f"| {m} | {label} | {s['n']} | {fmt(s['sar_tool'])} | "
                  f"{fmt(s['sar_resource'])} | {fmt(s['sar_flow'])} | {fmt(s['sar_avg'])} | "
                  f"{fmt(s['tcr_full'])} | {fmt(s['tcr_det'])} | {fmt(s['avs'])} | "
                  f"{fmt(s['viol_rate'], 2)} | {fmt(s['viol_per_task'], 2)} |")
        if missing:
            print(f"| {m} | missing | {len(missing)} | " + " | ".join(["—"] * 9) +
                  f" |  {sorted(missing)[:6]}{'...' if len(missing) > 6 else ''}")
    print("\nViolation codes by model (test tasks):")
    for m in models:
        per_task = load_model(RESULTS / m)
        codes = defaultdict(int)
        for r in per_task.values():
            for c, n in (r.get("violations_by_code") or {}).items():
                codes[c] += n
        print(f"  {m}: {dict(sorted(codes.items()))}")


if __name__ == "__main__":
    main()
