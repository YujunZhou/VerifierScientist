"""RQ1 report for check libraries from check_loop.py, on run_rq1 eval sets.

  python verifier_bench/eval_checks.py --name cl_sol_eval --sets-from rq1_luna_v1 \
      --library v0=runs/cl_sol/library_v0.json,vK=runs/cl_sol/library_v3.json --repeats 3

Rows: judge, rubric_bank, rubric_bank+scope (hand-written compiled V-OR),
rubric_bank+checks:<label> (judge verdicts UNION the library's checks) and
checks:<label> (the library alone). Judge verdicts come from the shared cache.
Also: each check's precision / fires on each set (does gate precision transfer?).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_loop import TASKS_DIR, Runner, check_stats, union  # noqa: E402
from common import OUT_ROOT, split_meta  # noqa: E402
from compiled_checks import CheckLibrary  # noqa: E402
from llm import spent_usd  # noqa: E402
from run_rq1 import paired_bootstrap, score  # noqa: E402
from verifiers import VerifierSet  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--sets-from", default="rq1_luna_v1")
    ap.add_argument("--library", required=True, help="label=path[,label=path...] (paths relative to OUT_ROOT ok)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--reference", default="rubric_bank+scope")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--budget", type=float, default=45.0)
    ap.add_argument("--drop-judge-codes", default="",
                    help="also report rubric_bank[-codes]+checks: judge channels that failed the precision gate")
    args = ap.parse_args()

    import llm
    llm.BUDGET_USD = args.budget
    out = OUT_ROOT / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    by_id = {}
    for l in open(OUT_ROOT / "pool.jsonl"):
        r = json.loads(l)
        by_id[r["traj_id"]] = r
    ids = json.load(open(OUT_ROOT / "runs" / args.sets_from / "eval_sets.json"))
    sets = {k: [by_id[t] for t in v] for k, v in ids.items()}
    del by_id
    libs = {}
    for item in args.library.split(","):
        label, path = item.split("=", 1)
        p = Path(path) if Path(path).is_absolute() else OUT_ROOT / path
        libs[label] = CheckLibrary.model_validate_json(p.read_text())

    splits = json.load(open(OUT_ROOT / "splits.json"))
    meta = split_meta()
    seed_tasks = [(t, str(TASKS_DIR / meta[t]["path"])) for t in splits["seed"]]
    base_names = ["judge", "rubric_bank", "rubric_bank+scope"]
    vs = VerifierSet(seed_tasks, base_names)
    recs = [r for rs in sets.values() for r in rs]
    preds = defaultdict(lambda: defaultdict(dict))   # verifier -> rep -> traj -> list
    jobs = [(v, k, r) for v in base_names for k in range(args.repeats) for r in recs]
    with ThreadPoolExecutor(args.workers) as ex:
        for (v, k, r), res in zip(jobs, ex.map(lambda j: vs.run(j[0], j[2], j[1]), jobs)):
            preds[v][k][r["traj_id"]] = [x.model_dump() for x in res.violations]
    print(f"judge verdicts ready, spent ${spent_usd():.2f}", flush=True)

    run = Runner(args.workers)
    per_check = {}
    for label, lib in libs.items():
        cp = {c.check_id: run.preds(c, recs) for c in lib.checks}
        only = union({r["traj_id"]: [] for r in recs}, list(cp.values()))
        for k in range(args.repeats):
            preds[f"rubric_bank+checks:{label}"][k] = union(preds["rubric_bank"][k], list(cp.values()))
            preds[f"checks:{label}"][k] = only
            if args.drop_judge_codes:
                drop = set(args.drop_judge_codes.split(","))
                kept = {t: [v for v in p if v["code"] not in drop] for t, p in preds["rubric_bank"][k].items()}
                preds[f"rb-{'-'.join(sorted(drop))}+checks:{label}"][k] = union(kept, list(cp.values()))
        for c in lib.checks:
            per_check[(label, c.check_id)] = {
                s: {kk: v for kk, v in check_stats(c, cp[c.check_id], rs, {r["traj_id"]: [] for r in rs}).items()
                    if kk != "per"}
                for s, rs in sets.items()}
        print(f"checks {label} done, spent ${spent_usd():.2f}", flush=True)

    names = list(preds)
    summary = {v: {s: [score(rs, preds[v][k]) for k in range(args.repeats)] for s, rs in sets.items()} for v in names}
    comps = {v: {s: paired_bootstrap(rs, preds[v], preds[args.reference], args.repeats) for s, rs in sets.items()}
             for v in names if v != args.reference}
    json.dump({"summary": summary, "vs_reference": {"reference": args.reference, **comps},
               "per_check": {f"{a}/{b}": v for (a, b), v in per_check.items()}},
              open(out / "summary.json", "w"), indent=1)

    print(f"\nmean over {args.repeats} repeats; dF1 = per-trajectory F1 minus {args.reference}, 95% CI")
    print(f"{'verifier':26s} {'set':8s} {'n':>3s} {'macroF1':>7s} {'P':>5s} {'R':>5s} {'OT':>5s} {'OR':>5s} {'ID':>5s}"
          f" {'benFA':>5s} | {'dF1 vs ref [95% CI]':>24s}")
    for v in names:
        for s in sets:
            ms = summary[v][s]
            mean = lambda f: sum(f(m) for m in ms) / len(ms)  # noqa: E731
            code = lambda c: mean(lambda m: m[c]["F1"] if c in m else float("nan"))  # noqa: E731
            c_ = comps.get(v, {}).get(s)
            ctxt = f"{c_['diff']:+.3f} [{c_['lo']:+.3f},{c_['hi']:+.3f}]" if c_ else "(reference)"
            print(f"{v:26s} {s:8s} {ms[0]['n']:3d} {mean(lambda m: m['macro_F1']):7.3f} "
                  f"{mean(lambda m: m['all']['P']):5.2f} {mean(lambda m: m['all']['R']):5.2f} "
                  f"{code('V-OT'):5.2f} {code('V-OR'):5.2f} {code('V-ID'):5.2f} "
                  f"{mean(lambda m: m['benign_fa']):5.2f} | {ctxt:>24s}")
    print(f"\nper-check precision (fires) on each set")
    for (label, cid), st in per_check.items():
        fam = next(c.family for c in libs[label].checks if c.check_id == cid)
        print(f"{label:4s} {cid:10s} {fam:15s} " + "  ".join(
            f"{s}: {d['precision']:.2f} ({d['fires']})" for s, d in st.items()))
    print(f"\ntotal spent ${spent_usd():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
