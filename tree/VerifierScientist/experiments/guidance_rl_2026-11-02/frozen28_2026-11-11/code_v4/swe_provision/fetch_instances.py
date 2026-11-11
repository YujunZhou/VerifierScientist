"""Collect the SWE-bench Verified records for every SWE-gen task in Train-U."""
import json, sys, yaml
from pathlib import Path
HA = Path("/groups/xzhang33/tguo2taicheng/HarnessAudit-main")
tasks = [l.strip() for l in (HA / "splits/train_u.txt").read_text().split() if "generated" in l]
want = {}
for t in tasks:
    y = yaml.safe_load((HA / t).read_text())
    src = y["metadata"]["source"]
    want[src["instance_id"]] = {"task_yaml": t, "fixture": y.get("fixture") or None, "base_commit": src["base_commit"]}
from datasets import load_dataset
ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
out = []
for d in ds:
    if d["instance_id"] in want:
        rec = dict(d); rec.update(want[d["instance_id"]]); out.append(rec)
assert len(out) == len(want), (len(out), len(want))
Path("instances.json").write_text(json.dumps(out, indent=1))
print(len(out), "instances")
