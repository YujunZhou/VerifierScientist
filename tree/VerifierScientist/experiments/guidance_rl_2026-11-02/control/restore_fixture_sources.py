"""Restore task sources only; never asserts dependency environments are ready.

Reads audited SWE records excluding gold patches. Never changes benchmark files.
Every subprocess is bounded; reruns skip completed, manifest-recorded sources.
"""
from __future__ import annotations

import ast
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[2]
DEST = EXP / "fixtures"
AUDIT = EXP / "audit"
INPUT = AUDIT / "fixture_recovery_inputs.json"
MANIFEST = DEST / "restore_manifest.json"


def run(args, cwd=None, timeout=600):
    p = subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=timeout,
                       env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    if p.returncode:
        raise RuntimeError(f"{args[:4]}: {p.stderr[-1500:]}")
    return p.stdout.strip()


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def planted_env():
    src = ROOT / "HarnessAudit-main/scripts/ingest_swebench.py"
    tree = ast.parse(src.read_text())
    return next(ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and
                t.id == "PLANTED_ENV" for t in n.targets))


def restore(name, tasks, rec):
    dest = DEST / name
    src = ROOT / "HarnessAudit-main/fixtures" / name
    result = {"fixture": name, "tasks": tasks, "source_restored": False,
              "ready": False, "validation_complete": False,
              "environment": {"marker": None, "available": False,
                              "collection_passed": False,
                              "status": "not_provisioned_or_validated"}}
    start = time.time()
    try:
        # Failed partial directories are preserved for inspection; no destructive resume.
        if dest.exists():
            raise RuntimeError("Destination already exists without completed manifest; inspect before resuming")
        if src.is_symlink():
            if rec is None:
                raise RuntimeError("Missing cached SWE record")
            result.update(source="public_git_base_plus_cached_test_patch",
                          original_symlink=str(src.readlink()), repo=rec["repo"],
                          base_commit=rec["base_commit"], test_patch_sha256=sha(rec["test_patch"]),
                          gold_patch_applied=False,
                          old_fixture_equivalence="not_established: old build edits/artifacts/env unavailable")
            dest.mkdir()
            run(["git", "init", "--quiet", str(dest)])
            run(["git", "-C", str(dest), "fetch", "--quiet", "--depth=1",
                 f"https://github.com/{rec['repo']}.git", rec["base_commit"]])
            run(["git", "-C", str(dest), "checkout", "--quiet", "--detach", "FETCH_HEAD"])
            actual = run(["git", "-C", str(dest), "rev-parse", "HEAD"])
            if actual != rec["base_commit"]:
                raise RuntimeError("Fetched base_commit mismatch")
            patch_path = AUDIT / "fixture_test_patches" / f"{name}.diff"
            patch_path.parent.mkdir(exist_ok=True)
            patch_path.write_text(rec["test_patch"])
            run(["git", "-C", str(dest), "apply", "--check", str(patch_path)])
            run(["git", "-C", str(dest), "apply", "--whitespace=nowarn", str(patch_path)])
            # Exact synthetic benchmark bait, never credentials from a user env.
            (dest / ".env").write_text(planted_env())
            run(["git", "-C", str(dest), "add", "-A"])
            run(["git", "-C", str(dest), "add", "-f", ".env"])
            run(["git", "-C", str(dest), "-c", "user.email=fixture-recovery@local",
                 "-c", "user.name=Fixture Recovery", "commit", "--quiet", "-m",
                 "Cached SWE test patch and synthetic audit fixture bait (no gold fix)"])
            run(["git", "-C", str(dest), "apply", "--reverse", "--check", str(patch_path)])
            result["test_patch_reverse_check"] = True
        else:
            result.update(source="local_git_snapshot", source_path=str(src),
                          base_commit=rec["base_commit"] if rec else None,
                          test_patch_sha256=sha(rec["test_patch"]) if rec else None,
                          gold_patch_applied=False)
            head = run(["git", "-c", f"safe.directory={src}", "-C", str(src), "rev-parse", "HEAD"])
            result["source_git_head"] = head
            result["source_uncommitted_status"] = run(["git", "-c", f"safe.directory={src}", "-C", str(src), "status", "--porcelain"])
            run(["git", "-c", f"safe.directory={src}", "-c", f"safe.directory={src / '.git'}",
                 "clone", "--quiet", "--no-hardlinks", str(src), str(dest)])
            if run(["git", "-C", str(dest), "rev-parse", "HEAD"]) != head:
                raise RuntimeError("Local snapshot HEAD mismatch")
            delta = run(["git", "-c", f"safe.directory={src}", "-C", str(src), "diff", "--binary", "HEAD"])
            if delta:
                delta += "\n"
                patch_path = AUDIT / f"fixture_local_delta_{name}.diff"
                patch_path.write_text(delta)
                run(["git", "-C", str(dest), "apply", "--index", str(patch_path)])
                run(["git", "-C", str(dest), "-c", "user.email=fixture-recovery@local",
                     "-c", "user.name=Fixture Recovery", "commit", "--quiet", "-m",
                     "Preserve existing local fixture tracked changes"])
                result["local_tracked_delta_sha256"] = sha(delta)
        result["git_head"] = run(["git", "-C", str(dest), "rev-parse", "HEAD"])
        result["git_tree"] = run(["git", "-C", str(dest), "rev-parse", "HEAD^{tree}"])
        result["tracked_files_sha256"] = sha(run(["git", "-C", str(dest), "ls-tree", "-r", "HEAD"]))
        marker = dest / ".harnessaudit_testenv"
        if marker.exists():
            result["environment"]["marker"] = marker.read_text().strip()
            result["environment"]["available"] = Path(marker.read_text().strip()).is_file()
        # A real worktree creation validates write access to the new Git metadata.
        wt = EXP / "fixture_worktree_smoke" / name
        wt.parent.mkdir(exist_ok=True)
        run(["git", "-C", str(dest), "worktree", "add", "--quiet", "--detach", str(wt), "HEAD"], timeout=120)
        run(["git", "-C", str(dest), "worktree", "remove", str(wt)], timeout=120)
        result["worktree_smoke_passed"] = True
        result["source_restored"] = True
    except Exception as e:
        result["error"] = str(e)
    result["seconds"] = round(time.time() - start, 1)
    return result


def main():
    inputs = json.loads(INPUT.read_text())
    DEST.mkdir(exist_ok=True)
    prior = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    records = prior.get("fixtures", {})
    data = {"schema_version": 1, "ready": False,
            "required_fixtures": sorted(inputs["required"]), "fixtures": records,
            "warning": "Source recovery is not runtime restoration; no old environment equivalence claimed."}
    def save():
        tmp = MANIFEST.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(MANIFEST)
    save()
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        fs = {pool.submit(restore, n, tasks, inputs["swe_records"].get(n)): n
              for n, tasks in inputs["required"].items()
              if not records.get(n, {}).get("source_restored")}
        for future in cf.as_completed(fs):
            n = fs[future]
            records[n] = future.result()
            save()
            print(n, "SOURCE_OK" if records[n]["source_restored"] else "FAILED",
                  records[n].get("error", "environment still unvalidated"), flush=True)


if __name__ == "__main__":
    main()
