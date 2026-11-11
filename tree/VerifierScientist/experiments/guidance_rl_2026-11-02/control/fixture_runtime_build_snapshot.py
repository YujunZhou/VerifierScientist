"""Bounded isolated replay of the historical SWE environment provisioner.

No API calls, training, source-manifest writes, or automatic publication. The
historical provisioner is loaded without executing main(), and every writable
root is redirected here. Gold patches are only used by optional validation in
disposable worktrees; never in published fixture repositories.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[2]
STORE = EXP / "fixture_envs"
AUDIT = EXP / "audit/fixture_runtime"
ORIGINAL = ROOT / "harnessaudit_verl/swe_provision"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def records():
    required = json.loads((EXP / "audit/fixture_recovery_inputs.json").read_text())["required"]
    return {r["fixture"]: r for r in json.loads((ORIGINAL / "instances.json").read_text())
            if r["fixture"] in required}


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def inventory():
    original = load(ORIGINAL / "provision.py", "original_provision")
    specs = original.load_specs()
    evidence = {}
    for filename in ("validation_main.json", "validation.json", "validation_mpl23476.json"):
        for row in json.loads((ROOT / "ha_runs/swe_provision" / filename).read_text()):
            evidence[row["fixture"]] = row
    rows = []
    for name, rec in records().items():
        spec = specs.MAP_REPO_VERSION_TO_SPECS_PY[rec["repo"]][rec["version"]]
        py = original.python_for(spec["python"])
        proc = subprocess.run([str(py), "-c", "import sys;print(sys.version.split()[0])"],
                              capture_output=True, text=True, timeout=15)
        old = evidence.get(name, {})
        rows.append({"fixture": name, "repo": rec["repo"], "version": rec["version"],
                     "spec": spec, "python": str(py), "python_available": proc.returncode == 0,
                     "python_actual": proc.stdout.strip(), "old_usable": old.get("usable"),
                     "tests": old.get("tests", []), "runtime_restored": False})
    result = {"fixtures": rows, "non_swe_local_fixtures": ["app-admin", "app-ecommerce", "psf-requests-1921"],
              "api_calls": 0, "ready": False,
              "dependency_equivalence": "Historical specs are available; no complete old pip freeze or readable old venv. Replaying unpinned dependencies is not bit-identical."}
    atomic(AUDIT / "inventory.json", result)
    return result


def worker(name, allow_network, validate, resume=False):
    rec = records()[name]
    original = load(ORIGINAL / "provision.py", "original_provision")
    root = STORE / name
    root.mkdir(parents=True, exist_ok=True)
    # Original provisioner cannot escape these writable roots.
    original.FIXTURES = root / "links"
    original.FIXTURES_REAL = root / "repos"
    original.OUT = AUDIT / name
    original.MIRRORS = root / "mirrors"
    for p in (original.FIXTURES, original.FIXTURES_REAL, original.OUT, original.MIRRORS):
        p.mkdir(parents=True, exist_ok=True)
    restored = EXP / "fixtures" / name
    manifest = json.loads((EXP / "fixtures/restore_manifest.json").read_text())
    if not manifest["fixtures"].get(name, {}).get("source_restored"):
        raise RuntimeError("source restoration must finish before environment replay")
    # Use the restored repo as a read-only clone source, retaining its base commit.
    mirror = original.MIRRORS / (rec["repo"].replace("/", "__") + ".git")
    if not mirror.exists():
        mirror.symlink_to(restored / ".git", target_is_directory=True)
    status_path = original.OUT / "status" / f"{name}.json"
    existing = (original.FIXTURES_REAL / name).exists()
    if existing and not (resume and status_path.exists() and json.loads(status_path.read_text()).get("ok")):
        raise RuntimeError("Build directory exists without successful resumable status; inspect it; never auto-replaced")
    os.environ.update(PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_RETRIES="1", PIP_TIMEOUT="30",
                      PIP_CACHE_DIR=str(STORE / "pip_cache"), UV_CACHE_DIR=str(STORE / "uv_cache"),
                      PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg", OMP_NUM_THREADS="1")
    if not allow_network:
        os.environ.update(PIP_NO_INDEX="1", UV_OFFLINE="1")
    # All commands stay bounded; caller additionally kills the entire process
    # group at the per-fixture wall clock deadline, including compilers.
    old_sh = original.Step.sh
    def bounded_sh(self, cmd, cwd=None, check=True, env=None, timeout=3600, shell=False):
        return old_sh(self, cmd, cwd, check, env, min(timeout, 1200), shell)
    original.Step.sh = bounded_sh
    result = json.loads(status_path.read_text()) if existing else original.provision(rec, original.load_specs())
    result.update(api_calls=0, ready=False, source_root=str(original.FIXTURES_REAL / name),
                  dependency_equivalence="spec_replay_not_exact_old_freeze", published=False)
    if result["ok"]:
        fx = original.FIXTURES_REAL / name
        py = fx / ".venv/bin/python"
        freeze = subprocess.run([str(py), "-m", "pip", "freeze", "--all"],
                                capture_output=True, text=True, timeout=60)
        (original.OUT / "pip_freeze.txt").write_text(freeze.stdout)
        validator = load(ORIGINAL / "validate.py", "original_validate")
        tests = validator.pytest_checks(ROOT / "HarnessAudit-main" / rec["task_yaml"])
        collections = []
        # Match runtime PYTHONPATH fallback without a mutable editable relink.
        env = dict(os.environ, PYTHONPATH=f"{fx}/src:{fx}")
        for test in tests:
            p = subprocess.run([str(py), "-m", "pytest", test, "--collect-only", "-q"],
                               cwd=fx, env=env, capture_output=True, text=True, timeout=300)
            collections.append({"test": test, "returncode": p.returncode,
                                "tail": (p.stdout + p.stderr)[-4000:]})
        result["collection"] = collections
        result["collection_passed"] = bool(collections) and all(c["returncode"] == 0 for c in collections)
        if validate and result["collection_passed"]:
            os.environ["PYTHONPATH"] = os.pathsep.join(map(str, [
                ROOT / "envs/verl08/lib/python3.11/site-packages",
                ROOT / "harnessaudit_verl", ROOT / "HarnessAudit-main"]))
            result["backend_validation_inherited_pythonpath"] = os.environ["PYTHONPATH"]
            # Reuse historical base-fails / gold-passes and two parallel worktrees.
            class IsolatedHA:
                def __truediv__(self, part):
                    return original.FIXTURES if str(part) == "fixtures" else ROOT / "HarnessAudit-main" / part
            validator.HA = IsolatedHA()
            validator.OUT = original.OUT
            validator.WS_ROOT = root / "validation_worktrees"
            result["validation"] = validator.validate(rec, 2)
            result["runtime_validated"] = result["validation"].get("usable", False)
    atomic(original.OUT / "runtime_result.json", result)
    print(json.dumps({k: result.get(k) for k in ("fixture", "ok", "collection_passed", "runtime_validated", "ready", "error")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--only", nargs="+", default=[])
    parser.add_argument("--allow-network", action="store_true", help="Package downloads; no model APIs")
    parser.add_argument("--validate", action="store_true", help="Old base/gold checks in disposable worktrees")
    parser.add_argument("--fixture-timeout", type=int, default=2400)
    parser.add_argument("--resume", action="store_true", help="Revalidate an already successful build; never replaces a failed build")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    AUDIT.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args.worker, args.allow_network, args.validate, args.resume)
        return
    result = inventory()
    print(f"Inventoried {len(result['fixtures'])} SWE environments; no ready manifest is written.")
    names = records()
    if any(n not in names for n in args.only):
        parser.error("--only accepts required SWE fixture slugs only; local apps need a separate validation policy")
    for name in args.only:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", name]
        if args.allow_network: cmd.append("--allow-network")
        if args.validate: cmd.append("--validate")
        if args.resume: cmd.append("--resume")
        logpath = AUDIT / f"{name}.driver.log"
        with logpath.open("w") as log:
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                rc = p.wait(timeout=args.fixture_timeout)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGTERM)
                try: p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
                rc = 124
        atomic(AUDIT / f"{name}.driver.json", {"fixture": name, "returncode": rc,
               "log": str(logpath), "deadline_seconds": args.fixture_timeout, "api_calls": 0})
        print(f"{name}: driver rc={rc}; inspect runtime_result.json (build success is not readiness)")


if __name__ == "__main__":
    main()
