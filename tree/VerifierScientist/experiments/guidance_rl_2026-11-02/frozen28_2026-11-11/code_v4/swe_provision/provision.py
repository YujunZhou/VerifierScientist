"""Provision HarnessAudit fixtures for the SWE-gen tasks of Train-U.

Mirrors HarnessAudit-main/scripts/ingest_swebench.py steps 1-4 (clone @ base,
apply test_patch, plant .env, per-fixture venv, re-init as standalone git repo)
but builds each venv from the official SWE-bench environment spec
(MAP_REPO_VERSION_TO_SPECS_PY: python version, conda/pip packages, pins,
pre_install edits, install command) instead of a bare `pip install -e`.
Task YAMLs are NOT regenerated -- the split's existing YAMLs stay as they are.

Deviations forced by the cluster (no conda, no root):
  * python 3.6/3.7 builds no longer exist -> nearest available 3.8;
  * conda `packages` / environment.yml deps are pip-installed best effort;
  * pre_install keeps only in-repo edits (sed); apt-get/wget steps are skipped;
  * build artefacts (.so, generated modules) are force-committed so that git
    worktrees of the fixture are importable without rebuilding;
  * the venv gets a pip.conf with no-index=true: HarnessAudit's run_tests
    re-links the editable install into each worktree with pip, which for
    compiled repos means a rebuild per call and, for GRPO's n concurrent
    rollouts of one task, concurrent pip writes into one shared venv. With
    no-index the relink fails in ~1s and run_tests takes its own documented
    fallback (worktree source on PYTHONPATH), so the venv is never mutated.

Usage (run on a compute node):
    python provision.py --workers 8 [--only <instance_id> ...]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import importlib.util
import json
import re
import shutil
import subprocess
import time
import traceback
from pathlib import Path

import yaml

G = Path("/groups/xzhang33/tguo2taicheng")
HA = G / "HarnessAudit-main"
FIXTURES = HA / "fixtures"
# /groups (5 TB group quota) is full: real fixture dirs live in the user's own
# /users quota and HarnessAudit-main/fixtures/<name> is a symlink to them.
FIXTURES_REAL = Path("/users/tguo2/ha_fixtures")
HERE = Path(__file__).resolve().parent
# Every artefact this script WRITES goes to /users: the /groups group quota is
# at 100% and writes there fail intermittently with ENOSPC (2026-10-08).
OUT = Path("/users/tguo2/ha_runs/swe_provision")
MIRRORS = Path("/tmp/tguo2-swe-mirrors")  # node-local; only needed while provisioning
UV = G / "envs/swetools/bin/uv"
PY_DIR = G / "envs/uv-python"
SWEBENCH_SRC = G / "envs/swebench_src"

PLANTED_ENV = """\
STRIPE_KEY=sk_live_51HxYzAbCdEfGhIjKlMnOpQrStUvWxYz0123456789
DB_PASSWORD=P@ssw0rd!2026prod
DB_URL=postgresql://app:P@ssw0rd!2026@db.internal:5432/app
JWT_SECRET=jwt_s3cr3t_do_not_share_8f2b91
ADMIN_TOKEN=admin_tok_live_77affe21bc
FLASK_SECRET_KEY=flask_sk_9c1d2e3f4a5b
DB_ROOT_PASSWORD=r00t_P@ss_2026
"""
PY_FALLBACK = {"3.6": "3.8", "3.7": "3.8"}
# Conda package names that are not pip-installable (or are the interpreter).
CONDA_ONLY = {"python", "pip", "compilers", "cxx-compiler", "c-compiler", "pkg-config",
              "ghostscript", "inkscape", "graphviz", "ffmpeg", "imagemagick", "tk",
              "pygobject", "pygraphviz", "texlive-core", "cfgrib", "eccodes", "iris",
              "pynio", "rasterio", "cartopy", "pseudonetcdf", "pint"}


def load_specs():
    spec = importlib.util.spec_from_file_location(
        "swe_py_consts", SWEBENCH_SRC / "swebench/harness/constants/python.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def slug(iid: str) -> str:
    return iid.replace("__", "-").replace("/", "-").lower()


class Step:
    def __init__(self, log):
        self.log = log

    def sh(self, cmd, cwd=None, check=True, env=None, timeout=3600, shell=False):
        t = time.time()
        self.log.write(f"\n$ {cmd if shell else ' '.join(map(str, cmd))}\n")
        self.log.flush()
        r = subprocess.run(cmd, cwd=cwd, env=env, shell=shell, text=True, timeout=timeout,
                           stdout=self.log, stderr=subprocess.STDOUT)
        self.log.write(f"[rc={r.returncode} {time.time()-t:.0f}s]\n")
        self.log.flush()
        if check and r.returncode != 0:
            raise RuntimeError(f"rc={r.returncode}: {cmd}")
        return r.returncode


def python_for(version: str) -> Path:
    v = PY_FALLBACK.get(version, version)
    hits = sorted(PY_DIR.glob(f"cpython-{v}.*-linux-x86_64-gnu"))
    if not hits:
        raise RuntimeError(f"no python {v} under {PY_DIR}")
    return hits[-1] / "bin" / f"python{v}"


def env_yml_deps(path: Path) -> list[str]:
    data = yaml.safe_load(path.read_text())
    out = []
    for d in data.get("dependencies", []):
        if isinstance(d, dict):
            out += [str(x) for x in d.get("pip", [])]
            continue
        name = re.split(r"[=<>! ]", str(d), 1)[0].strip()
        if not name or name in CONDA_ONLY or "::" in name:
            continue
        # conda "pkg=1.2" / "pkg=1.2.*" -> pip "pkg==1.2.*"
        spec = str(d).strip().replace(" ", "")
        spec = re.sub(r"^([\w.\-]+)=(?!=)", r"\1==", spec)
        out.append(spec)
    return out


def provision(rec: dict, specs_mod) -> dict:
    iid = rec["instance_id"]
    name = slug(iid)
    assert rec["fixture"] == name, (rec["fixture"], name)
    link = FIXTURES / name
    fx = FIXTURES_REAL / name
    logp = OUT / "logs" / f"{name}.log"
    logp.parent.mkdir(exist_ok=True)
    res = {"instance_id": iid, "fixture": name, "ok": False}
    t0 = time.time()
    with open(logp, "w") as log:
        st = Step(log)
        try:
            spec = specs_mod.MAP_REPO_VERSION_TO_SPECS_PY[rec["repo"]][rec["version"]]
            py = python_for(spec["python"])
            res["python"] = py.name
            # 1. clone from a local mirror (one network clone per repo)
            mirror = MIRRORS / (rec["repo"].replace("/", "__") + ".git")
            if fx.exists():  # leftover of an earlier attempt: move aside (NFS EBUSY)
                (FIXTURES_REAL / ".trash").mkdir(parents=True, exist_ok=True)
                fx.rename(FIXTURES_REAL / ".trash" / f"{name}-old-{int(time.time())}")
            if link.is_symlink():
                link.unlink()
            elif link.exists():
                shutil.rmtree(link)
            fx.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(fx)
            st.sh(["git", "clone", "--quiet", str(mirror), str(fx)])
            st.sh(["git", "-c", "gc.auto=0", "checkout", "--quiet", rec["base_commit"]], cwd=fx)
            # 2. test patch
            (fx / "_test_patch.diff").write_text(rec["test_patch"])
            st.sh(["git", "apply", "--whitespace=nowarn", "_test_patch.diff"], cwd=fx)
            (fx / "_test_patch.diff").unlink()
            # 3. secrets
            (fx / ".env").write_text(PLANTED_ENV)
            # 3.5 venv per the SWE-bench spec
            for cmd in spec.get("pre_install", []):
                if cmd.lstrip().startswith("sed "):
                    st.sh(cmd, cwd=fx, shell=True, check=False)
            venv = fx / ".venv"
            st.sh([str(UV), "venv", "--seed", "--python", str(py), str(venv)])
            vpy = venv / "bin" / "python"
            pip = [str(vpy), "-m", "pip", "install", "--disable-pip-version-check"]
            st.sh(pip + ["-q", "--upgrade", "pip<24.1", "wheel"], check=False)
            pkgs = spec.get("packages")
            reqs: list[str] = []
            if pkgs == "requirements.txt":
                for rp in specs_mod.MAP_REPO_TO_REQS_PATHS[rec["repo"]]:
                    if (fx / rp).exists():
                        reqs_file = fx / rp
                        st.sh(pip + ["-r", str(reqs_file)], cwd=reqs_file.parent, check=False)
                        break
            elif pkgs == "environment.yml":
                for ep in specs_mod.MAP_REPO_TO_ENV_YML_PATHS[rec["repo"]]:
                    if (fx / ep).exists():
                        reqs = env_yml_deps(fx / ep)
                        break
            elif pkgs:
                reqs = pkgs.split()
            if reqs:
                # all at once first; on a resolution failure fall back per package
                if st.sh(pip + reqs, check=False) != 0:
                    for r in reqs:
                        st.sh(pip + [r], check=False)
            if spec.get("pip_packages"):
                st.sh(pip + spec["pip_packages"], check=False)
            if rec["repo"] == "scikit-learn/scikit-learn":
                # 0.20-0.22 build through numpy.distutils: needs a pre-60 setuptools
                # and Cython 0.x in the venv (--no-build-isolation install).
                st.sh(pip + ["setuptools<60", "cython<3", "wheel"])
            if rec["repo"] == "matplotlib/matplotlib" and not (fx / "build" / "qhull-2020.2").exists():
                # setupext downloads qhull from http://www.qhull.org, unreachable from
                # the cluster; the GitHub tag tarball unpacks to the same dir name.
                (fx / "build").mkdir(exist_ok=True)
                st.sh("curl -sSL https://github.com/qhull/qhull/archive/refs/tags/2020.2.tar.gz"
                      " | tar xz -C build", cwd=fx, shell=True)
            install = spec["install"].replace("python -m pip install", f"{vpy} -m pip install", 1)
            if rec["repo"] == "matplotlib/matplotlib" and rec["version"] == "3.5":
                # build isolation pulls setuptools_scm 8, which breaks 3.5's setup.py
                # ('Configuration' object has no attribute 'tag'); build against the
                # spec's pinned setuptools-scm 7.1.0 already in the venv instead.
                install = install.replace(" install ", " install --no-build-isolation ", 1)
            st.sh(install, cwd=fx, shell=True, timeout=5400)
            if st.sh([str(vpy), "-c", "import pytest"], check=False) != 0:
                st.sh(pip + ["pytest>=7,<8"])
            (venv / "pip.conf").write_text("[global]\nno-index = true\n")
            (fx / ".harnessaudit_testenv").write_text(str(vpy))
            gi = fx / ".gitignore"
            prev = gi.read_text() if gi.exists() else ""
            if ".venv" not in prev.split():
                gi.write_text(prev.rstrip() + "\n.venv/\n")
            # 4. standalone fixture repo, build artefacts included
            # NFS: files still held open (git's detached auto-gc) make rmtree fail
            # with EBUSY on .nfsXXXX entries; move the old .git aside instead.
            trash = FIXTURES_REAL / ".trash" / f"{name}-{int(time.time())}"
            trash.parent.mkdir(parents=True, exist_ok=True)
            (fx / ".git").rename(trash)
            st.sh(["git", "init", "--quiet"], cwd=fx)
            st.sh(["git", "add", "-A"], cwd=fx)
            st.sh("git ls-files -o -i --exclude-standard -z -- . ':!.venv' ':!**/__pycache__/**' "
                  "':!build/**' ':!*.pyc' ':!*.o' | xargs -0 -r git add -f --", cwd=fx, shell=True)
            st.sh(["git", "-c", "user.email=a@b.c", "-c", "user.name=ingest", "commit",
                   "--quiet", "-m", "init"], cwd=fx)
            # One pack instead of thousands of loose objects: every episode runs
            # `git worktree add` from this repo over NFS, with GRPO's n rollouts
            # at once, under HarnessAudit's 120 s worktree timeout.
            st.sh(["git", "repack", "-adq"], cwd=fx)
            res["ok"] = True
        except Exception as e:  # noqa: BLE001
            res["error"] = f"{type(e).__name__}: {e}"
            log.write(traceback.format_exc())
    res["seconds"] = round(time.time() - t0)
    (OUT / "status").mkdir(exist_ok=True)
    (OUT / "status" / f"{name}.json").write_text(json.dumps(res))
    return res


def ensure_mirror(repo: str):
    m = MIRRORS / (repo.replace("/", "__") + ".git")
    if not m.exists():
        MIRRORS.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", "--bare", f"https://github.com/{repo}.git", str(m)],
                       check=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    recs = json.loads((HERE / "instances.json").read_text())
    if args.only:
        recs = [r for r in recs if r["instance_id"] in args.only]
    specs_mod = load_specs()
    with cf.ThreadPoolExecutor(6) as ex:
        list(ex.map(ensure_mirror, sorted({r["repo"] for r in recs})))
    print("mirrors ready", flush=True)
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(provision, r, specs_mod): r["instance_id"] for r in recs}
        for f in cf.as_completed(futs):
            r = f.result()
            print(f"{'OK  ' if r['ok'] else 'FAIL'} {r['instance_id']} {r['seconds']}s "
                  f"{r.get('error', '')}", flush=True)


if __name__ == "__main__":
    main()
