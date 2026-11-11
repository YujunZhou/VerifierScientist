#!/usr/bin/env bash
# Runs INSIDE the container. Started by handoff/run.sh; do not call it on the host.
set -euo pipefail
R=/groups/xzhang33/tguo2taicheng
E=$R/VerifierScientist/experiments/guidance_rl_2026-11-02
N=$E/frozen28_2026-11-11
REPO=${REPO:-/workspace/repo}
PY=$R/envs/uv-python/cpython-3.11.17-linux-x86_64-gnu/bin/python3.11
step=${1:-help}

[ -d "$REPO/tree" ] || { echo "ERROR: repository not mounted at $REPO"; exit 2; }
[ -x "$PY" ] || { echo "ERROR: image content missing ($PY)"; exit 2; }
[ -d "$N/out" ] && [ -w "$N/out" ] || { echo "ERROR: output directory is not mounted writable at $N/out"; exit 2; }

# 1. Lay the repository content over the fixed paths the code expects.
# Done once per container: a second entry into a running container (docker exec ... status) must not rewrite files in use.
if [ ! -f "$R/.repo_tree_applied" ]; then
  cp -a "$REPO/tree/." "$R/"
  while IFS= read -r d; do [ -n "$d" ] && mkdir -p "$R/$d"; done < "$REPO/handoff/empty_dirs.txt"
  touch "$R/.repo_tree_applied"
fi
mkdir -p "$E/audit" "$N/out"
# A new, empty API ledger for this run (the budget pre-check needs the file to exist).
[ -f "$N/out/api_budget.json" ] || echo '{"spent": 0.0, "calls": 0, "reserved": {}}' > "$N/out/api_budget.json"

# 2. Environment used by every step (same variables as the original launcher).
export GUIDANCE_PROJECT_ROOT=$R HARNESSAUDIT_ROOT=$R/HarnessAudit-main
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$N/code_v4:$R/envs/verl08/lib/python3.11/site-packages
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$N/code_v4"
CHK=$REPO/handoff/checks

run_training() {  # $1 = config, $2 = approval
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
  [ -f "$N/out/openai.env" ] || { echo "ERROR: put the API key file at <OUT>/openai.env first (see handoff/README.md)"; exit 2; }
  exec "$PY" -B -u -m guidance_rl.cli run --config "$1" --approval "$2"
}

case "$step" in
  gpucheck)    exec "$PY" -B "$CHK/gpu_check.py" ;;
  envcheck)    exec "$PY" -B "$CHK/env_check.py" ;;
  verify)      exec "$PY" -B "$CHK/verify_package.py" "$N/training_config.json" "$N/training_approval.json" "$N/smoke_config.json" "$N/smoke_approval.json" ;;
  netcheck)    exec "$PY" -B "$CHK/net_check.py" "$N/training_config.json" ;;
  workercheck) export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}; exec "$PY" -B "$CHK/worker_check.py" "$N/training_config.json" ;;
  smoke)       run_training "$N/smoke_config.json" "$N/smoke_approval.json" ;;
  train)       run_training "$N/training_config.json" "$N/training_approval.json" ;;
  status)      exec "$PY" -B "$CHK/status.py" "$N/out" ;;
  pack)        exec "$PY" -B "$CHK/pack_results.py" "$N/out" ;;
  shell)       exec bash ;;
  *) echo "steps: gpucheck | envcheck | verify | netcheck | workercheck | smoke | train | status | pack | shell"; exit 1 ;;
esac
