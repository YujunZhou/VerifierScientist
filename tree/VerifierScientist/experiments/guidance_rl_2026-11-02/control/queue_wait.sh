#!/usr/bin/env bash
set -euo pipefail
ROOT=/groups/xzhang33/tguo2taicheng
PROJECT="$ROOT/VerifierScientist"
RUN="$PROJECT/experiments/guidance_rl_2026-11-02"
PY="$ROOT/envs/uv-python/cpython-3.11.17-linux-x86_64-gnu/bin/python3.11"
cd "$PROJECT"
printf 'GPU allocation acquired: job=%s host=%s at=%s\n' "${JOB_ID:-unknown}" "$(hostname)" "$(date -Is)"
printf '%s\n' "${JOB_ID:-unknown}" > "$RUN/control/allocated_job_id"
printf '%s\n' "$(hostname)" > "$RUN/control/allocated_host"
while [[ ! -f "$RUN/control/review_approved.json" || ! -f "$RUN/control/launch.json" || ! -f "$RUN/initial_optimization/artifact.json" ]]; do
    printf 'Waiting for reviewed code, launch config and optimized guidance: %s\n' "$(date -Is)"
    sleep 60
done
export PYTHONPATH="$ROOT/envs/verl08/lib/python3.11/site-packages:$ROOT/harnessaudit_verl:$ROOT/HarnessAudit-main"
export PYTHONDONTWRITEBYTECODE=1
exec "$PY" -B -m guidance_rl.cli run --config "$RUN/control/launch.json" --approval "$RUN/control/review_approved.json"
