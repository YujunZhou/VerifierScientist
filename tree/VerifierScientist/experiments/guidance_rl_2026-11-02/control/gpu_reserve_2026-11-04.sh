#!/usr/bin/env bash
# Reservation only. No task execution, training, model calls or old approvals.
set -euo pipefail
RUN=/groups/xzhang33/tguo2taicheng/VerifierScientist/experiments/guidance_rl_2026-11-02
STATE="$RUN/gpu_requeue_2026-11-04"
mkdir -p "$STATE"
printf '%s\n' "${JOB_ID:?Scheduler JOB_ID required}" > "$STATE/allocated_job_id"
hostname > "$STATE/allocated_host"
date -u -Is > "$STATE/allocated_at_utc"
printf '%s\n' "$JOB_ID" > "$RUN/control/allocated_job_id"
hostname > "$RUN/control/allocated_host"
printf 'Reservation ready: job=%s host=%s. No training or task runner is enabled.\n' "$JOB_ID" "$(hostname)"
trap 'exit 0' TERM INT
while true; do
    sleep 60 &
    wait "$!"
done
