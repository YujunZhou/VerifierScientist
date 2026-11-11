#!/usr/bin/env bash
# Host-side launcher. Usage:  OUT=/path/with/space  ./handoff/run.sh <step>
# Steps, in order: gpucheck envcheck verify netcheck workercheck smoke train   (then: status, pack)
set -euo pipefail
IMAGE=${IMAGE:-docker.io/yujun0017/verifierscientist-train:2026-11-11}
REPO=$(cd "$(dirname "$0")/.." && pwd)
: "${OUT:?Set OUT to an output directory on a disk with at least 350 GB free}"
GPUS=${GPUS:-0,1,2,3}            # host indices of the four GPUs to use
SHM=${SHM:-64g}
step=${1:?step name required (see handoff/README.md)}
mkdir -p "$OUT"
N=/groups/xzhang33/tguo2taicheng/VerifierScientist/experiments/guidance_rl_2026-11-02/frozen28_2026-11-11
proxy=()
for v in HTTPS_PROXY HTTP_PROXY NO_PROXY https_proxy http_proxy no_proxy; do [ -n "${!v:-}" ] && proxy+=(-e "$v=${!v}"); done
tty=(); [ -t 0 ] && tty=(-it)
name=vs-frozen28-$step
exec docker run --rm "${tty[@]}" --name "$name" \
  --gpus "\"device=$GPUS\"" --shm-size="$SHM" --ulimit nofile=1048576:1048576 \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 "${proxy[@]}" ${DOCKER_EXTRA:-} \
  -v "$REPO":/workspace/repo:ro -v "$OUT":"$N/out" \
  "$IMAGE" bash /workspace/repo/handoff/container_entry.sh "$step"
