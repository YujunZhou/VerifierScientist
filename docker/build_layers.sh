#!/usr/bin/env bash
# Build the image layers as gzip tarballs with root ownership and the original absolute paths.
set -uo pipefail
W=$1; L=$W/layers; R=groups/xzhang33/tguo2taicheng; E=$R/VerifierScientist/experiments/guidance_rl_2026-11-02
T="tar --owner=0 --group=0 --numeric-owner --no-acls --no-xattrs --no-selinux -C /"
mk(){ name=$1; shift; echo "$(date +%T) start $name"; nice $T -cf - "$@" 2> $L/$name.err | nice pigz -3 > $L/$name.tar.gz; echo "$(date +%T) done $name rc=${PIPESTATUS[0]} size=$(du -h $L/$name.tar.gz | cut -f1) errs=$(wc -l < $L/$name.err)"; }
( cd $W/layer_sys && nice tar --owner=0 --group=0 --numeric-owner -cf - . | pigz -3 > $L/01_sys.tar.gz ); echo "done 01_sys $(du -h $L/01_sys.tar.gz | cut -f1)"
mk 02_python $R/envs/uv-python $R/envs/swebench_src $R/envs/swetools
mk 03_env_jit $R/envs/verl08/lib/python3.11/site-packages/flashinfer_jit_cache
mk 04_env_main --exclude=$R/envs/verl08/lib/python3.11/site-packages/flashinfer_jit_cache $R/envs/verl08
mk 05_tasks $E/fixtures $E/fixture_envs $E/audit/fixture_runtime $E/audit/fixture_runtime_local
mk 06_model $R/workAmazon/hf/hub/models--Qwen--Qwen3.5-4B
echo ALL LAYERS DONE
