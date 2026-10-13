#!/usr/bin/env bash
set -euo pipefail

repo=/mnt/c/repo/rwkv-qualcomm-g1j-work
weights=/mnt/c/Users/halow/Desktop/RWKV-Weights/g1j/20260831-ctx16384-quantized
model="${1:?model key required}"
socs="${2:?comma-separated SoC list required}"
workers="${3:?context worker count required}"

source "$repo/g1j_local/qnn_env.sh"
export CUDA_VISIBLE_DEVICES=""
export TMPDIR="$weights/_work/qnn/tmp"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
mkdir -p "$TMPDIR"
cd "$repo"

nice -n 10 ionice -c 2 -n 7 python -u parallel_convert_pipeline.py \
  --filter_model "$model" \
  --filter_soc "$socs" \
  --skip_convert \
  --context_workers "$workers" \
  --skip_pack \
  --no_cleanup
