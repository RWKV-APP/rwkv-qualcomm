#!/usr/bin/env bash
set -euo pipefail

REPO=/mnt/c/repo/rwkv-qualcomm-g1j-work
WEIGHTS=/mnt/c/Users/halow/Desktop/RWKV-Weights/g1j/20260831-ctx16384-quantized
MODEL="${1:?model key required}"
GRAPH_FAMILY="${2:?custom or nocustom required}"
WORKERS="${3:-1}"

case "$GRAPH_FAMILY" in
  custom) SOC=6490 ;;
  nocustom) SOC=xelite ;;
  *) printf 'graph family must be custom or nocustom\n' >&2; exit 2 ;;
esac

source "$REPO/g1j_local/qnn_env.sh"
export CUDA_VISIBLE_DEVICES=""
export TMPDIR="$WEIGHTS/_work/qnn/tmp"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
cd "$REPO"

nice -n 10 ionice -c 2 -n 7 python -u parallel_convert_pipeline.py \
  --filter_model "$MODEL" \
  --filter_soc "$SOC" \
  --convert_workers "$WORKERS" \
  --skip_binary_gen \
  --skip_pack \
  --no_cleanup
