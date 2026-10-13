#!/usr/bin/env bash
set -euo pipefail

umask 077
REPO=/mnt/c/repo/rwkv-qualcomm-g1j-work
WEIGHTS=/mnt/c/Users/halow/Desktop/RWKV-Weights/g1j/20260831-ctx16384-quantized
LOG_ROOT=/home/halo/g1j-qnn/logs
SCOPE="${1:-all}"

if [[ "$SCOPE" != "all" && "$SCOPE" != "1.5" && "$SCOPE" != "2.9" ]]; then
  printf 'usage: %s [all|1.5|2.9]\n' "$0" >&2
  exit 2
fi

source "$REPO/g1j_local/qnn_env.sh"
export CUDA_VISIBLE_DEVICES=""
export TMPDIR="$WEIGHTS/_work/qnn/tmp"
mkdir -p "$TMPDIR" "$LOG_ROOT"
cd "$REPO"

if [[ "$SCOPE" == "all" || "$SCOPE" == "1.5" ]]; then
  mkdir -p quant_export/g1j-1b5-w8
  nice -n 10 ionice -c 2 -n 7 python -u -c \
    'import torch.utils.cpp_extension, runpy; runpy.run_path("compute_quant_encodings_experimental.py", run_name="__main__")' \
    "$WEIGHTS/source/rwkv7-g1j-1.5b-20260831-ctx16384.pth" \
    --output_folder quant_export/g1j-1b5-w8 \
    --calib_num_batches 1 \
    --heads_per_split 4 \
    --use_cpu \
    2>&1 | tee "$LOG_ROOT/quantize-1.5b-w8.log"
fi

if [[ "$SCOPE" == "all" || "$SCOPE" == "2.9" ]]; then
  mkdir -p quant_export/g1j-2b9-lpbq-b32-w4
  nice -n 10 ionice -c 2 -n 7 python -u -c \
    'import torch.utils.cpp_extension, runpy; runpy.run_path("compute_quant_encodings_experimental.py", run_name="__main__")' \
    "$WEIGHTS/source/rwkv7-g1j-2.9b-20260831-ctx16384.pth" \
    --output_folder quant_export/g1j-2b9-lpbq-b32-w4 \
    --calib_num_batches 1 \
    --heads_per_split 4 \
    --blockwise_quant \
    --blockwise_block_size 32 \
    --low_memory_onnx_stage \
    --use_cpu \
    2>&1 | tee "$LOG_ROOT/quantize-2.9b-w4-stage1.log"

  nice -n 10 ionice -c 2 -n 7 python -u -c \
    'import torch.utils.cpp_extension, runpy; runpy.run_path("compute_quant_encodings_experimental.py", run_name="__main__")' \
    "$WEIGHTS/source/rwkv7-g1j-2.9b-20260831-ctx16384.pth" \
    --output_folder quant_export/g1j-2b9-lpbq-b32-w4 \
    --calib_num_batches 1 \
    --heads_per_split 4 \
    --blockwise_quant \
    --blockwise_block_size 32 \
    --skip_compute_encodings \
    --load_encodings quant_export/g1j-2b9-lpbq-b32-w4/quant_encodings_checkpoint_calib.json \
    --reuse_onnx_models \
    --use_cpu \
    2>&1 | tee "$LOG_ROOT/quantize-2.9b-w4-stage2.log"
fi

sha256sum \
  quant_export/g1j-1b5-w8/rwkv7-g1j-1.5b-20260831-ctx16384.encodings \
  quant_export/g1j-1b5-w8/rwkv7-g1j-1.5b-20260831-ctx16384_prefill.encodings \
  quant_export/g1j-2b9-lpbq-b32-w4/rwkv7-g1j-2.9b-20260831-ctx16384.encodings \
  quant_export/g1j-2b9-lpbq-b32-w4/rwkv7-g1j-2.9b-20260831-ctx16384_prefill.encodings
