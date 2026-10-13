#!/usr/bin/env bash
set -euo pipefail

REPO=/mnt/c/repo/rwkv-qualcomm-g1j-work
WEIGHTS=/mnt/c/Users/halow/Desktop/RWKV-Weights/g1j/20260831-ctx16384-quantized

source "$REPO/g1j_local/qnn_env.sh"
export CUDA_VISIBLE_DEVICES=""
export TMPDIR="$WEIGHTS/_work/qnn/tmp"
cd "$REPO"

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
  2>&1 | tee /home/halo/g1j-qnn/logs/quantize-2.9b-w4-stage2-retry1.log
