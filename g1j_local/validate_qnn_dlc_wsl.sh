#!/usr/bin/env bash
set -euo pipefail

source /mnt/c/repo/rwkv-qualcomm-g1j-work/g1j_local/qnn_env.sh

dlc_root=/mnt/c/Users/halow/Desktop/RWKV-Weights/g1j/20260831-ctx16384-quantized/_work/qnn/onnx
total=0
failed=0

while IFS= read -r -d '' dlc; do
  total=$((total + 1))
  if ! qairt-dlc-info -i "$dlc" >/dev/null 2>&1; then
    printf 'FAIL %s\n' "$dlc"
    failed=$((failed + 1))
  fi
done < <(find "$dlc_root" -type f -name '*.dlc' -print0)

printf 'DLC_INFO_TOTAL=%d DLC_INFO_FAILED=%d\n' "$total" "$failed"
test "$total" -eq 120
test "$failed" -eq 0
