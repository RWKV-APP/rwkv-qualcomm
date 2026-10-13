#!/usr/bin/env bash
set -euo pipefail

g1j_onnx_root=/mnt/c/Users/halow/Desktop/RWKV-Weights/g1j/20260831-ctx16384-quantized/_work/qnn/onnx
g1j_expected=/mnt/c/Users/halow/Desktop/RWKV-Weights/g1j/20260831-ctx16384-quantized/_work/qnn/onnx
g1j_resolved=$(realpath -- "$g1j_onnx_root")

if [[ "$g1j_resolved" != "$g1j_expected" ]]; then
  printf 'refusing unexpected target: %s\n' "$g1j_resolved" >&2
  exit 2
fi

g1j_count=$(find "$g1j_resolved" -xdev -type f ! -name '*.dlc' -printf x | wc -c)
g1j_bytes=$(find "$g1j_resolved" -xdev -type f ! -name '*.dlc' -printf '%s\n' |
  awk '{size += $1} END {print size + 0}')
find "$g1j_resolved" -xdev -type f ! -name '*.dlc' -delete

printf 'resolved=%s removed_files=%s removed_bytes=%s\n' \
  "$g1j_resolved" "$g1j_count" "$g1j_bytes"
df -h /mnt/c | tail -1
