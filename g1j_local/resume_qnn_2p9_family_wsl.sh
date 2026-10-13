#!/usr/bin/env bash
set -euo pipefail

REPO=/mnt/c/repo/rwkv-qualcomm-g1j-work
WEIGHTS=/mnt/c/Users/halow/Desktop/RWKV-Weights/g1j/20260831-ctx16384-quantized
FAMILY="${1:?custom or nocustom required}"
BASE=rwkv7-g1j-2.9b-20260831-ctx16384-a16w4-lpbq-b32
MODEL="$WEIGHTS/source/rwkv7-g1j-2.9b-20260831-ctx16384.pth"
ENCODING=quant_export/g1j-2b9-lpbq-b32-w4/rwkv7-g1j-2.9b-20260831-ctx16384.encodings
PREFILL_ENCODING=quant_export/g1j-2b9-lpbq-b32-w4/rwkv7-g1j-2.9b-20260831-ctx16384_prefill.encodings

case "$FAMILY" in
  custom)
    OUTPUT_NAME="$BASE"
    FAMILY_ARGS=(--wkv_customop)
    ;;
  nocustom)
    OUTPUT_NAME="$BASE-nocustomop"
    FAMILY_ARGS=()
    ;;
  *)
    printf 'family must be custom or nocustom\n' >&2
    exit 2
    ;;
esac

source "$REPO/g1j_local/qnn_env.sh"
export CUDA_VISIBLE_DEVICES=""
export TMPDIR=/home/halo/g1j-qnn/tmp
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
mkdir -p "$TMPDIR"
cd "$REPO"

run_variant() {
  local label="$1"
  local dir_suffix="$2"
  local file_suffix="$3"
  local encoding="$4"
  shift 4
  local variant_args=("$@")
  local complete=0
  local chunk

  for chunk in 1 2 3 4; do
    local dlc="onnx/${OUTPUT_NAME}_chunk${chunk}of4${dir_suffix}/${OUTPUT_NAME}${file_suffix}_chunk${chunk}of4.dlc"
    if [[ -s "$dlc" ]]; then
      complete=$((complete + 1))
    fi
  done

  if [[ "$complete" -eq 4 ]]; then
    printf '[%s/%s] already complete; skipping model load\n' "$FAMILY" "$label"
    return
  fi

  printf '[%s/%s] resuming; existing chunks=%s/4\n' "$FAMILY" "$label" "$complete"
  nice -n 10 ionice -c 2 -n 7 python -u convert_model_dlc.py \
    "$MODEL" \
    --chunks 4 \
    --quant_encodings "$encoding" \
    --heads_per_split 4 \
    --output_name "$OUTPUT_NAME" \
    --resume_existing_dlc \
    --prune_intermediates_after_chunk \
    --no_cleanup \
    "${FAMILY_ARGS[@]}" \
    "${variant_args[@]}"
}

run_variant decode '' '' "$ENCODING"
run_variant prefill _prefill _prefill "$PREFILL_ENCODING" --prefill_model
run_variant bsz2 _bsz2 _bsz2 "$ENCODING" --batch_size 2
run_variant bsz4 _bsz4 _bsz4 "$ENCODING" --batch_size 4
run_variant bsz6 _bsz6 _bsz6 "$ENCODING" --batch_size 6
run_variant bsz8 _bsz8 _bsz8 "$ENCODING" --batch_size 8
