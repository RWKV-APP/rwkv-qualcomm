#!/usr/bin/env bash
set -euo pipefail

runner=/mnt/c/repo/rwkv-qualcomm-g1j-work/g1j_local/run_qnn_context_wsl.sh
model="${1:?model key required}"
socs="${2:?comma-separated SoC list required}"
workers="${3:?context worker count required}"

max_generators=0
max_related_rss_kib=0
min_mem_available_kib=9223372036854775807
max_swap_used_kib=0
min_c_available_bytes=9223372036854775807

bash "$runner" "$model" "$socs" "$workers" &
runner_pid=$!

while kill -0 "$runner_pid" 2>/dev/null; do
  generators=$(ps -eo args= | awk '/qnn-context-binary-generator/ && ! /awk/ {count++} END {print count+0}')
  related_rss=$(ps -eo rss=,args= | awk '/qnn-context-binary-generator|make_context_cache_binary_dlc.py|parallel_convert_pipeline.py/ && ! /awk/ {sum+=$1} END {print sum+0}')
  mem_available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
  swap_total=$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo)
  swap_free=$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)
  swap_used=$((swap_total - swap_free))
  c_available=$(df -B1 --output=avail /mnt/c | tail -n 1 | tr -d ' ')

  (( generators > max_generators )) && max_generators=$generators
  (( related_rss > max_related_rss_kib )) && max_related_rss_kib=$related_rss
  (( mem_available < min_mem_available_kib )) && min_mem_available_kib=$mem_available
  (( swap_used > max_swap_used_kib )) && max_swap_used_kib=$swap_used
  (( c_available < min_c_available_bytes )) && min_c_available_bytes=$c_available
  sleep 1
done

set +e
wait "$runner_pid"
status=$?
set -e

printf 'CONTEXT_MONITOR model=%s socs=%s requested_workers=%s max_generators=%d max_related_rss_kib=%d min_mem_available_kib=%d max_swap_used_kib=%d min_c_available_bytes=%d exit=%d\n' \
  "$model" "$socs" "$workers" "$max_generators" "$max_related_rss_kib" \
  "$min_mem_available_kib" "$max_swap_used_kib" "$min_c_available_bytes" "$status"
exit "$status"
