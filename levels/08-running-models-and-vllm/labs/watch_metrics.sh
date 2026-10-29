#!/usr/bin/env bash
# Lab 8d: watch the engine's scheduler state live while you send load.
#   ./watch_metrics.sh http://localhost:8000
# Works against vLLM and against the academy mockllm (same metric names).
URL=${1:-http://localhost:8000}
while true; do
  m=$(curl -s "${URL}/metrics")
  get() { echo "$m" | grep -E "^$1(\{| )" | awk '{s+=$NF} END {printf "%.3g", s}'; }
  hits=$(get 'vllm:prefix_cache_hits_total'); queries=$(get 'vllm:prefix_cache_queries_total')
  printf '%s running=%-4s waiting=%-4s kv_cache=%-6s prefix_hit_rate=%s\n' \
    "$(date +%T)" "$(get 'vllm:num_requests_running')" "$(get 'vllm:num_requests_waiting')" \
    "$(get 'vllm:kv_cache_usage_perc')" \
    "$(awk -v h="$hits" -v q="$queries" 'BEGIN { if (q > 0) printf "%.0f%%", 100*h/q; else print "n/a" }')"
  sleep 1
done
