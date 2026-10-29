#!/usr/bin/env bash
# Lab 14b: chaos drill on the laptop fleet. Kill a replica mid-load and watch what the
# gateway, the load generator and the metrics tell you. No GPU needed.
#
#   ./chaos.sh            # uses the Go tools from platform/
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=$(mktemp -d)
(cd "$ROOT/platform" && go build -o "$BIN/" ./cmd/...)
trap 'kill $(jobs -p) 2>/dev/null || true; rm -rf "$BIN"' EXIT

for i in 1 2 3; do "$BIN/mockllm" -addr ":810$i" -name "replica-$i" -max-num-seqs 16 >/dev/null 2>&1 & done
"$BIN/gateway" -addr :9100 -policy least-loaded -health-interval 1s \
  -backends http://localhost:8101,http://localhost:8102,http://localhost:8103 2>"$BIN/gateway.log" &
sleep 1

echo "== Baseline: 3 healthy replicas"
"$BIN/loadgen" -url http://localhost:9100 -rate 8 -n 120 -output-min 32 -output-max 96 -slo-ttft 500ms -slo-tpot 50ms | grep -E "Requests|SLO|Backends"

echo; echo "== Drill: kill replica-2 five seconds into the run"
( sleep 5; pkill -f -- "-name replica-2" && echo "   [chaos] replica-2 killed at t=5s" ) &
"$BIN/loadgen" -url http://localhost:9100 -rate 8 -n 120 -output-min 32 -output-max 96 -slo-ttft 500ms -slo-tpot 50ms -seed 7 2>&1 | grep -E "Requests|SLO|Backends|first error"

echo; echo "== Gateway log (health-check transitions)"
grep -i healthy "$BIN/gateway.log" || true
cat <<'EOF'

Questions to answer in your runbook:
  1. How many requests failed, and were they the ones in flight on replica-2 when it died?
     (Streams that already started cannot be retried transparently: tokens were sent.)
  2. How long until the gateway stopped routing to the dead replica? What sets that bound?
  3. Did the surviving replicas' TTFT rise? Is there enough headroom to lose one replica
     at peak? (That is your N+1 capacity requirement.)
  4. What alert would have fired in production, and after how long?
EOF
