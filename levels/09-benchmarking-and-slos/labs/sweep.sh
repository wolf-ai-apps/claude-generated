#!/usr/bin/env bash
# Lab 9a: find the maximum request rate that still meets the SLO (max goodput).
#
#   ./sweep.sh http://localhost:8000 mock-llama-3.1-8b        # against mockllm or vLLM
#
# For each arrival rate, run an open-loop test and record throughput and SLO attainment.
# Each rate uses a different seed: reusing prompts across runs would hit the prefix cache
# and flatter every run after the first.
# The answer to "how many users can one replica serve?" is the highest rate where
# attainment stays >= your target (commonly 95% or 99%).
set -euo pipefail
URL=${1:-http://localhost:8000}
MODEL=${2:-mock-llama-3.1-8b}
RATES=${RATES:-"1 2 4 8 12 16 24 32"}
N=${N:-200}
SLO_TTFT=${SLO_TTFT:-500ms}
SLO_TPOT=${SLO_TPOT:-50ms}
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
OUT=$(mkdir -p "${OUT:-sweep-results}" && cd "${OUT:-sweep-results}" && pwd)

printf '%-8s %-10s %-12s %-12s %-12s %s\n' rate req/s out_tok/s ttft_p95 tpot_p95 slo_met
for r in $RATES; do
  (cd "$ROOT/platform" && go run ./cmd/loadgen -url "$URL" -model "$MODEL" -rate "$r" -n "$N" \
      -prompt-min 256 -prompt-max 1024 -output-min 64 -output-max 256 \
      -seed "$r" -slo-ttft "$SLO_TTFT" -slo-tpot "$SLO_TPOT" -out "$OUT/rate-$r.json" >/dev/null)
  python3 - "$OUT/rate-$r.json" "$r" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))["summary"]
pct = 100 * s["SLOMet"] / max(s["Requests"], 1)
print(f"{sys.argv[2]:<8} {s['RequestRate']:<10.2f} {s['OutputTokRate']:<12.0f} "
      f"{s['TTFT']['P95']:<12.0f} {s['TPOT']['P95']:<12.1f} {pct:.1f}%")
EOF
done
echo "raw results in $OUT/ (one JSON per rate, with every request's timings)"
