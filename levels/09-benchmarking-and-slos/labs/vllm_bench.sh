#!/usr/bin/env bash
# Lab 9b: cross-check the academy load generator against vLLM's own benchmark.
# Run from a machine with vLLM installed (pip install vllm) while serve_vllm.sh is running.
set -euo pipefail
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
RATE=${RATE:-8}

vllm bench serve \
  --backend openai-chat \
  --base-url http://localhost:8000 \
  --endpoint /v1/chat/completions \
  --model "$MODEL" \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 128 \
  --num-prompts 300 \
  --request-rate "$RATE" \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,95,99

# Then run the same shape with the academy tool and compare p50/p95 TTFT and TPOT:
#   cd platform && go run ./cmd/loadgen -url http://localhost:8000 -model "$MODEL" -rate $RATE -n 300 \
#     -prompt-min 512 -prompt-max 512 -output-min 128 -output-max 128
# Flags change between vLLM releases: check `vllm bench serve --help` for your version.
