#!/usr/bin/env bash
# Lab 8a: serve a model with vLLM's OpenAI-compatible server in Docker.
#
#   MODEL=Qwen/Qwen2.5-0.5B-Instruct ./serve_vllm.sh           # ungated, fits any GPU
#   MODEL=meta-llama/Llama-3.1-8B-Instruct HF_TOKEN=hf_... ./serve_vllm.sh
#
# Pin VLLM_TAG to a specific release; never run :latest in anything you benchmark.
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
VLLM_TAG=${VLLM_TAG:-v0.11.0}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
GPU_UTIL=${GPU_UTIL:-0.90}

docker run --rm -it --gpus all \
  --ipc=host \
  -p 8000:8000 \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  "vllm/vllm-openai:${VLLM_TAG}" \
  --model "${MODEL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --enable-prefix-caching \
  "$@"

# What to read in the startup log:
#   "Loading weights took ... GB"            -> matches params x bytes? (level 6 calculator)
#   "Available KV cache memory: ... GiB"      -> what is left after weights + activations
#   "GPU KV cache size: N tokens"             -> N / max_model_len = max full-length sequences
#   "Maximum concurrency for 8192 tokens per request: Xx"
#   "Capturing CUDA graphs"                   -> decode launch-overhead optimisation (level 7)
# --ipc=host (or a large /dev/shm) is required for multi-GPU tensor parallelism.
