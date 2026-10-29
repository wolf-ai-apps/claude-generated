"""Lab 8c: offline (batch) inference with vLLM's Python API.

    pip install vllm
    python3 offline_batch.py --model Qwen/Qwen2.5-0.5B-Instruct --n 512

No HTTP server: the engine gets all prompts at once and schedules them for
maximum throughput. This is the mode for evals, labeling and synthetic data.
Compare tokens/s here with what the online server achieves at a latency SLO (level 9).
"""

import argparse
import time

from vllm import LLM, SamplingParams


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--max-tokens", type=int, default=128)
    a = p.parse_args()

    llm = LLM(model=a.model, max_model_len=4096, gpu_memory_utilization=0.9, enable_prefix_caching=True)
    params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=a.max_tokens)
    topics = ["KV cache", "PagedAttention", "continuous batching", "quantization", "speculative decoding",
              "tensor parallelism", "prefix caching", "FlashAttention"]
    conversations = [[{"role": "system", "content": "You are a GPU inference expert. Answer in two sentences."},
                      {"role": "user", "content": f"Question {i}: explain {topics[i % len(topics)]}."}]
                     for i in range(a.n)]

    t = time.perf_counter()
    outputs = llm.chat(conversations, params)
    dt = time.perf_counter() - t

    gen = sum(len(o.outputs[0].token_ids) for o in outputs)
    prompt = sum(len(o.prompt_token_ids) for o in outputs)
    print(outputs[0].outputs[0].text.strip()[:300])
    print(f"\n{a.n} requests in {dt:.1f}s: {gen / dt:,.0f} output tok/s, {(gen + prompt) / dt:,.0f} total tok/s")


if __name__ == "__main__":
    main()
