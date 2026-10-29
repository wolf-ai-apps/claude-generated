"""Lab 0: feel TTFT and inter-token latency.

Streams one chat completion from any OpenAI-compatible server and prints a
timestamp for every token. Works against:
  - the academy mock server:   go run ./platform/cmd/mockllm         (no GPU)
  - Ollama:                    http://localhost:11434
  - vLLM:                      vllm serve <model>                     (level 8)

Standard library only:
  python3 stream_timing.py --url http://localhost:8000 --model mock-llama-3.1-8b
"""

import argparse
import json
import time
import urllib.request


def stream(url: str, model: str, prompt: str, max_tokens: int, api_key: str | None):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    start = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if not chunk.get("choices"):
                continue
            text = chunk["choices"][0].get("delta", {}).get("content") or ""
            if text:
                yield time.perf_counter() - start, text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--model", default="mock-llama-3.1-8b")
    p.add_argument("--prompt", default="Explain what a GPU is in three sentences.")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--api-key")
    a = p.parse_args()

    times = []
    for t, text in stream(a.url, a.model, a.prompt, a.max_tokens, a.api_key):
        gap = (t - times[-1]) * 1000 if times else t * 1000
        label = "TTFT" if not times else "ITL "
        print(f"{t*1000:8.1f} ms  {label} {gap:7.1f} ms  {text!r}")
        times.append(t)

    if len(times) >= 2:
        itls = [(b - a_) * 1000 for a_, b in zip(times, times[1:])]
        itls.sort()
        print(f"\nTTFT {times[0]*1000:.1f} ms | tokens {len(times)} | "
              f"ITL p50 {itls[len(itls)//2]:.1f} ms | "
              f"decode speed {(len(times)-1)/(times[-1]-times[0]):.1f} tok/s | "
              f"end-to-end {times[-1]*1000:.0f} ms")


if __name__ == "__main__":
    main()
