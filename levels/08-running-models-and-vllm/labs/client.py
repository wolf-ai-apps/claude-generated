"""Lab 8b: talk to vLLM with the standard OpenAI SDK.

    pip install openai
    python3 client.py --base-url http://localhost:8000/v1

Shows: listing models, streaming chat, structured JSON output, and the
vLLM-specific extra parameters. Swap base-url for the academy gateway
(http://localhost:9000/v1) or a SageMaker/EKS endpoint later: same code.
"""

import argparse
import json
import time

from openai import OpenAI


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--api-key", default="not-needed")
    a = p.parse_args()
    client = OpenAI(base_url=a.base_url, api_key=a.api_key)

    model = client.models.list().data[0].id
    print("model:", model)

    print("\n1) Streaming chat")
    t0 = time.perf_counter()
    first = None
    n = 0
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are a concise tutor."},
                  {"role": "user", "content": "Why is LLM decode memory-bandwidth bound?"}],
        max_tokens=120, temperature=0.2, stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            first = first or time.perf_counter()
            n += 1
            print(chunk.choices[0].delta.content, end="", flush=True)
    dt = time.perf_counter() - t0
    print(f"\n   TTFT {(first - t0) * 1000:.0f} ms, {n} chunks in {dt:.2f}s")

    print("\n2) Structured output (JSON schema enforced by constrained decoding, lab 5c)")
    schema = {"type": "object",
              "properties": {"gpu": {"type": "string"}, "memory_gb": {"type": "integer"},
                             "good_for_8b_bf16": {"type": "boolean"}},
              "required": ["gpu", "memory_gb", "good_for_8b_bf16"]}
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Describe the NVIDIA L4 GPU as JSON."}],
        response_format={"type": "json_schema", "json_schema": {"name": "gpu", "schema": schema}},
        max_tokens=100, temperature=0,
    )
    print("  ", json.loads(r.choices[0].message.content))
    print("   usage:", r.usage)

    print("\n3) vLLM extras via extra_body (not part of the OpenAI spec)")
    r = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": "Count: 1, 2, 3,"}],
        max_tokens=16, temperature=0.8,
        extra_body={"top_k": 20, "min_p": 0.05, "repetition_penalty": 1.1, "ignore_eos": True},
    )
    print("  ", r.choices[0].message.content, "| finish_reason:", r.choices[0].finish_reason)


if __name__ == "__main__":
    main()
