"""Lab 13b: call the endpoint, non-streaming and streaming, and time it.

    python3 invoke.py --name academy-llama-8b
    python3 invoke.py --name academy-llama-8b --inference-component academy-llama-8b-ic   # lab 13d

The request body is an ordinary OpenAI chat-completions payload; vLLM's
/invocations route dispatches it to the chat handler.
"""

import argparse
import json
import time

import boto3


def body(stream: bool, max_tokens: int) -> bytes:
    return json.dumps({
        "model": "llm",
        "messages": [{"role": "user", "content": "In three bullet points: why do LLM servers batch requests?"}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": stream,
    }).encode()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="academy-llama-8b")
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--inference-component", default="")
    p.add_argument("--max-tokens", type=int, default=128)
    a = p.parse_args()
    rt = boto3.client("sagemaker-runtime", region_name=a.region)
    extra = {"InferenceComponentName": a.inference_component} if a.inference_component else {}

    t = time.perf_counter()
    r = rt.invoke_endpoint(EndpointName=a.name, ContentType="application/json", Body=body(False, a.max_tokens), **extra)
    out = json.loads(r["Body"].read())
    print(out["choices"][0]["message"]["content"])
    print(f"-- non-streaming: {time.perf_counter() - t:.2f}s total, usage {out.get('usage')}\n")

    t = time.perf_counter()
    first, n, buf = None, 0, b""
    r = rt.invoke_endpoint_with_response_stream(EndpointName=a.name, ContentType="application/json",
                                                Body=body(True, a.max_tokens), **extra)
    for event in r["Body"]:
        buf += event.get("PayloadPart", {}).get("Bytes", b"")
        # PayloadParts are arbitrary byte slices of the SSE stream: split on line boundaries ourselves.
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:") or line.endswith(b"[DONE]"):
                continue
            chunk = json.loads(line[5:])
            if chunk.get("choices") and chunk["choices"][0]["delta"].get("content"):
                first = first or time.perf_counter()
                n += 1
                print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
    total = time.perf_counter() - t
    print(f"\n-- streaming: TTFT {(first - t) * 1000:.0f} ms (includes SageMaker's routing hop), "
          f"{n} chunks, {total:.2f}s total")


if __name__ == "__main__":
    main()
