"""Lab 3b: profile one generation and see where the time goes.

    pip install torch transformers accelerate
    python3 profile_generate.py --model Qwen/Qwen2.5-0.5B-Instruct

Outputs a table of the top CUDA kernels and a Chrome trace (trace.json) you
can open at https://ui.perfetto.dev . Things to look for:
  - prefill: a few large matmul (GEMM) kernels
  - decode: many tiny kernels per token, with gaps between them (launch overhead)
  - how much time is matmul vs attention vs "everything else"
"""

import argparse

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--new-tokens", type=int, default=32)
    a = p.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).cuda().eval()
    inputs = tok("Explain the KV cache in one paragraph.", return_tensors="pt").to("cuda")

    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=4)  # warm-up
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            model.generate(**inputs, max_new_tokens=a.new_tokens, do_sample=False)
            torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    prof.export_chrome_trace("trace.json")
    print("wrote trace.json -> open in https://ui.perfetto.dev")


if __name__ == "__main__":
    main()
