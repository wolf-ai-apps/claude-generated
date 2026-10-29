"""Lab 11a: choose AWS GPU instances for a model, from first principles.

    python3 pick_instance.py --params 8 --context 8192 --concurrency 32
    python3 pick_instance.py --params 70 --dtype fp8 --context 8192 --concurrency 64

For every instance shape it checks: do the weights fit, how many concurrent
sequences fit in the remaining memory, and is tensor parallelism needed.
Prices change and differ by region: pass your own with --prices-file
(JSON: {"g6.2xlarge": 0.98, ...}) from the AWS pricing page. Without prices,
the tool ranks by fit only.
"""

import argparse
import json
from dataclasses import dataclass


@dataclass
class Shape:
    name: str
    gpu: str
    gpus: int
    gpu_mem_gb: float
    bw_gbs: float      # per GPU
    nvlink: bool
    fp8: bool


# EC2 names; SageMaker uses the same hardware with an "ml." prefix (ml.g6.2xlarge, ...).
SHAPES = [
    Shape("g5.2xlarge", "A10G", 1, 24, 600, False, False),
    Shape("g5.12xlarge", "A10G", 4, 24, 600, False, False),
    Shape("g5.48xlarge", "A10G", 8, 24, 600, False, False),
    Shape("g6.2xlarge", "L4", 1, 24, 300, False, True),
    Shape("g6.12xlarge", "L4", 4, 24, 300, False, True),
    Shape("g6.48xlarge", "L4", 8, 24, 300, False, True),
    Shape("g6e.2xlarge", "L40S", 1, 48, 864, False, True),
    Shape("g6e.12xlarge", "L40S", 4, 48, 864, False, True),
    Shape("g6e.48xlarge", "L40S", 8, 48, 864, False, True),
    Shape("p4d.24xlarge", "A100-40GB", 8, 40, 1555, True, False),
    Shape("p4de.24xlarge", "A100-80GB", 8, 80, 2039, True, False),
    Shape("p5.48xlarge", "H100", 8, 80, 3350, True, True),
    Shape("p5e.48xlarge", "H200", 8, 141, 4800, True, True),
]

BYTES = {"bf16": 2, "fp16": 2, "fp8": 1, "int8": 1, "int4": 0.5}


def kv_bytes_per_token(params_b: float) -> float:
    """Rough GQA KV size per token by model scale (Llama 3 family values)."""
    if params_b <= 10:
        return 2 * 32 * 8 * 128 * 2      # 8B: 128 KiB
    if params_b <= 80:
        return 2 * 80 * 8 * 128 * 2      # 70B: 320 KiB
    return 2 * 126 * 8 * 128 * 2         # 405B: 504 KiB


def evaluate(shape: Shape, params_b, dtype, kv_dtype, context, mem_util=0.9, overhead_gb=2.0):
    """Return (tp, max_seqs, single-stream decode tok/s) for the smallest TP degree that fits, or None."""
    if kv_dtype == "fp8" and not shape.fp8:
        return None                      # FP8 KV cache needs Ada/Hopper (L4, L40S, H100, H200) or newer
    weights = params_b * 1e9 * BYTES[dtype] / 1e9
    kv_seq = kv_bytes_per_token(params_b) * BYTES[kv_dtype] / 2 * context / 1e9
    for tp in (1, 2, 4, 8):
        if tp > shape.gpus:
            break
        budget = shape.gpu_mem_gb * mem_util * tp - weights - overhead_gb * tp
        if budget > kv_seq:
            # PCIe-only multi-GPU tensor parallelism pays an all-reduce tax every layer
            eff = 1.0 if tp == 1 or shape.nvlink else 0.7
            tok_s = shape.bw_gbs * tp * 0.75 * eff / weights
            return tp, int(budget // kv_seq), tok_s
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--params", type=float, required=True, help="billions of parameters")
    p.add_argument("--dtype", default="bf16", choices=BYTES)
    p.add_argument("--kv-dtype", default="bf16", choices=["bf16", "fp8"])
    p.add_argument("--context", type=int, default=8192, help="tokens held per sequence")
    p.add_argument("--concurrency", type=int, default=32, help="concurrent sequences you need per replica")
    p.add_argument("--prices-file", help="JSON map instance -> $/hour")
    a = p.parse_args()
    prices = json.load(open(a.prices_file)) if a.prices_file else {}

    print(f"{a.params:g}B params, weights {a.dtype}, KV {a.kv_dtype}, {a.context} tokens/seq, "
          f"need {a.concurrency} concurrent seqs\n")
    print(f"{'instance':<15}{'GPUs':<11}{'TP':>3}{'max seqs':>10}{'tok/s@b1':>11}{'replicas/inst':>15}  verdict")
    for s in SHAPES:
        r = evaluate(s, a.params, a.dtype, a.kv_dtype, a.context)
        if r is None:
            print(f"{s.name:<15}{s.gpus}x{s.gpu:<9}{'-':>3}{'-':>10}{'-':>11}{'-':>15}  does not fit")
            continue
        tp, seqs, tok_s = r
        replicas = s.gpus // tp
        verdict = "OK" if seqs >= a.concurrency else f"only {seqs} seqs"
        if a.dtype == "fp8" and not s.fp8:
            verdict += " (no FP8 tensor cores: weight-only FP8 kernels, slower prefill)"
        price = prices.get(s.name)
        if price and seqs:
            verdict += f"  ${price / (replicas * min(seqs, a.concurrency)):.3f}/seq-hour"
        print(f"{s.name:<15}{s.gpus}x{s.gpu:<9}{tp:>3}{seqs:>10}{tok_s:>11.0f}{replicas:>15}  {verdict}")
    print("\nRules of thumb: prefer the smallest TP that fits (fewer all-reduces); prefer NVLink for TP>=4;"
          "\nif only a big multi-GPU box fits, run several replicas on it (replicas/inst column).")


if __name__ == "__main__":
    main()
