"""Lab 6a: place prefill and decode on a GPU's roofline.

    python3 roofline.py --gpu h100
    python3 roofline.py --gpu l4 --batch 1 8 64 256 --prompt 2048

Roofline: achievable FLOP/s = min(peak FLOP/s, bandwidth x arithmetic intensity).
The "ridge point" (peak FLOP/s / bandwidth) is the intensity where a kernel
stops being memory-bound and becomes compute-bound.

For a transformer's weight matmuls in BF16 (2 bytes per weight):
  - each token does 2 FLOPs per weight (multiply + add)
  - a decode step with batch B reads each weight once and uses it B times
      -> intensity ~= 2B FLOPs / 2 bytes = B FLOPs/byte
  - a prefill of P prompt tokens uses each weight P times
      -> intensity ~= P FLOPs/byte
(Attention over the KV cache adds bytes but few FLOPs, pushing decode further left.)
"""

import argparse

GPUS = {  # name: (dense BF16 TFLOP/s, memory GB/s)  -- approximate datasheet peaks
    "t4": (65, 320), "a10g": (70, 600), "l4": (121, 300), "l40s": (362, 864),
    "a100": (312, 2039), "h100": (989, 3350), "h200": (989, 4800),
}


def intensity(tokens_per_weight_read: int, bytes_per_weight: float = 2.0) -> float:
    return 2 * tokens_per_weight_read / bytes_per_weight


def attainable_tflops(gpu: str, ai: float) -> float:
    peak, bw = GPUS[gpu]
    return min(peak, bw * ai / 1000)


def ridge(gpu: str) -> float:
    peak, bw = GPUS[gpu]
    return peak * 1000 / bw


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", default="h100", choices=GPUS)
    p.add_argument("--batch", type=int, nargs="+", default=[1, 8, 32, 128, 512])
    p.add_argument("--prompt", type=int, default=1024)
    p.add_argument("--bytes-per-weight", type=float, default=2.0, help="2=BF16, 1=FP8, 0.5=INT4")
    a = p.parse_args()

    peak, bw = GPUS[a.gpu]
    r = ridge(a.gpu)
    print(f"{a.gpu}: peak {peak} TFLOP/s, {bw} GB/s -> ridge point {r:.0f} FLOPs/byte\n")
    print(f"{'phase':<22}{'FLOPs/byte':>12}{'attainable':>14}{'% of peak':>11}  bound")
    rows = [(f"decode batch {b}", b) for b in a.batch] + [(f"prefill {a.prompt} tok", a.prompt)]
    for name, n in rows:
        ai = intensity(n, a.bytes_per_weight)
        t = attainable_tflops(a.gpu, ai)
        bound = "memory" if ai < r else "compute"
        bar = "#" * max(1, int(30 * t / peak))
        print(f"{name:<22}{ai:>12.0f}{t:>11.0f} TF{t / peak:>10.0%}  {bound:<8}{bar}")
    print(f"\nDecode needs a batch of ~{r * a.bytes_per_weight / 2:.0f} sequences before it uses the tensor cores fully."
          "\nThat is why serving systems batch aggressively, and why quantizing weights"
          "\n(fewer bytes per weight) speeds up decode even when FLOPs stay the same.")


if __name__ == "__main__":
    main()
