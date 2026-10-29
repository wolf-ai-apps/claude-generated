"""Lab 2c: read your GPU's specs the way an inference engineer does.

    python3 gpu_info.py

Prints memory, SM count and compute capability for each GPU, then measures
real device-to-device bandwidth (the number that bounds LLM decode speed).
"""

import torch


def main():
    if not torch.cuda.is_available():
        print("No CUDA GPU visible. Check `nvidia-smi` and level 3.")
        return
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {p.name}")
        print(f"  memory              {p.total_memory / 1e9:.1f} GB")
        print(f"  SMs                 {p.multi_processor_count}")
        print(f"  compute capability  {p.major}.{p.minor}"
              "   (8.0 A100, 8.6 A10/A10G, 8.9 L4/L40S, 9.0 H100/H200, 10.0 B200)")
        print(f"  BF16 supported      {torch.cuda.is_bf16_supported()}")
        print(f"  FP8 tensor cores    {'yes' if (p.major, p.minor) >= (8, 9) else 'no'}")

        x = torch.empty(512 * 1024**2, dtype=torch.uint8, device=i)  # 512 MiB
        y = torch.empty_like(x)
        for _ in range(3):
            y.copy_(x)
        torch.cuda.synchronize(i)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        iters = 20
        for _ in range(iters):
            y.copy_(x)
        end.record()
        torch.cuda.synchronize(i)
        ms = start.elapsed_time(end) / iters
        gbs = 2 * x.numel() / (ms / 1e3) / 1e9  # read + write
        print(f"  measured bandwidth  {gbs:.0f} GB/s (copy); compare with the datasheet peak")


if __name__ == "__main__":
    main()
