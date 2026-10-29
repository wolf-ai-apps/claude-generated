"""Lab 1b: memory-bound vs compute-bound, measured on your own machine.

Arithmetic intensity = FLOPs performed / bytes moved from memory.

  elementwise add  c = a + b       : 1 FLOP per 12 bytes  -> memory-bound
  matrix multiply  C = A @ B (n^3) : 2n^3 FLOPs per 12n^2 bytes -> compute-bound for big n

LLM decode is closer to the first; prefill is closer to the second.

    python3 memory_vs_compute.py
"""

import time

import numpy as np


def best(fn, repeat=5):
    fn()
    t = float("inf")
    for _ in range(repeat):
        s = time.perf_counter()
        fn()
        t = min(t, time.perf_counter() - s)
    return t


def elementwise(n=20_000_000):
    a = np.ones(n, np.float32)
    b = np.ones(n, np.float32)
    c = np.empty_like(a)
    t = best(lambda: np.add(a, b, out=c))
    bytes_moved = 3 * n * 4          # read a, read b, write c
    flops = n
    return flops / t / 1e9, bytes_moved / t / 1e9, flops / bytes_moved


def matmul(n):
    a = np.ones((n, n), np.float32)
    b = np.ones((n, n), np.float32)
    t = best(lambda: a @ b, repeat=3)
    flops = 2 * n**3
    bytes_moved = 3 * n * n * 4      # ideal: read A, read B, write C once
    return flops / t / 1e9, bytes_moved / t / 1e9, flops / bytes_moved


def main():
    print(f"{'kernel':<22}{'GFLOP/s':>10}{'GB/s':>10}{'FLOPs/byte':>12}")
    g, bw, ai = elementwise()
    print(f"{'add 20M':<22}{g:>10.1f}{bw:>10.1f}{ai:>12.3f}")
    for n in (64, 256, 1024, 2048):
        g, bw, ai = matmul(n)
        print(f"{f'matmul {n}x{n}':<22}{g:>10.1f}{bw:>10.2f}{ai:>12.1f}")
    print("\nRead it like this: the add is limited by GB/s (it barely does any math);"
          "\nthe big matmuls are limited by GFLOP/s. Same machine, two different ceilings.")


if __name__ == "__main__":
    main()
