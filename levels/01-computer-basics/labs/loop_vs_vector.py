"""Lab 1a: one worker vs many workers.

A Python for-loop processes one element at a time. NumPy hands the whole
array to compiled code that uses SIMD instructions (several elements per
instruction). A GPU takes the same idea much further: thousands of threads.

    python3 loop_vs_vector.py
"""

import time

import numpy as np


def timeit(fn, repeat=3):
    best = float("inf")
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def main(n=2_000_000):
    a = np.random.rand(n).astype(np.float32)
    b = np.random.rand(n).astype(np.float32)
    al, bl = a.tolist(), b.tolist()

    t_loop = timeit(lambda: [x + y for x, y in zip(al, bl)], repeat=1)
    t_vec = timeit(lambda: a + b)
    print(f"add {n:,} floats")
    print(f"  python loop : {t_loop*1000:8.1f} ms")
    print(f"  numpy (SIMD): {t_vec*1000:8.2f} ms   -> {t_loop/t_vec:,.0f}x faster")

    try:
        import torch
        if torch.cuda.is_available():
            ga, gb = torch.from_numpy(a).cuda(), torch.from_numpy(b).cuda()
            torch.cuda.synchronize()
            def gpu_add():
                ga + gb
                torch.cuda.synchronize()  # GPU calls are async: wait before stopping the clock
            gpu_add()  # warm-up (CUDA context, kernel load)
            t_gpu = timeit(gpu_add)
            print(f"  GPU (torch) : {t_gpu*1000:8.3f} ms   -> {t_vec/t_gpu:,.1f}x faster than numpy")
            t_copy = timeit(lambda: (torch.from_numpy(a).cuda(), torch.cuda.synchronize()))
            print(f"  ...but copying the input to the GPU took {t_copy*1000:.2f} ms over PCIe."
                  " Data movement often costs more than the math.")
        else:
            print("  (no CUDA GPU found; rerun on a GPU machine to see the third line)")
    except ImportError:
        print("  (install torch to compare against a GPU)")


if __name__ == "__main__":
    main()
