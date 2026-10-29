"""Lab 2b: the same kernel in Triton, then a fused kernel.

Triton lets you write GPU kernels in Python at the level of *blocks* of data
rather than single threads. vLLM and SGLang use Triton for many kernels.

    pip install torch triton
    python3 vector_add_triton.py
"""

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)                 # which block am I?
    offs = pid * BLOCK + tl.arange(0, BLOCK)    # the BLOCK element indices this program owns
    mask = offs < n                             # guard the tail
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(c_ptr + offs, a + b, mask=mask)


@triton.jit
def fused_add_relu_scale_kernel(a_ptr, b_ptr, c_ptr, n, scale, BLOCK: tl.constexpr):
    # One pass over memory instead of three: this is kernel fusion.
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(a_ptr + offs, mask=mask) + tl.load(b_ptr + offs, mask=mask)
    tl.store(c_ptr + offs, tl.maximum(x, 0.0) * scale, mask=mask)


def add(a, b):
    c = torch.empty_like(a)
    n = a.numel()
    grid = (triton.cdiv(n, 1024),)
    add_kernel[grid](a, b, c, n, BLOCK=1024)
    return c


def fused(a, b, scale):
    c = torch.empty_like(a)
    n = a.numel()
    fused_add_relu_scale_kernel[(triton.cdiv(n, 1024),)](a, b, c, n, scale, BLOCK=1024)
    return c


def bench(fn):
    return triton.testing.do_bench(fn)  # median ms, handles warm-up and sync


def main():
    n = 1 << 26
    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")
    assert torch.allclose(add(a, b), a + b)
    assert torch.allclose(fused(a, b, 0.5), torch.relu(a + b) * 0.5)

    gb = 3 * a.numel() * a.element_size() / 1e9
    for name, fn in [("torch a+b", lambda: a + b), ("triton add", lambda: add(a, b))]:
        ms = bench(fn)
        print(f"{name:<28}{ms:7.3f} ms  {gb / (ms / 1e3):7.0f} GB/s")

    ms_unfused = bench(lambda: torch.relu(a + b) * 0.5)
    ms_fused = bench(lambda: fused(a, b, 0.5))
    print(f"{'torch add+relu+scale (3 ops)':<28}{ms_unfused:7.3f} ms")
    print(f"{'triton fused (1 kernel)':<28}{ms_fused:7.3f} ms  -> {ms_unfused / ms_fused:.1f}x")
    print("Each unfused op reads and writes the whole array from HBM; the fused kernel does it once.")


if __name__ == "__main__":
    main()
