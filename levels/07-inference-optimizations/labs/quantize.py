"""Lab 7c: weight quantization (INT8 per-channel, INT4 group-wise) from scratch.

    python3 quantize.py

Decode reads every weight for every step, so fewer bytes per weight means
proportionally faster decode (until compute-bound) and more room for KV cache.
The question is always: how much accuracy does it cost?
"""

import numpy as np


def quant_int8_per_channel(w):
    """Symmetric INT8, one scale per output channel (row)."""
    scale = np.abs(w).max(axis=1, keepdims=True) / 127.0
    scale[scale == 0] = 1.0
    q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    return q, scale.astype(np.float32)


def dequant_int8(q, scale):
    return q.astype(np.float32) * scale


def quant_int8_per_tensor(w):
    scale = np.abs(w).max() / 127.0
    return np.clip(np.round(w / scale), -127, 127).astype(np.int8), np.float32(scale)


def quant_int4_group(w, group=128):
    """Asymmetric INT4 with a scale and zero-point per group of `group` input weights
    (the layout GPTQ/AWQ checkpoints use). Two 4-bit values are packed per byte."""
    rows, cols = w.shape
    assert cols % group == 0 and group % 2 == 0
    g = w.reshape(rows, cols // group, group)
    lo, hi = g.min(-1, keepdims=True), g.max(-1, keepdims=True)
    scale = (hi - lo) / 15.0
    scale[scale == 0] = 1.0
    zero = np.round(-lo / scale)
    q = np.clip(np.round(g / scale + zero), 0, 15).astype(np.uint8).reshape(rows, cols)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    return packed, scale.astype(np.float16), zero.astype(np.uint8)


def dequant_int4_group(packed, scale, zero, group=128):
    rows = packed.shape[0]
    q = np.empty((rows, packed.shape[1] * 2), np.uint8)
    q[:, 0::2] = packed & 0x0F
    q[:, 1::2] = packed >> 4
    g = q.reshape(rows, -1, group).astype(np.float32)
    return ((g - zero) * scale.astype(np.float32)).reshape(rows, -1)


def rel_err(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(a))


def main():
    rng = np.random.default_rng(0)
    d_in, d_out = 4096, 1024
    w = (rng.standard_normal((d_out, d_in)) * 0.02).astype(np.float32)
    x = rng.standard_normal((8, d_in)).astype(np.float32)   # a batch of activations
    y = x @ w.T

    q8, s8 = quant_int8_per_channel(w)
    p4, s4, z4 = quant_int4_group(w)
    w8, w4 = dequant_int8(q8, s8), dequant_int4_group(p4, s4, z4)

    bf16_bytes = w.size * 2
    rows = [("BF16 (baseline)", bf16_bytes, 0.0, 0.0),
            ("INT8 per-channel", q8.nbytes + s8.nbytes, rel_err(w, w8), rel_err(y, x @ w8.T)),
            ("INT4 group=128", p4.nbytes + s4.nbytes + z4.nbytes, rel_err(w, w4), rel_err(y, x @ w4.T))]
    print(f"{'format':<20}{'MB':>8}{'vs BF16':>9}{'weight err':>12}{'output err':>12}")
    for name, b, we, oe in rows:
        print(f"{name:<20}{b / 1e6:>8.2f}{b / bf16_bytes:>9.0%}{we:>12.2%}{oe:>12.2%}")

    w_out = w.copy()
    w_out[:, 7] *= 50                                       # one outlier input channel
    qt, st = quant_int8_per_tensor(w_out)
    qc, sc = quant_int8_per_channel(w_out)
    print("\nWith one outlier column (50x larger), the error depends on WHERE the scale lives:")
    print(f"  INT8 per-tensor : {rel_err(w_out, qt.astype(np.float32) * st):.2%}  one scale for everything")
    print(f"  INT8 per-channel: {rel_err(w_out, dequant_int8(qc, sc)):.2%}")
    print("\nOutliers in ACTIVATIONS are the harder problem (LLM.int8(), SmoothQuant, FP8 scaling);"
          "\nalways validate a quantized model on your own eval set, not just perplexity.")


if __name__ == "__main__":
    main()
