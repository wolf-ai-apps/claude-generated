"""Lab 4c: scaled dot-product attention, multi-head and grouped-query, in NumPy.

    python3 attention.py

Shapes used throughout:
    T  = sequence length
    d  = model width (hidden_size)
    h  = query heads, kv = key/value heads, hd = head dim (d = h * hd)
"""

import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)  # subtract max for numerical stability
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def causal_mask(t_q: int, t_k: int) -> np.ndarray:
    """True where attention is NOT allowed. Query i (absolute pos t_k - t_q + i) sees keys <= its position."""
    offset = t_k - t_q
    return np.triu(np.ones((t_q, t_k), dtype=bool), k=1 + offset)


def attention(q, k, v, causal=True):
    """q: (h, Tq, hd); k, v: (kv, Tk, hd) with h % kv == 0. Returns (h, Tq, hd).

    Grouped-query attention (GQA): each group of h/kv query heads shares one K/V head.
    kv == h is ordinary multi-head attention (MHA); kv == 1 is multi-query attention (MQA).
    """
    h, t_q, hd = q.shape
    kv = k.shape[0]
    assert h % kv == 0
    k = np.repeat(k, h // kv, axis=0)   # share each K/V head across its query group
    v = np.repeat(v, h // kv, axis=0)
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(hd)       # (h, Tq, Tk)
    if causal:
        scores = np.where(causal_mask(t_q, k.shape[1]), -np.inf, scores)
    return softmax(scores) @ v                             # (h, Tq, hd)


def main():
    rng = np.random.default_rng(0)
    T, h, kv, hd = 6, 8, 2, 16
    q = rng.standard_normal((h, T, hd))
    k = rng.standard_normal((kv, T, hd))
    v = rng.standard_normal((kv, T, hd))
    out = attention(q, k, v)
    print("output shape", out.shape)

    w = softmax(np.where(causal_mask(T, T), -np.inf, q[0] @ np.repeat(k, h // kv, 0)[0].T / np.sqrt(hd)))
    print("\nattention weights of head 0 (rows = query position, cols = key position):")
    print(np.array2string(w, precision=2, suppress_small=True))
    print("upper triangle is 0: a token can only look at itself and the past (causal).")

    kv_bytes_mha = 2 * h * T * hd * 2
    kv_bytes_gqa = 2 * kv * T * hd * 2
    print(f"\nKV cache for these {T} tokens in BF16: MHA {kv_bytes_mha} B vs GQA {kv_bytes_gqa} B "
          f"({h // kv}x smaller). That ratio is why Llama 3 uses 8 KV heads for 32 query heads.")


if __name__ == "__main__":
    main()
