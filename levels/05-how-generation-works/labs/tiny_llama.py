"""Lab 5a: a complete Llama-style decoder in NumPy, with and without a KV cache.

The weights are random (the text is gibberish), but the architecture and the
inference mechanics are real: RMSNorm, RoPE, grouped-query attention, SwiGLU,
prefill vs decode, and a KV cache. The test proves the cache is exact.

    python3 tiny_llama.py
"""

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class Config:
    vocab: int = 256
    d: int = 128          # hidden_size
    layers: int = 4
    heads: int = 8
    kv_heads: int = 2     # GQA: 4 query heads share each K/V head
    ff: int = 344         # intermediate_size
    rope_theta: float = 10000.0

    @property
    def hd(self):
        return self.d // self.heads


def init_weights(cfg: Config, seed=0):
    rng = np.random.default_rng(seed)
    s = 0.02
    w = {"embed": rng.standard_normal((cfg.vocab, cfg.d)) * s,
         "norm_f": np.ones(cfg.d),
         "lm_head": rng.standard_normal((cfg.d, cfg.vocab)) * s,
         "layers": []}
    for _ in range(cfg.layers):
        w["layers"].append({
            "norm1": np.ones(cfg.d), "norm2": np.ones(cfg.d),
            "wq": rng.standard_normal((cfg.d, cfg.heads * cfg.hd)) * s,
            "wk": rng.standard_normal((cfg.d, cfg.kv_heads * cfg.hd)) * s,
            "wv": rng.standard_normal((cfg.d, cfg.kv_heads * cfg.hd)) * s,
            "wo": rng.standard_normal((cfg.heads * cfg.hd, cfg.d)) * s,
            "w_gate": rng.standard_normal((cfg.d, cfg.ff)) * s,
            "w_up": rng.standard_normal((cfg.d, cfg.ff)) * s,
            "w_down": rng.standard_normal((cfg.ff, cfg.d)) * s,
        })
    return w


def rms_norm(x, weight, eps=1e-6):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * weight


def rope(x, positions, theta):
    """Rotary position embedding (HF Llama 'rotate_half' layout). x: (heads, T, hd)."""
    hd = x.shape[-1]
    inv_freq = 1.0 / theta ** (np.arange(0, hd, 2) / hd)       # (hd/2,)
    ang = np.outer(positions, inv_freq)                        # (T, hd/2)
    cos = np.concatenate([np.cos(ang)] * 2, -1)                # (T, hd)
    sin = np.concatenate([np.sin(ang)] * 2, -1)
    x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
    rotated = np.concatenate([-x2, x1], -1)
    return x * cos + rotated * sin


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def silu(x):
    return x / (1 + np.exp(-x))


class KVCache:
    """Per layer, keys and values of every token processed so far: (kv_heads, T, hd)."""

    def __init__(self, cfg: Config):
        self.k = [np.zeros((cfg.kv_heads, 0, cfg.hd)) for _ in range(cfg.layers)]
        self.v = [np.zeros((cfg.kv_heads, 0, cfg.hd)) for _ in range(cfg.layers)]

    @property
    def length(self):
        return self.k[0].shape[1]

    def nbytes(self, bytes_per_elem=2):
        return sum(k.size + v.size for k, v in zip(self.k, self.v)) * bytes_per_elem


def forward(cfg: Config, w, tokens: list[int], cache: KVCache | None = None):
    """Run the model on `tokens`. With a cache, tokens are the NEW tokens only and
    attend to everything already in the cache. Returns logits (T, vocab)."""
    start = cache.length if cache else 0
    T = len(tokens)
    pos = np.arange(start, start + T)
    x = w["embed"][tokens]                                     # (T, d)
    group = cfg.heads // cfg.kv_heads
    for i, L in enumerate(w["layers"]):
        h = rms_norm(x, L["norm1"])
        q = (h @ L["wq"]).reshape(T, cfg.heads, cfg.hd).transpose(1, 0, 2)
        k = (h @ L["wk"]).reshape(T, cfg.kv_heads, cfg.hd).transpose(1, 0, 2)
        v = (h @ L["wv"]).reshape(T, cfg.kv_heads, cfg.hd).transpose(1, 0, 2)
        q, k = rope(q, pos, cfg.rope_theta), rope(k, pos, cfg.rope_theta)
        if cache is not None:
            k = cache.k[i] = np.concatenate([cache.k[i], k], axis=1)
            v = cache.v[i] = np.concatenate([cache.v[i], v], axis=1)
        Tk = k.shape[1]
        kk, vv = np.repeat(k, group, 0), np.repeat(v, group, 0)
        scores = q @ kk.transpose(0, 2, 1) / np.sqrt(cfg.hd)   # (heads, T, Tk)
        mask = np.triu(np.ones((T, Tk), bool), k=1 + Tk - T)
        attn = softmax(np.where(mask, -np.inf, scores)) @ vv   # (heads, T, hd)
        x = x + attn.transpose(1, 0, 2).reshape(T, -1) @ L["wo"]
        h = rms_norm(x, L["norm2"])
        x = x + (silu(h @ L["w_gate"]) * (h @ L["w_up"])) @ L["w_down"]
    return rms_norm(x, w["norm_f"]) @ w["lm_head"]


def generate_no_cache(cfg, w, prompt, n):
    """Recompute the whole sequence for every new token: O(T^2) work per token."""
    seq = list(prompt)
    for _ in range(n):
        logits = forward(cfg, w, seq)
        seq.append(int(np.argmax(logits[-1])))
    return seq[len(prompt):]


def generate_with_cache(cfg, w, prompt, n, timings=None):
    """Prefill once, then decode one token at a time reusing cached K/V."""
    cache = KVCache(cfg)
    t = time.perf_counter()
    logits = forward(cfg, w, list(prompt), cache)              # PREFILL: whole prompt in one pass
    out = [int(np.argmax(logits[-1]))]
    if timings is not None:
        timings.append(("prefill", time.perf_counter() - t))
    for _ in range(n - 1):
        t = time.perf_counter()
        logits = forward(cfg, w, [out[-1]], cache)             # DECODE: one token
        out.append(int(np.argmax(logits[-1])))
        if timings is not None:
            timings.append(("decode", time.perf_counter() - t))
    return out, cache


def main():
    cfg = Config()
    w = init_weights(cfg)
    prompt = list(b"The KV cache makes decoding fast because ")
    n = 48

    t = time.perf_counter()
    a = generate_no_cache(cfg, w, prompt, n)
    t_no = time.perf_counter() - t

    timings = []
    t = time.perf_counter()
    b, cache = generate_with_cache(cfg, w, prompt, n, timings)
    t_kv = time.perf_counter() - t

    assert a == b, "KV cache must not change the output"
    decode = [s for k, s in timings if k == "decode"]
    print(f"prompt {len(prompt)} tokens, generate {n} tokens (greedy)")
    print(f"  without KV cache: {t_no*1000:7.1f} ms")
    print(f"  with KV cache:    {t_kv*1000:7.1f} ms   ({t_no/t_kv:.1f}x faster, identical tokens)")
    print(f"  prefill {timings[0][1]*1000:.2f} ms for {len(prompt)} tokens; "
          f"decode {np.median(decode)*1000:.2f} ms per token")
    print(f"  cache now holds {cache.length} tokens = {cache.nbytes()/1024:.0f} KiB in BF16 "
          f"(2 x {cfg.layers} layers x {cfg.kv_heads} kv_heads x {cfg.hd} head_dim x 2 bytes per token)")


if __name__ == "__main__":
    main()
