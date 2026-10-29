"""Lab 4b: count a model's parameters from its config.json, by hand.

    python3 count_params.py                       # built-in Llama 3.1 8B / 70B configs
    python3 count_params.py path/to/config.json   # any Llama-style model

Works for Llama/Qwen/Mistral-style decoders: RMSNorm, GQA attention, SwiGLU MLP.
"""

import json
import sys

LLAMA_31_8B = {
    "hidden_size": 4096, "intermediate_size": 14336, "num_hidden_layers": 32,
    "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 128256,
    "tie_word_embeddings": False,
}
LLAMA_31_70B = {
    "hidden_size": 8192, "intermediate_size": 28672, "num_hidden_layers": 80,
    "num_attention_heads": 64, "num_key_value_heads": 8, "vocab_size": 128256,
    "tie_word_embeddings": False,
}


def count(cfg: dict) -> dict:
    d = cfg["hidden_size"]
    ff = cfg["intermediate_size"]
    L = cfg["num_hidden_layers"]
    h = cfg["num_attention_heads"]
    kv = cfg.get("num_key_value_heads", h)
    hd = cfg.get("head_dim", d // h)
    V = cfg["vocab_size"]
    bias = cfg.get("attention_bias", False)

    attn = d * (h * hd) + 2 * d * (kv * hd) + (h * hd) * d   # q, k, v, o projections
    if bias:
        attn += h * hd + 2 * kv * hd
    mlp = 3 * d * ff                                         # gate, up, down (SwiGLU)
    norms = 2 * d                                            # two RMSNorm weight vectors
    per_layer = attn + mlp + norms
    embed = V * d
    lm_head = 0 if cfg.get("tie_word_embeddings", False) else V * d
    total = embed + L * per_layer + d + lm_head              # + final norm
    return {"embedding": embed, "attention/layer": attn, "mlp/layer": mlp,
            "per layer": per_layer, "all layers": L * per_layer, "lm_head": lm_head, "total": total}


def show(name: str, cfg: dict) -> None:
    c = count(cfg)
    print(f"\n{name}")
    for k, v in c.items():
        print(f"  {k:<16}{v/1e6:>12,.1f} M   ({v / c['total']:.1%})")
    print(f"  => {c['total']/1e9:.2f} B params, {c['total']*2/1e9:.1f} GB in BF16, "
          f"{c['total']/1e9:.1f} GB in FP8, {c['total']*0.5/1e9:.1f} GB in INT4")


def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            show(sys.argv[1], json.load(f))
        return
    show("Llama 3.1 8B", LLAMA_31_8B)
    show("Llama 3.1 70B", LLAMA_31_70B)
    print("\nNotice: the MLP holds ~2/3 of the weights; GQA makes k/v projections small.")


if __name__ == "__main__":
    main()
