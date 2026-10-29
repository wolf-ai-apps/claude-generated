# Level 4: Models from first principles

> You can't size, serve or optimize a model you can't read. By the end of this level, a `config.json` tells you its memory footprint and where the work goes.

**You will:** learn tensors and matmuls, tokenization, the transformer decoder block, MHA/MQA/GQA attention, and parameter counting, and implement each in NumPy.
**Time:** 4–5 hours · **Hardware:** laptop (Tier 0)

---

## 1. Concepts

### 1.1 Everything is a matrix multiply

A **tensor** is an n-dimensional array. A **linear layer** `y = x @ W` multiplies a batch of input vectors `x` (shape `[tokens, d_in]`) by a weight matrix `W` (`[d_in, d_out]`). It costs `2 × tokens × d_in × d_out` FLOPs and reads `d_in × d_out` weights. More than 95% of a transformer's FLOPs are in matmuls like this one, which is why GPUs (tensor cores) and memory bandwidth matter so much.

### 1.2 Tokenization

Models don't see text. They see **token IDs**. A tokenizer splits text into pieces from a fixed vocabulary (32k–256k entries).

- **Byte-pair encoding (BPE):** start from bytes, then repeatedly merge the most frequent adjacent pair into a new token. Frequent words become single tokens; rare words split into pieces.
- **Rule of thumb:** ~4 characters or ~0.75 English words per token. Code, non-English scripts and numbers usually take more tokens.
- **Why it matters for inference:** latency, KV cache memory, context limits and price are all counted in tokens. Hindi or Japanese text can cost 2–4× more tokens than the same meaning in English with an English-heavy tokenizer.

### 1.3 The decoder-only transformer (Llama-style)

```
token IDs ──► embedding lookup ──► x  (shape [T, d])
                                   │
          ┌────────── repeated L times (L = 32 for Llama 3.1 8B) ──────────┐
          │  h = RMSNorm(x)                                                 │
          │  q, k, v = h @ Wq, h @ Wk, h @ Wv        (projections)          │
          │  q, k = RoPE(q, k, positions)            (position information) │
          │  a = softmax(q kᵀ / √d_head + causal mask) v    (ATTENTION)      │
          │  x = x + a @ Wo                          (residual)             │
          │  h = RMSNorm(x)                                                 │
          │  x = x + W_down( SiLU(h @ W_gate) * (h @ W_up) )   (SwiGLU MLP) │
          └─────────────────────────────────────────────────────────────────┘
                                   │
                    RMSNorm ──► x @ W_lm_head ──► logits  ([T, vocab]) ──► sampler (level 5)
```

- **Attention** is the only place tokens interact. Each token's query is compared with every earlier token's key, and the scores weight a sum of their values.
- **Causal mask:** a token can only attend to itself and earlier tokens. That's what makes generation one-token-at-a-time and makes the KV cache possible (level 5).
- **MLP** transforms each token independently and holds most of the parameters.
- **RoPE** encodes position by rotating q and k. **RMSNorm** keeps activations well-scaled.

### 1.4 MHA, MQA and GQA

Attention uses `h` query heads. The question is how many key/value heads there are, because **K and V are what gets cached** for every token:

| Variant | KV heads | KV cache size | Quality | Used by |
|---|---|---|---|---|
| Multi-head (MHA) | = h | 1× (largest) | baseline | GPT-2, Llama 1/2 7B |
| Grouped-query (GQA) | h / g (e.g. 8) | g× smaller | ≈ MHA | Llama 3, Qwen 2.5, Mistral |
| Multi-query (MQA) | 1 | h× smaller | slightly lower | some older models (PaLM, Falcon) |

Llama 3.1 8B has 32 query heads and **8 KV heads**, so its KV cache is 4× smaller than MHA would need. That means 4× more concurrent users per GPU. (DeepSeek's **MLA** compresses KV further with a learned low-rank projection.)

### 1.5 Dense vs Mixture-of-Experts (MoE)

An MoE model replaces each MLP with N "expert" MLPs plus a router that sends each token to the top-k experts (e.g. 2 of 8).

- **Total parameters** (all experts) decide **memory**: Mixtral 8×7B needs ~94 GB in BF16.
- **Active parameters** per token decide **FLOPs** and, at small batch, bytes read: ~13B for Mixtral.
- Serving challenges: every expert must stay in memory, load across experts is uneven, and expert parallelism needs all-to-all communication (level 7).

### 1.6 Model files on the Hugging Face Hub

```
config.json                 architecture: layers, hidden size, heads, kv heads, vocab, rope, max positions
generation_config.json      default sampling params, EOS token ids
tokenizer.json, tokenizer_config.json (incl. chat template)
model-00001-of-00004.safetensors ...  weights, sharded
model.safetensors.index.json          which tensor lives in which shard
```

- **safetensors** is a simple, memory-mappable format with no code execution. Avoid pickle-based `.bin`/`.pt` checkpoints from untrusted sources, because unpickling can run arbitrary code (level 14).
- The **chat template** turns a list of messages into the exact token sequence the model was trained on. A wrong template silently degrades quality.
- **Gated** models (Llama) need a Hugging Face token and an accepted license. Ungated alternatives for labs: Qwen2.5 (0.5B–72B).

---

## 2. Worked example: Llama 3.1 8B from its config

```json
{"hidden_size": 4096, "intermediate_size": 14336, "num_hidden_layers": 32,
 "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 128256}
```

`head_dim = 4096 / 32 = 128`. KV projection width = 8 × 128 = 1024.

| Part | Formula | Parameters |
|---|---|---|
| Embedding | 128256 × 4096 | 525.3 M |
| Attention per layer | q: 4096×4096 + k: 4096×1024 + v: 4096×1024 + o: 4096×4096 | 41.9 M |
| MLP per layer | 3 × 4096 × 14336 | 176.2 M |
| Norms per layer | 2 × 4096 | 0.008 M |
| × 32 layers | | 6,979.6 M |
| LM head (untied) | 4096 × 128256 | 525.3 M |
| **Total** | | **8.03 B** |

What it means: **16.1 GB in BF16**, 8.0 GB in FP8, ~4 GB in INT4 (plus group scales). The MLP holds ~70% of the weights, so it dominates decode bytes. Attention projections are small because of GQA.

## 3. Labs

```bash
cd levels/04-models-first-principles/labs
python3 bpe.py            # train a tiny BPE tokenizer; compare token counts across English, Hindi, code
python3 count_params.py   # reproduce the table above for 8B and 70B; pass any config.json to count it
python3 attention.py      # MHA vs GQA attention with a causal mask; see the attention matrix
python3 -m unittest test_level04   # tests: tokenizer round-trip, param counts, GQA == MHA with repeated KV
```

Exercises:
1. Download `config.json` for `Qwen/Qwen2.5-7B-Instruct` and run `count_params.py` on it. Qwen uses attention biases and tied/untied embeddings differently, so check that your total matches the published size.
2. In `attention.py`, set `kv = 1` (MQA) and `kv = h` (MHA). Confirm the output shape is unchanged and compute the KV-cache saving.
3. In `bpe.py`, raise `vocab_size` to 1000. What happens to token counts for corpus-like text vs Hindi? Why do production tokenizers have 100k+ entries?

## 4. Check yourself

<details><summary>1. A model has 70B parameters. How much GPU memory do its weights need in BF16, FP8 and INT4?</summary>
~140 GB, ~70 GB, ~35 GB (plus a little for scales and zero-points).</details>

<details><summary>2. Why does GQA matter more for serving than for model quality?</summary>
It barely changes quality, but it shrinks the KV cache by num_heads / num_kv_heads, which multiplies how many concurrent sequences (and how much context) fit in GPU memory.</details>

<details><summary>3. Mixtral 8×7B has ~47B total and ~13B active parameters. Which number sets memory, and which sets FLOPs per token?</summary>
Total parameters set memory (every expert must be resident). Active parameters set FLOPs per token.</details>

<details><summary>4. Why is the causal mask essential for efficient generation?</summary>
Because token i never depends on later tokens, earlier tokens' keys and values never change as the sequence grows. They can be computed once and cached.</details>

## 5. Going deeper
- Vaswani et al., *Attention Is All You Need* (2017); Ainslie et al., *GQA* (2023)
- Andrej Karpathy, *Let's build GPT* and *Let's build the GPT tokenizer*; nanoGPT, minbpe
- Jay Alammar, *The Illustrated Transformer*
- The Llama 3 paper (*The Llama 3 Herd of Models*): architecture and scaling choices

**Next → [Level 5: How generation works](../05-how-generation-works/README.md)**
