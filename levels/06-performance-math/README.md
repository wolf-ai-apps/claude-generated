# Level 6: Performance math

> If you can estimate memory, speed and cost on a whiteboard to within 2×, you can make good decisions before spending a dollar on GPUs. This is the most useful skill in the course.

**You will:** learn six formulas, the roofline model applied to LLMs, and a calculator (Go) you'll use for every later decision.
**Time:** 3–4 hours · **Hardware:** laptop (Tier 0)

---

## 1. The six formulas

Let `P` = parameters, `b` = bytes per weight, `L` = layers, `Hkv` = KV heads, `d` = head dim, `bkv` = bytes per KV element.

| # | Quantity | Formula | Llama 3.1 8B, BF16 |
|---|---|---|---|
| 1 | Weight memory | `P × b` | 8.03e9 × 2 = **16.1 GB** |
| 2 | KV cache per token | `2 × L × Hkv × d × bkv` | 2×32×8×128×2 = **128 KiB** |
| 3 | Max concurrent sequences | `(mem × util − weights − overhead) / (KV/token × context)` | (24×0.9 − 16.1 − 1) / (128 KiB × 4096) ≈ **8** on a 24 GB GPU |
| 4 | Decode step time (memory-bound) | `(active weight bytes + batch × KV bytes read) / (bandwidth × MBU)` | (16.1 GB + 8 × 0.54 GB) / (600 GB/s × 0.75) ≈ **45 ms** on an A10G |
| 5 | Prefill time (compute-bound) | `2 × P_active × prompt_tokens / (peak FLOP/s × MFU)` | 2×8e9×1024 / (70e12×0.5) ≈ **470 ms** on an A10G |
| 6 | Cost per 1M output tokens | `$/hour / (aggregate tok/s × 3600) × 1e6` | $1/h ÷ (177 tok/s × 3600) × 1e6 ≈ **$1.57** |

**MBU** (memory bandwidth utilization, typically 0.6–0.85) and **MFU** (model FLOPs utilization, typically 0.3–0.6) are how far real kernels fall short of the datasheet. Measure your own (level 2's `gpu_info.py`, level 9's benchmarks) and plug them in.

Conservative simplification: formula 3 uses the *full* context per sequence. Real requests are at various stages of generation, so measured concurrency is usually higher. vLLM prints its own number at startup ("Maximum concurrency for N tokens per request").

## 2. The roofline model for LLMs

```
attainable FLOP/s = min( peak FLOP/s ,  memory bandwidth × arithmetic intensity )

          FLOP/s ▲
     peak ───────│...........................●━━━━━━━━━━━━━━━  compute-bound region
                 │                        ╱  ridge point = peak / bandwidth
                 │                     ╱       (H100: 989e12 / 3.35e12 ≈ 295 FLOPs/byte)
                 │                  ╱
                 │  decode b=1 ● ╱     memory-bound region (slope = bandwidth)
                 └──────────────────────────────────────────────►  FLOPs per byte (log)
                        1      8     32    128   295    1024
```

For the weight matmuls in BF16 (2 bytes per weight, 2 FLOPs per weight per token):
- **Decode with batch B:** each weight is read once and used B times, so intensity ≈ **B FLOPs/byte**.
- **Prefill of P tokens:** intensity ≈ **P FLOPs/byte**.

So:
1. **Batch-1 decode on an H100 uses well under 1% of its FLOPs.** Batching is the main lever.
2. Decode stays memory-bound until batch ≈ ridge point (~295 on H100, ~400 on an L4 with 121 TFLOP/s over only 300 GB/s). In practice the KV cache runs out of memory long before that.
3. **Quantizing weights** (FP8: 1 byte, INT4: 0.5 bytes) doubles or quadruples intensity and cuts bytes per step. Decode gets faster even though the FLOPs are unchanged.
4. Attention over the KV cache adds bytes with few FLOPs, so long contexts push decode further into the memory-bound region.

## 3. Worked examples

**Example A: one GPU or two for Llama 3.1 8B?**

```
$ go run ./cmd/infercalc -model llama-3.1-8b -gpu a10g -context 4096 -prompt 1024 -price 1.0
Weights (bf16):        16.1 GB  = 8.03B params x 2.0 bytes
KV cache per token:  128 KiB = 2 x 32 layers x 8 kv_heads x 128 head_dim x 2.0 bytes (bf16)
KV per sequence:     0.54 GB  at 4096 tokens of context
Usable memory:       21.6 GB  (90% of 24 GB x 1)
KV budget:           4.5 GB  = usable - weights - overhead
Max concurrent seqs: 8
At batch 8:
  decode step        45.2 ms
  per-user speed     22.1 tok/s
  aggregate          177 tok/s
Prefill (1024 tok):   470 ms  (TTFT floor on an idle GPU)
Cost:                $1.571 per 1M output tokens at $1.00/h
```

Read it: it fits, but only 8 concurrent 4k-token users, and 22 tok/s each. Try `-dtype fp8 -kv-dtype fp8 -gpu l4`. The L4 has half the A10G's bandwidth, but FP8 halves the bytes, and the KV budget grows enough for several times more users.

**Example B: Llama 3.1 70B.** 141 GB in BF16 doesn't fit on any single GPU in the table except the H200 (141 GB), and there it leaves no room for KV cache. Options:

```
$ go run ./cmd/infercalc -model llama-3.1-70b -gpu h100 -gpus 4 -context 4096
Weights (bf16):        141.2 GB ...
KV budget:           142.8 GB
Max concurrent seqs: 106
At batch 106: decode step 28.2 ms, per-user 35.5 tok/s, aggregate 3758 tok/s
```

Or quantize to FP8 (70 GB) and use 2× H100 (TP=2) per replica, giving four replicas per p5.48xlarge. Level 11 adds the AWS instance view.

**Example C: the long-context trap.** The same 8B on an L4, but users paste 32k-token documents: KV per sequence = 4 GiB, so **one** sequence fits. Long context is a capacity problem before it's a speed problem. Solutions: FP8 KV, a bigger GPU, chunked prefill plus a limit on concurrent long requests, or routing long-context traffic to a separate pool.

## 4. Labs

```bash
cd platform
go run ./cmd/infercalc -h                                   # all flags
go run ./cmd/infercalc -model llama-3.1-8b -gpu l4 -dtype fp8 -kv-dtype fp8 -context 8192
go run ./cmd/infercalc -model mixtral-8x7b -gpu h100 -gpus 2 # MoE: memory uses total params, speed uses active
cd ../levels/06-performance-math/labs
python3 roofline.py --gpu h100
python3 roofline.py --gpu l4 --bytes-per-weight 1 --batch 1 16 64 128
```

Assignments:
1. Fill in this table with the calculator, then check it against vLLM's startup log in level 8 ("Available KV cache memory", "Maximum concurrency"):

   | Model | GPU | dtype | Max seqs @ 8k | tok/s/user @ max batch | $/1M tok |
   |---|---|---|---|---|---|
   | 8B | L4 | bf16 | | | |
   | 8B | L4 | fp8 | | | |
   | 8B | L40S | bf16 | | | |
   | 70B | 4×L40S | fp8 | | | |
   | 70B | 8×H100 | bf16 | | | |

2. Find the batch size where an L4's decode stops being memory-bound with FP8 weights (`roofline.py`). Is it reachable given its KV budget?
3. Add a `-gpu` preset for a GPU you have access to, and a `-model` preset for a model you care about (`platform/internal/calc/calc.go`). Add a test with your hand calculation.

## 5. Check yourself

<details><summary>1. Why does aggregate throughput rise with batch size while per-user speed falls?</summary>
Each decode step reads the weights once for the whole batch, so more sequences share that fixed cost (aggregate rises). Each step also reads every sequence's KV cache and does more compute, so it takes longer (per-user speed falls).</details>

<details><summary>2. Estimate the single-stream decode speed of a 16 GB model on an H100 (3.35 TB/s, MBU 0.75).</summary>
16e9 / (3.35e12 × 0.75) ≈ 6.4 ms per token, about 157 tok/s.</details>

<details><summary>3. A colleague proposes an H100 instead of an L40S "because it has 3× the FLOPs" for a batch-4 chat workload. What's the better argument?</summary>
At batch 4, decode is memory-bound, so compare bandwidth (3.35 vs 0.86 TB/s, ~3.9×) and memory size, and then cost per token. FLOPs matter for prefill-heavy or large-batch workloads.</details>

<details><summary>4. Why does FP8 weight quantization speed up decode even though tensor-core FLOPs aren't the bottleneck?</summary>
Decode time is dominated by bytes read per step. FP8 halves the weight bytes.</details>

## 6. Going deeper
- Kipply, *Transformer Inference Arithmetic*
- Williams, Waterman, Patterson, *Roofline: An Insightful Visual Performance Model* (2009)
- Databricks / Baseten / Anyscale posts on LLM inference performance engineering (MBU, MFU)

**Next → [Level 7: Inference optimizations](../07-inference-optimizations/README.md)**
