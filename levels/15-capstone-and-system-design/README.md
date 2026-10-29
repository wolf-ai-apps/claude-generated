# Level 15: Capstone and system design

> Build it, measure it, defend it. Then design systems ten times bigger on a whiteboard.

**Time:** 1–3 weeks · **Hardware:** Tier 1 for the capstone; none for the design studio

---

## Part A: Capstone, "Support Copilot at scale"

### The brief

A company wants a support assistant for **20 tenants** (business units). Each tenant has its own ~1,500-token system prompt and knowledge snippets. Traffic: **average 3 req/s, peak 10 req/s** during business hours; prompts 500–3,000 tokens; answers 100–400 tokens. SLO: **p95 TTFT ≤ 1.0 s, p95 TPOT ≤ 60 ms, 99% of 5-minute windows**. Model: Llama 3.1 8B Instruct (or a justified alternative). Budget: as low as possible at that SLO.

### Deliverables

1. **Capacity plan (before touching AWS):** `infercalc` and `pick_instance.py` estimates of memory, concurrency per replica, replicas at peak, and $/month. One page.
2. **Running system on EKS** (level 12): vLLM + academy gateway (tenant keys, token budgets, prefix-aware routing) + KEDA + GPU NodePool + dashboards + alerts.
3. **The same model on SageMaker** (level 13) with autoscaling.
4. **Benchmark report** (level 9 template):
   - sweep per replica → max goodput at SLO
   - full-system test at 3 and 10 req/s with a realistic tenant mix (`-prefix-groups 20`)
   - a 3 → 10 req/s ramp: time to recover the SLO
   - routing comparison: round-robin vs prefix-aware (prefix-cache hit rate, TTFT, goodput)
   - one optimization experiment of your choice: FP8 weights / FP8 KV / speculative decoding / chunked-prefill tuning, with a quality check
5. **Chaos results** (level 14): replica loss and GPU node loss at peak, with the runbooks.
6. **Cost comparison:** EKS vs SageMaker vs an API, using your measured goodput (`cost_compare.py`).
7. **Design review (20 minutes):** justify each decision by the bottleneck it addresses, and say what you'd change at 100× traffic.

### Grading rubric

| Area | Excellent looks like |
|---|---|
| Estimation | predictions within ~2× of measurements; gaps explained |
| Correct measurement | open-loop load, realistic distributions, distinct seeds, warm-up excluded, versions recorded |
| SLO met at peak | yes, with headroom stated; N+1 shown by the chaos test |
| Cost | $/1M tokens at SLO for both platforms; break-even vs API |
| Operability | dashboards answer "is it healthy, is it saturated, is it the GPU"; alerts have runbooks |
| Reasoning | every flag and replica count traced to a measurement or a formula |

### Suggested schedule

| Week | Focus |
|---|---|
| 1 | capacity plan; EKS stack up; per-replica sweep; dashboards |
| 2 | full-system tests, routing comparison, autoscaling ramp, optimization experiment |
| 3 | SageMaker deployment, chaos drills, cost comparison, report and review |

---

## Part B: System design studio

Use this structure for every case (it also works as an interview framework):

1. **Requirements:** users, traffic (avg/peak), prompt/output lengths, SLOs, quality bar, tenancy, compliance, budget.
2. **Back-of-envelope:** tokens/s at peak → replicas (goodput per replica) → GPUs → $/month (level 6).
3. **Architecture:** API/gateway, routing, engines, model storage, autoscaling, regions.
4. **The bottleneck:** which of prefill / decode / KV memory / network / capacity dominates, and the optimization that targets it (level 7).
5. **Failure modes and operations:** what breaks, how you know, what happens automatically (level 14).
6. **Evolution:** what changes at 10× and 100×.

### Case 1: Consumer chat assistant, 10M daily active users
*Hints:* convert DAU to peak tokens/s (sessions × turns × tokens, peak factor). Multi-region active-active with capacity reservations. Prefix caching of conversation history needs **session affinity** in routing. Tiered models (small model for easy queries, a router model). Disaggregated prefill/decode at this scale. Graceful degradation (shorter max_tokens, smaller model) under capacity loss.

### Case 2: 500 fine-tuned LoRA adapters on one base model
*Hints:* multi-LoRA serving (`--enable-lora`, `--max-loras`, `--max-lora-rank`). Adapters load into GPU memory, so adapter-aware routing keeps each adapter on a few replicas. Cold adapters load from S3 on demand (latency for the first request). Per-tenant quotas. Compare with merging popular adapters into dedicated deployments.

### Case 3: Code completion (p95 end-to-end < 300 ms)
*Hints:* small models (1–7B), short prompts with heavy prefix reuse (the open file), aggressive cancellation (most requests are superseded by the next keystroke, so cancellation must free the GPU immediately), speculative decoding / prompt-lookup decoding, regional deployment close to users, reject or truncate long contexts.

### Case 4: RAG over a 50M-document corpus
*Hints:* three different inference workloads: an **embedding** model (batch, throughput), a **reranker** (prefill-only, short outputs), and the **generator** (long prompts, prefill-heavy). Each needs different instance shapes and batching. Long retrieved contexts favor prefix caching of shared documents, chunked prefill and FP8 KV. Measure retrieval and generation latency separately.

### Case 5: Nightly evaluation of 1B tokens at minimum cost
*Hints:* offline engine API, Spot capacity with checkpointing, length bucketing, maximum `max-num-seqs`, the cheapest $/token GPU (not the fastest), a provider batch API as a baseline, a deadline-aware scheduler that falls back to on-demand if Spot is scarce.

### Case 6: Global platform with data residency
*Hints:* per-region deployments; routing by tenant residency, not latency; model weights replicated per region; per-region capacity planning and reservations; cross-region failover only where residency allows; central control plane, regional data planes.

### Case 7: Serving a large MoE model (hundreds of billions of total parameters)
*Hints:* memory set by total params, compute by active params. Expert parallelism + tensor parallelism; multi-node with EFA; all-to-all communication cost; load imbalance across experts; FP8 weights; disaggregated prefill/decode; LeaderWorkerSet for gang scheduling; the blast radius of one node failure (the whole group restarts).

---

## Part C: Interview preparation

### Concept questions (answer each in 2 minutes, with numbers)
1. Walk through what happens to a chat request from HTTP to the last token.
2. Why is decode memory-bound? Estimate single-stream tokens/s for an 8B model on an H100.
3. How big is the KV cache for 70B at 32k context? What do you do about it?
4. Explain PagedAttention to a backend engineer using virtual memory.
5. Continuous batching vs static batching: what improves, and what gets worse?
6. When does speculative decoding hurt?
7. TP vs PP vs replicas: pick for 70B on 8× H100 and justify.
8. Which metric do you autoscale vLLM on, and why not GPU utilization?
9. Why does round-robin load balancing hurt LLM serving?
10. EKS vs SageMaker vs Bedrock for a startup with one model and 1M requests/day?

### DSA problems that come from inference systems
| Problem | Where it appears |
|---|---|
| LRU cache with O(1) operations | prefix-cache block eviction (`paged_kv.py`, `mockllm`) |
| Trie / radix tree insert and longest-prefix match | prefix caching (SGLang RadixAttention), constrained decoding |
| Priority queue scheduling with preemption | request scheduler (priorities, fairness) |
| Consistent / rendezvous hashing with bounded load | prefix-aware routing (`gateway/router.go`) |
| Token bucket / sliding-window rate limiter | tenant token budgets (`gateway/router.go`) |
| Free-list allocator with reference counts | KV block manager, copy-on-write (`paged_kv.py`) |
| Streaming percentiles (t-digest, histograms) | latency metrics at scale |
| Bin packing | placing model copies on GPUs (SageMaker inference components, MIG) |

### Behavioral stories to prepare from this course
- **Debugging:** the startup OOM (`max-model-len`), the 60 s ALB timeout, the driver/runtime mismatch.
- **Trade-off:** chunked prefill (ITL vs TTFT), autoscaling headroom vs cost, EKS vs SageMaker.
- **Impact with numbers:** "prefix-aware routing raised goodput 44% on the same hardware", "FP8 weights + KV roughly tripled concurrency per GPU at an eval cost we measured and accepted".
- **Failure and learning:** the coordinated-omission benchmark that looked great and lied.

**Congratulations!** Go back to [the course index](../../README.md) and pick a stretch goal: contribute a fix to vLLM, add engine-metric-aware routing to the gateway, or run the capstone on Inferentia2.
