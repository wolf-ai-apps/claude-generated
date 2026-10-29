# AI Inference Academy — Curriculum Plan

> Learn how a prompt becomes tokens, how those tokens are produced fast and cheaply on GPUs,
> and how to run that as a reliable, observable, autoscaled service. Understand the basics, then build the system.

---

## 1. Who this is for

The academy serves two audiences that share one foundation and then split into tracks:

| | **Engine track (E)** | **Platform track (P)** |
|---|---|---|
| You are | ML / systems engineer who wants to know what happens inside vLLM / SGLang / TensorRT-LLM | Platform / SRE / infra engineer who has to run inference in production |
| You build | A small inference engine from scratch: KV cache, continuous batching, paged memory, kernels, quantization, speculative decoding | An inference platform: OpenAI-compatible gateway, GPU Kubernetes, cache-aware routing, autoscaling, SLO observability, cost model |
| Main languages | Python (NumPy → PyTorch → Triton) | Go (gateway, load generator, operators), YAML/Helm, some Python |
| Ends with | An engine that beats a naive baseline by >10× throughput | A platform that meets a stated SLO under a realistic load test |

Both tracks finish together in a **shared capstone** where the engine and the platform meet.

**Prerequisites**
- Comfortable programming in Python *or* Go (both, ideally).
- Basic linear algebra (matrix multiply, softmax). No ML research background needed.
- Docker basics; Platform track also assumes basic Kubernetes (Pods, Deployments, Services).

**Paths**
- **Engine path:** Foundations → E1–E10 → Capstone (~14 weeks at ~8 h/week)
- **Platform path:** Foundations → P1–P11 → Capstone (~14 weeks)
- **Full-stack path:** Foundations → both tracks interleaved → Capstone (~24 weeks)

---

## 2. Learning outcomes

By the end, a graduate can:

1. **Do the back-of-envelope math.** Given a model, GPU and workload, estimate memory footprint, KV cache capacity, max batch size, decode tokens/s ceiling and cost per million tokens — and be within ~2× of measured reality.
2. **Explain every major serving optimization** (KV cache, continuous batching, PagedAttention, prefix caching, FlashAttention, quantization, speculative decoding, chunked prefill, disaggregation, parallelism) in terms of *which bottleneck it attacks*.
3. **Implement** a minimal version of each of those optimizations and measure its effect.
4. **Operate** LLM inference on Kubernetes: GPU scheduling, cold starts, routing, autoscaling, failure handling.
5. **Define and defend SLOs** with the right metrics (TTFT, TPOT/ITL, goodput), and benchmark correctly (open-loop load, realistic length distributions, no coordinated omission).
6. **Design** inference systems in an interview or architecture review, with explicit trade-offs.

---

## 3. Course map

```
                         ┌──────────────────────────────┐
                         │  FOUNDATIONS (F0–F4, shared) │
                         │  mental model · transformer  │
                         │  sampling · GPU · the math   │
                         └──────────────┬───────────────┘
                 ┌──────────────────────┴───────────────────────┐
     ┌───────────▼────────────┐                     ┌───────────▼────────────┐
     │  ENGINE TRACK (E1–E10) │                     │ PLATFORM TRACK (P1–P11)│
     │  build the engine      │◄──── shared ───────►│ build the platform     │
     │  Python · PyTorch ·    │      metrics &      │ Go · Docker · K8s ·    │
     │  Triton                │      benchmarks     │ Prometheus · Cloudflare│
     └───────────┬────────────┘                     └───────────┬────────────┘
                 └──────────────────────┬───────────────────────┘
                         ┌──────────────▼───────────────┐
                         │  CAPSTONE + SYSTEM DESIGN    │
                         │  MiniServe end-to-end        │
                         └──────────────────────────────┘
```

Every module uses the same template:

- **Why it matters** — the bottleneck or failure it addresses
- **Concepts** — what to understand before writing code
- **Lab** — what you build
- **Done when** — measurable acceptance criteria (a lab is not finished until it is measured)
- **Read** — primary sources (papers, docs, code)

---

## 4. Foundations (shared, ~3 weeks)

### F0 — The inference mental model
- **Why:** Every optimization later is a response to one of two regimes. Without this, the rest is a list of tricks.
- **Concepts:** Training vs inference. Autoregressive generation. **Prefill** (process the whole prompt in parallel, compute-bound) vs **decode** (one token per step, memory-bandwidth-bound). Latency vs throughput. The user-facing metrics: **TTFT**, **TPOT / ITL**, end-to-end latency, tokens/s, requests/s, **goodput** (throughput that meets SLO).
- **Lab:** Call a hosted model and a local model (llama.cpp or Ollama) with streaming; record TTFT and per-token timings; plot them.
- **Done when:** You can explain from your plot why the first token is slow and the rest are steady.

### F1 — The transformer, for inference
- **Concepts:** Tokenization (BPE), embeddings, decoder block (attention + MLP + norm + residual), causal masking, **MHA vs MQA vs GQA**, RoPE, logits. Where the parameters live and where the FLOPs go. Dense vs Mixture-of-Experts.
- **Lab:** Implement a GPT-2-style forward pass in **NumPy only**, load real GPT-2 small weights from `safetensors`, and match Hugging Face logits.
- **Done when:** Max abs diff vs reference logits < 1e-3; you generate coherent text greedily.
- **Read:** *Attention Is All You Need*; Karpathy's nanoGPT / llm.c; Jay Alammar's *Illustrated Transformer*.

### F2 — Decoding and sampling
- **Concepts:** Greedy, temperature, top-k, top-p, min-p, repetition/presence penalties, stop sequences, seeds and determinism (and why batched GPU inference is often *not* bit-deterministic). Logprobs. **Structured output** via constrained decoding (JSON schema → grammar/FSM → token mask).
- **Lab:** Add a sampler module to your F1 model; then add a tiny regex/JSON constrained decoder that masks invalid tokens.
- **Done when:** 100/100 generations parse as valid JSON for a given schema.
- **Read:** Holtzman et al., *The Curious Case of Neural Text Degeneration*; Outlines / XGrammar docs.

### F3 — Hardware for inference
- **Concepts:** GPU anatomy (SMs, tensor cores, HBM, L2, shared memory/SRAM, registers). Memory bandwidth vs FLOPs. **Roofline model and arithmetic intensity.** Kernel launch overhead, CUDA graphs. PCIe vs NVLink vs InfiniBand. Datacenter GPU generations and what changed for inference (FP8, FP4, memory size). CPUs and Apple Silicon for local inference. Brief: TPUs and other accelerators.
- **Lab:** Benchmark a matmul at increasing sizes (PyTorch) and plot achieved TFLOP/s and GB/s against the roofline of your hardware (or a CPU roofline if no GPU).
- **Done when:** You can point at your plot and say where decode and prefill each sit.
- **Read:** Horace He, *Making Deep Learning Go Brrrr From First Principles*; NVIDIA GPU architecture whitepapers.

### F4 — The back-of-envelope toolkit
- **Concepts / formulas** (memorize these):
  - Weight memory ≈ `params × bytes_per_param` (FP16 = 2, FP8/INT8 = 1, INT4 ≈ 0.5)
  - KV cache per token ≈ `2 × layers × kv_heads × head_dim × bytes`
  - Decode tokens/s per sequence ceiling ≈ `memory_bandwidth / bytes_read_per_step`
  - Prefill FLOPs ≈ `2 × params × prompt_tokens`
  - Max concurrent sequences ≈ `(GPU_mem − weights − overhead) / (KV_per_token × avg_context)`
  - Cost per 1M tokens ≈ `GPU_$per_hour / (tokens_per_s × 3600) × 1e6`
- **Lab:** Build a small calculator CLI (Go or Python) that takes a model config + GPU spec + workload and prints the estimates above. It is reused in every later module to compare *predicted vs measured*.
- **Done when:** The calculator's KV cache and weight-memory numbers match what vLLM logs at startup for 3 different models.

---

## 5. Engine track (E1–E10)

You build **`minfer`**, a small but real inference engine, one bottleneck at a time. Each module keeps a benchmark script so progress is always measured against the previous version.

### E1 — Naive engine on PyTorch + GPU
- **Concepts:** Moving F1 to PyTorch; loading Llama-family weights (safetensors, sharded checkpoints, `config.json`); dtype choices (FP32/FP16/BF16).
- **Lab:** Port the NumPy model to PyTorch, support a small Llama-architecture model (RMSNorm, SwiGLU, RoPE, GQA). Recompute the whole sequence each step — on purpose.
- **Done when:** Logits match HF; baseline tokens/s and TTFT recorded. This is the number you will beat by 10×+.

### E2 — KV cache
- **Concepts:** Why attention over past tokens is recomputed; caching K and V; prefill vs decode code paths; memory cost growing with sequence length (use F4).
- **Lab:** Add a contiguous pre-allocated KV cache.
- **Done when:** Decode step time is ~flat w.r.t. sequence length; measured KV memory matches the calculator.

### E3 — Batching and scheduling
- **Concepts:** Static batching → dynamic batching → **continuous (iteration-level) batching**. Padding waste. Request lifecycle (waiting → running → finished/preempted). Scheduler policies (FCFS, max tokens per step, fairness). **Chunked prefill** and why mixing prefill and decode in one step protects ITL.
- **Lab:** Build an async engine loop with a request queue and a continuous-batching scheduler; expose a streaming Python API.
- **Done when:** Under a fixed synthetic load, throughput improves ≥5× over static batching and you can show the ITL spike that chunked prefill removes.
- **Read:** Orca (OSDI '22); Sarathi-Serve (OSDI '24).

### E4 — Memory management: PagedAttention and prefix caching
- **Concepts:** Fragmentation of contiguous KV allocation; fixed-size **blocks + block tables** (virtual memory for KV); copy-on-write for parallel sampling; preemption by swap vs recompute; **prefix caching** (hash-based blocks, radix tree), eviction (LRU).
- **Lab:** Replace contiguous KV with a block allocator and block tables (a reference attention that gathers by block table is fine — kernels come in E5). Add hash-based prefix caching.
- **Done when:** Max concurrent sequences increases measurably at the same memory; a shared-system-prompt workload shows a large TTFT drop with prefix caching on.
- **Read:** vLLM / PagedAttention (SOSP '23); SGLang / RadixAttention.

### E5 — Kernels: attention and fusion
- **Concepts:** Why naive attention is memory-bound (materializing the N×N matrix); **FlashAttention** (tiling + online softmax); FlashDecoding (split-K for long-context decode); kernel fusion (RMSNorm, RoPE, SwiGLU); CUDA graphs to kill launch overhead in decode; `torch.compile`.
- **Lab:** Write a fused softmax and then a simplified FlashAttention forward kernel in **Triton**; plug a paged-attention decode kernel (yours or FlashInfer) into `minfer`; capture decode with CUDA graphs.
- **Done when:** Your Triton attention matches the reference numerically and beats naive PyTorch attention at long sequence lengths; decode step latency drops with CUDA graphs.
- **Read:** FlashAttention 1/2/3 papers; Triton tutorials; FlashInfer.

### E6 — Quantization
- **Concepts:** What is quantized (weights, activations, KV cache); symmetric/asymmetric, per-tensor/channel/group; weight-only INT8/INT4 (**GPTQ, AWQ**); activation outliers and **SmoothQuant**; FP8 (E4M3/E5M2) and FP4 formats on recent GPUs; GGUF / llama.cpp quant types; accuracy vs speed trade-offs and how to *measure* quality loss.
- **Lab:** Implement per-group INT8 and INT4 weight-only quantization with a dequant-in-matmul path; add FP8 KV cache. Evaluate perplexity and a small task benchmark before/after.
- **Done when:** You publish a table: memory, tokens/s, perplexity Δ for FP16 / INT8 / INT4 / FP8-KV.
- **Read:** LLM.int8(); GPTQ; AWQ; SmoothQuant.

### E7 — Speculative decoding
- **Concepts:** Draft-then-verify; why it is lossless (rejection sampling); acceptance rate and the expected speed-up formula; draft models vs n-gram/prompt lookup vs **Medusa** heads vs **EAGLE**; when speculation hurts (high batch sizes, compute-bound regime).
- **Lab:** Implement draft-model speculative decoding and prompt-lookup decoding in `minfer`.
- **Done when:** You show speed-up vs batch size and find the crossover point where speculation stops paying off.
- **Read:** Leviathan et al. 2023; Chen et al. 2023; Medusa; EAGLE.

### E8 — Parallelism and large models
- **Concepts:** **Tensor parallelism** (column/row split, all-reduce per layer), **pipeline parallelism** (bubbles), **expert parallelism** for MoE (all-to-all, load balance), sequence/context parallelism for long context. NCCL collectives and their cost on PCIe vs NVLink. Picking a layout for a given model and node.
- **Lab:** Add 2-way tensor parallelism to `minfer` with `torch.distributed` (2 GPUs; a CPU/gloo fallback exists for learning). Model the all-reduce cost with the calculator.
- **Done when:** Outputs match single-GPU; you can explain measured scaling efficiency from interconnect bandwidth.
- **Read:** Megatron-LM; DeepSpeed-Inference; MoE serving papers (e.g., DeepSeek-V3 inference notes).

### E9 — Disaggregated serving
- **Concepts:** Prefill and decode interfere; split them onto different workers; KV cache transfer cost (RDMA/NIXL-style transports); when disaggregation wins vs chunked prefill; KV cache tiering to CPU/SSD.
- **Lab:** Run `minfer` as separate prefill and decode processes and ship KV blocks between them over a socket; measure TTFT/ITL vs colocated.
- **Done when:** You produce a graph showing the workload regime where disaggregation improves goodput and where it doesn't.
- **Read:** DistServe (OSDI '24); Splitwise (ISCA '24); Mooncake (FAST '25).

### E10 — Beyond text-only chat (survey + optional lab)
- **Concepts:** Long-context techniques (sliding window, attention sinks), LoRA serving (multi-LoRA batching, S-LoRA/Punica), embedding and reranker models (very different profile: no decode), vision-language models (image encoder prefill), reasoning models (very long outputs → decode-dominated).
- **Optional lab:** Add multi-LoRA batching to `minfer`.

---

## 6. Platform track (P1–P11)

You build **`inferplat`**: a Go gateway and tooling around off-the-shelf engines (vLLM / SGLang first, later the capstone swaps in `minfer`).

### P1 — The serving landscape
- **Concepts:** vLLM, SGLang, TensorRT-LLM, llama.cpp / Ollama, Hugging Face TGI, NVIDIA Triton / Dynamo, llm-d; managed APIs and edge options (e.g., Cloudflare Workers AI). What each optimizes for; build-vs-buy.
- **Lab:** Serve the same model with two engines; run the same benchmark; read their startup logs and `/metrics`.
- **Done when:** A one-page comparison with your own numbers, and you can map each engine feature to an Engine-track module.

### P2 — The API layer (Go)
- **Concepts:** OpenAI-compatible API shape (chat/completions, streaming via **SSE**), request validation, client disconnect → **cancellation propagation** (abort generation to free KV cache), timeouts, backpressure and load shedding, **token-based** rate limits and quotas, multi-tenancy, API keys.
- **Lab:** Write a Go gateway that proxies to one or more engine backends, streams tokens, cancels upstream on disconnect, and enforces per-tenant token budgets.
- **Done when:** Killing a client mid-stream frees the request on the engine within 1 s (verified via engine metrics); gateway overhead adds < 2 ms p99.

### P3 — Containers and model artifacts
- **Concepts:** CUDA base images, driver vs toolkit vs runtime, NVIDIA Container Toolkit; image size; **model weights are not code** — store in object storage, pull or stream at start; safetensors (and why pickle-based checkpoints are a supply-chain risk); **cold start anatomy** (image pull, weight download, load to GPU, CUDA graph capture, warm-up).
- **Lab:** Containerize an engine; measure and break down cold start; reduce it (pre-pulled images, local NVMe cache, parallel download, lazy loading).
- **Done when:** A cold-start waterfall chart before/after with ≥2× improvement.

### P4 — GPUs on Kubernetes
- **Concepts:** NVIDIA GPU Operator and device plugin; node pools, taints/tolerations, node affinity; GPU sharing (**MIG**, time-slicing, MPS) and its isolation trade-offs; **Dynamic Resource Allocation (DRA)**; multi-node serving (LeaderWorkerSet); KServe; probes for slow-starting pods; PodDisruptionBudgets for expensive replicas.
- **Lab:** On a cloud GPU node (or `kind`/`k3s` with a CPU engine like llama.cpp to learn mechanics), deploy vLLM via Helm with correct resources, probes and a PVC/cache for weights.
- **Done when:** Rolling update without dropped requests under load.

### P5 — Routing and load balancing
- **Concepts:** Why round-robin and least-connections fail for LLMs (requests differ 1000× in cost); load signals (queue depth, running requests, **KV cache utilization**); **prefix/KV-cache-aware routing** (consistent hashing on prefix vs approximate cache state); LoRA-aware routing; session affinity; Kubernetes Gateway API **Inference Extension**.
- **Lab:** Add pluggable routing to the Go gateway: round-robin → least-load → prefix-aware. Benchmark with a multi-turn chat workload.
- **Done when:** Prefix-aware routing shows a higher cache hit rate and lower p95 TTFT than round-robin across ≥3 replicas.

### P6 — Autoscaling and capacity
- **Concepts:** Why CPU/GPU utilization is the wrong signal; scaling on queue depth, KV cache pressure or SLO burn; HPA with custom metrics, **KEDA**; scale-to-zero and its cold-start cost; cluster autoscaling / node provisioning for GPUs; capacity planning from the F4 calculator; spot/preemptible GPUs and graceful drain.
- **Lab:** Autoscale vLLM on a queue-depth metric with KEDA; drive a load ramp; tune for stability (no flapping).
- **Done when:** p95 TTFT stays within SLO during a 5× load ramp, and you can show the cost of the extra replicas.

### P7 — Observability for inference
- **Concepts:** Golden signals for LLMs: TTFT, ITL/TPOT, E2E, queue time, tokens in/out, cache hit rate, preemptions, KV utilization, error/abort rates; GPU telemetry via **DCGM exporter** (SM activity, memory, power, Xid errors); Prometheus histograms and correct percentile math; **OpenTelemetry** traces across gateway → engine, GenAI semantic conventions; SLOs, error budgets, burn-rate alerts; cost attribution per tenant.
- **Lab:** Ship metrics + traces from gateway and engine; build a Grafana dashboard and SLO burn-rate alerts.
- **Done when:** From the dashboard alone, you can diagnose three injected faults (overloaded replica, cold-start storm, prefix-cache thrash).

### P8 — Benchmarking and load testing
- **Concepts:** Closed-loop vs **open-loop** (Poisson arrivals) load generation; **coordinated omission**; realistic input/output length distributions (conversation-style datasets); measuring goodput at an SLO; warm-up and steady state; reproducibility (pinned versions, recorded configs).
- **Lab:** Write a Go load generator: open-loop arrivals, configurable length distributions, streaming-aware timings, output as JSON + plots. Use it for every later benchmark.
- **Done when:** Your tool's numbers agree with an established benchmark tool (e.g., vLLM's benchmark scripts) within ~5% on the same run.

### P9 — Networking, edge and hybrid
- **Concepts:** Long-lived streaming connections through proxies/load balancers (idle timeouts, buffering); **Cloudflare** in front of inference (Tunnel, AI Gateway for caching/rate limiting/analytics, Workers AI for edge models); **Tailscale** for private connectivity between a homelab GPU, cloud clusters and developers; hybrid burst-to-cloud.
- **Lab:** Run a model on a local/homelab GPU, join it to a tailnet, register it as a backend in the Go gateway running in the cloud, and expose the gateway through Cloudflare.
- **Done when:** Streaming works end-to-end with no buffering; failover to a cloud backend when the home node goes offline.

### P10 — Reliability, security and cost
- **Concepts:** Failure modes (GPU Xid errors, OOM, NCCL hangs, stuck requests, poison prompts that blow up context); health checks that actually test generation; retries vs idempotency for streaming; multi-region and provider fallbacks; isolation between tenants; model/artifact integrity; **cost per million tokens** and unit economics; reserved vs on-demand vs spot.
- **Lab:** Chaos day: inject failures into your platform and write runbooks; build a cost report from metrics.
- **Done when:** Each injected failure is detected by an alert and has a runbook; you produce cost/1M tokens per model and tenant.

### P11 — Offline and batch inference
- **Concepts:** Throughput-optimized jobs (evals, data labeling, embeddings), batch APIs, scheduling on spare capacity, checkpointing progress, Ray Data / Kubernetes Jobs.
- **Lab:** Run a 100k-prompt batch job on preemptible capacity with resume-on-failure.
- **Done when:** Job completes despite a forced node loss; you report cost vs the online path.

---

## 7. Capstone: MiniServe (shared, ~3 weeks)

Teams (or solo full-stack learners) combine both tracks:

- **Engine:** `minfer` with continuous batching, paged KV cache, prefix caching and one of {quantization, speculative decoding}, exposing a streaming API and Prometheus metrics.
- **Platform:** `inferplat` Go gateway with prefix-aware routing, token quotas and cancellation, deployed on Kubernetes with autoscaling, dashboards and SLO alerts.
- **Proof:** A benchmark report using the P8 load generator against a stated SLO (e.g., p95 TTFT < 500 ms, p95 ITL < 50 ms) showing goodput, cost/1M tokens, predicted-vs-measured from the F4 calculator, and a comparison against vLLM on the same hardware.
- **Defense:** A 20-minute design review — explain every design choice in terms of the bottleneck it targets, and what you'd change at 100× scale.

---

## 8. System design studio (interview-style, throughout)

Weekly case studies, worked on a whiteboard and then critiqued:

1. Design a ChatGPT-style chat service for 10M daily users.
2. Design multi-tenant serving for 500 fine-tuned LoRA adapters on a shared base model.
3. Design a coding assistant with low-latency autocomplete (tight TTFT) and long-context chat.
4. Design a RAG system's inference path: embeddings, reranker, generator — and where each one's bottleneck is.
5. Design an offline pipeline that runs evals over 1B tokens per night at minimum cost.
6. Design a global inference platform with regional data residency and provider failover.
7. Design serving for a large MoE model across multiple nodes.

Each case is graded on: requirements & SLOs, capacity math (F4), component design, the right optimization for the bottleneck, failure modes, observability, and cost.

A companion **interview prep** set covers: concept questions per module, DSA problems that appear in inference systems (LRU/radix trees for prefix caches, priority queues for schedulers, consistent hashing for routing, ring buffers for streaming, token-bucket rate limiters), and behavioral stories drawn from the labs (a debugging story, a trade-off story, a performance-win story with numbers).

---

## 9. Hardware and cost tiers

Every lab states which tier it needs and has a fallback where possible.

| Tier | Hardware | Covers |
|---|---|---|
| **0 — Laptop** | CPU or Apple Silicon, 16 GB RAM | All Foundations, E1–E4 with small models (GPT-2, ~0.5B Llama-family models), P2, P3, P5, P7, P8 using llama.cpp/Ollama or CPU engines, P4 mechanics on kind/k3s |
| **1 — Single GPU** | One 16–24 GB GPU (consumer card or cloud L4/A10-class), rented by the hour | E5–E7, P1, P4 for real, P6, P9 homelab lab, most of the capstone |
| **2 — Multi-GPU** | 2–8 GPUs with NVLink, short rentals | E8, E9, large-model and MoE labs, capstone scale test |

Cost guidance: design labs so Tier 1 work fits in a small number of GPU-hours per module; always shut down on completion (the P10 cost report includes the academy's own lab spend).

---

## 10. Proposed repository layout (for when content is built)

```
ai-inference-academy/
├── README.md
├── CURRICULUM.md                # this file
├── foundations/
│   ├── f0-mental-model/
│   │   ├── README.md            # concept guide
│   │   ├── lab/                 # starter code + TODOs
│   │   ├── solution/            # reference solution
│   │   └── checks/              # automated "Done when" tests
│   └── ...
├── engine/                      # E1–E10, building minfer/
├── platform/                    # P1–P11, building inferplat/
├── tools/
│   ├── calc/                    # F4 back-of-envelope calculator (Go)
│   └── loadgen/                 # P8 load generator (Go)
├── capstone/
├── system-design/
├── interview-prep/
└── infra/                       # Terraform/Helm for lab environments
```

Each module's `checks/` makes "Done when" executable: numerical-correctness tests for engine labs, benchmark threshold tests for performance labs, and `kubectl`/HTTP probes for platform labs.

---

## 11. Build roadmap for the academy itself

| Phase | Scope | Output |
|---|---|---|
| **1 — Spine** | F0–F4, `tools/calc`, `tools/loadgen` | Anyone can start; the measuring tools exist before the things being measured |
| **2 — Core loops** | E1–E4 and P1–P5 | The highest-value content: a working engine with batching + paging, and a gateway on K8s with smart routing |
| **3 — Performance & ops** | E5–E7 and P6–P8 | Kernels, quantization, speculation; autoscaling, observability, benchmarking |
| **4 — Scale** | E8–E10 and P9–P11 | Multi-GPU, disaggregation, hybrid/edge, reliability, batch |
| **5 — Capstone & studio** | Capstone, system design cases, interview prep | Graduation path |

Recommended next step: build **Phase 1** — start with F0 and the F4 calculator, since every later lab compares measured numbers against it.

---

## 12. Core reading list

- Vaswani et al., *Attention Is All You Need* (2017)
- Horace He, *Making Deep Learning Go Brrrr From First Principles*
- Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models* (OSDI '22)
- Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention* (SOSP '23)
- Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs*
- Dao et al., *FlashAttention* 1, 2 and 3
- Agrawal et al., *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve* (OSDI '24)
- Zhong et al., *DistServe* (OSDI '24); Patel et al., *Splitwise* (ISCA '24); Qin et al., *Mooncake* (FAST '25)
- Leviathan et al., *Fast Inference from Transformers via Speculative Decoding* (2023); Medusa; EAGLE
- Dettmers et al., *LLM.int8()*; Frantar et al., *GPTQ*; Lin et al., *AWQ*; Xiao et al., *SmoothQuant*
- Shoeybi et al., *Megatron-LM*
- Source code: vLLM, SGLang, llama.cpp, nanoGPT, FlashInfer
