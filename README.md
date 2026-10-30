# AI Inference Academy

**From "what is a GPU" to serving vLLM on Amazon EKS and SageMaker.** Understand the basics, then build the system.

The course has 16 levels. Each one has concepts, worked examples with real numbers, runnable labs and self-check questions. Most labs run on a laptop with no GPU: a GPU-free mock inference server reproduces the queueing, prefill/decode timing, prefix caching and vLLM metrics that the platform layer cares about. The GPU and AWS labs use the same tools against real vLLM.

**Running example:** a support chatbot on **Llama 3.1 8B**, which grows to 70B and multi-node. Every level asks what happens to this model and this workload at that layer.

## Read the course as a website

```bash
python3 site/build.py                    # rebuilds site/index.html from the level READMEs
python3 -m http.server -d site 8080      # open http://localhost:8080
```

Or just double-click `site/index.html`: it's one self-contained file (it needs internet for fonts and the Markdown renderer). Progress ("Mark level as done") is saved in your browser.

## Course map

| # | Level | You'll be able to | Labs | Hardware |
|---|---|---|---|---|
| 0 | [Why inference matters](levels/00-why-inference/README.md) | explain prefill vs decode, TTFT/ITL/throughput/cost | stream timing | laptop |
| 1 | [Computer basics](levels/01-computer-basics/README.md) | tell compute-bound from memory-bound; use Little's Law | SIMD vs loop; measure GFLOP/s vs GB/s | laptop |
| 2 | [What a GPU is](levels/02-what-is-a-gpu/README.md) | read GPU specs for inference; write CUDA/Triton kernels | CUDA + Triton vector add, fusion, bandwidth | any GPU |
| 3 | [GPU software stack](levels/03-gpu-software-stack/README.md) | debug driver/runtime mismatches; profile generation | stack checker, PyTorch profiler | any GPU |
| 4 | [Models from first principles](levels/04-models-first-principles/README.md) | read a config.json; count params; explain GQA, MoE | BPE tokenizer, param counter, attention | laptop |
| 5 | [How generation works](levels/05-how-generation-works/README.md) | implement the KV cache, samplers, JSON-constrained decoding | NumPy Llama with exact KV cache | laptop |
| 6 | [Performance math](levels/06-performance-math/README.md) | estimate memory, concurrency, speed and $/token within 2× | `infercalc` (Go), roofline | laptop |
| 7 | [Inference optimizations](levels/07-inference-optimizations/README.md) | map 10 optimizations to the bottleneck each removes | batching simulator, paged KV + prefix cache, INT8/INT4, speculative decoding | laptop |
| 8 | [Running models and vLLM](levels/08-running-models-and-vllm/README.md) | run and tune vLLM; read its log and metrics | serve, OpenAI client, offline batch, metrics | 24 GB GPU |
| 9 | [Benchmarking and SLOs](levels/09-benchmarking-and-slos/README.md) | measure goodput at an SLO without lying to yourself | `loadgen` (Go), capacity sweep | laptop / GPU |
| 10 | [Containers and Kubernetes](levels/10-containers-and-kubernetes/README.md) | schedule GPUs, survive cold starts, route and autoscale LLM traffic | `gateway` (Go), Compose fleet, kind, KEDA, GPU manifests | laptop / GPU cluster |
| 11 | [AWS foundations](levels/11-aws-foundations/README.md) | pick instances, get quota and capacity, store weights, set up IAM | instance picker, S3 weights | AWS account |
| 12 | [vLLM on EKS](levels/12-vllm-on-eks/README.md) | run a production-shaped LLM service on EKS end to end | Auto Mode cluster, NodePool, vLLM, gateway, ALB, KEDA, dashboards, alerts, multi-node | AWS GPUs |
| 13 | [SageMaker](levels/13-sagemaker/README.md) | deploy, stream, autoscale and pack models on SageMaker | BYOC vLLM image, boto3 deploy/invoke/autoscale, inference components | AWS GPUs |
| 14 | [Production operations](levels/14-production-operations/README.md) | choose platforms with numbers; ship safely; handle failures | cost model, chaos drill | laptop |
| 15 | [Capstone and system design](levels/15-capstone-and-system-design/README.md) | build, measure and defend a system; design 7 more | capstone brief + rubric, design cases, interview prep | AWS GPUs |

Suggested paths (detail in [CURRICULUM.md](CURRICULUM.md)):
- **Full path:** 0 → 15 in order (~10–12 weeks at ~8 h/week).
- **Platform engineer, fast track:** 0, 1, 5 (KV cache), 6, 7 (skim), 8–15.
- **ML / engine engineer:** 0–9 in depth, then 10 and 12, plus the level 7 exercises.

## Tools you build and use (`platform/`, Go, standard library only)

| Tool | What it does | First used |
|---|---|---|
| `cmd/infercalc` | back-of-envelope sizing: weights, KV cache, concurrency, decode speed, prefill time, $/1M tokens | level 6 |
| `cmd/mockllm` | GPU-free OpenAI-compatible server: FCFS queueing, KV budget, prefill/decode timing, block-hash prefix cache, cancellation, **vLLM metric names** | level 0 |
| `cmd/loadgen` | streaming load generator: open-loop Poisson or closed loop, length distributions, shared prefixes, TTFT/TPOT/ITL percentiles, SLO goodput, truncated-stream detection | level 9 |
| `cmd/gateway` | LLM gateway: tenant API keys + token-bucket budgets, round-robin / least-loaded / **prefix-aware (rendezvous hashing + bounded load)** routing, Kubernetes DNS discovery, health checks, pre-first-byte retry, cancellation propagation, Prometheus metrics | level 10 |

```bash
cd platform && go build -o bin/ ./cmd/...      # Go 1.24+
```

## Quick start (5 minutes, no GPU)

```bash
cd platform && go build -o bin/ ./cmd/...
for i in 1 2 3; do bin/mockllm -addr :800$i -name replica-$i -max-num-seqs 16 -cache-blocks 1000 & done
bin/gateway -addr :9000 -policy prefix -backends http://localhost:8001,http://localhost:8002,http://localhost:8003 &
bin/loadgen -url http://localhost:9000 -rate 20 -n 300 -prompt-min 50 -prompt-max 200 -output-min 32 -output-max 64 \
  -prefix-tokens 2000 -prefix-groups 12 -slo-ttft 300ms -slo-tpot 40ms
bin/infercalc -model llama-3.1-8b -gpu l4 -dtype fp8 -kv-dtype fp8 -context 8192
```

## Requirements

| For | Needs |
|---|---|
| Levels 0–1, 4–7, 9, 14 and all Go tools | Python 3.10+ with NumPy, Go 1.24+ |
| Levels 2–3, 8 | an NVIDIA GPU (24 GB for the 8B model; any GPU for the 0.5B model), Docker with the NVIDIA Container Toolkit, PyTorch / vLLM |
| Level 10 | Docker, kind, kubectl, Helm |
| Levels 11–13, 15 | an AWS account with G-instance quota, aws CLI v2, eksctl, kubectl, Helm; a Hugging Face token for Llama (or use Qwen2.5, which is ungated) |

## Verify everything

```bash
scripts/test.sh     # go vet + race tests for the tools, and unit tests for every Python lab (no GPU needed)
```

## Conventions
- **Versions are pinned** (e.g. `vllm/vllm-openai:v0.11.0`). Bump them deliberately, and check flags with `vllm serve --help` for your version.
- **Prices are inputs, never constants.** Look up current EC2/SageMaker prices for your region.
- **Every AWS lab ends with cleanup.** GPU instances bill by the second.
