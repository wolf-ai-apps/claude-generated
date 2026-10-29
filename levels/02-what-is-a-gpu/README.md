# Level 2: What a GPU is

> A GPU is a large set of simple processors fed by very fast memory. To serve LLMs well, you need to know what's inside it and which spec matters for which workload.

**You will:** learn GPU anatomy, the CUDA programming model, number formats and the GPU generations you'll rent on AWS. Then write and benchmark your first kernels.
**Time:** 3–4 hours · **Hardware:** laptop for reading; any NVIDIA GPU for the labs (Tier 1, or a free Colab/Kaggle GPU)

---

## 1. Concepts

### 1.1 Anatomy of an NVIDIA GPU

```
┌──────────────────────────── GPU die ─────────────────────────────┐
│  ┌─────────┐ ┌─────────┐ ┌─────────┐        ┌─────────┐          │
│  │   SM    │ │   SM    │ │   SM    │  ...   │   SM    │  (H100 SXM: 132 SMs)
│  │ ─────── │ │         │ │         │        │         │          │
│  │ warp    │ │         │ │         │        │         │          │
│  │ schedulers, CUDA cores (FP32/INT), TENSOR CORES (matmul),     │
│  │ registers, shared memory / L1 (~256 KB, on-chip SRAM)         │
│  └─────────┘ └─────────┘ └─────────┘        └─────────┘          │
│  ───────────────────── L2 cache (tens of MB) ─────────────────── │
└───────────────┬──────────────────────────────────┬───────────────┘
                │ memory controllers               │ NVLink / PCIe
        ┌───────┴────────┐                  ┌──────┴───────┐
        │  HBM (80-192GB │                  │ other GPUs / │
        │  at 2-8 TB/s)  │                  │ the CPU      │
        └────────────────┘                  └──────────────┘
```

| Part | What it is | Why inference engineers care |
|---|---|---|
| **SM** (Streaming Multiprocessor) | The GPU's "core"; runs many threads at once | A kernel needs enough parallel work to fill all SMs. Small batches don't. |
| **Warp** | 32 threads executing the same instruction in lockstep | Divergent branches inside a warp serialize, so kernels avoid them |
| **CUDA cores** | Scalar FP32/INT units | Elementwise ops: activations, norms, sampling |
| **Tensor cores** | Units that multiply small matrix tiles in one instruction | Nearly all LLM FLOPs. FP8/FP4 support depends on the generation |
| **Registers / shared memory (SRAM)** | Tiny, extremely fast on-chip memory | FlashAttention's trick is keeping attention tiles here (level 7) |
| **L2 cache** | Shared on-chip cache | Helps reuse across SMs |
| **HBM** (High Bandwidth Memory) | The GPU's main memory | **Capacity** decides if the model and KV cache fit. **Bandwidth** decides decode speed |
| **NVLink / PCIe** | Links to other GPUs and the CPU | Tensor parallelism across GPUs needs NVLink-class bandwidth (level 7) |

### 1.2 The CUDA programming model

You write a **kernel**, a function that runs once per **thread**. Threads are grouped into **blocks** (up to 1,024 threads; a block runs on one SM and can share its shared memory), and blocks form a **grid**.

```c
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;   // which element is mine?
  if (i < n) c[i] = a[i] + b[i];
}
vector_add<<<num_blocks, 256>>>(a, b, c, n);       // launch: grid of blocks of 256 threads
```

Key facts:
- **Launches are asynchronous.** The CPU queues work and moves on; to time GPU work you must synchronize. Every "my GPU is 1000× faster" benchmark that forgets this is wrong.
- **Kernel launch overhead** is several microseconds. LLM decode runs hundreds of small kernels per token, so launch overhead becomes visible. **CUDA graphs** record the sequence once and replay it with one launch (vLLM does this for decode).
- **Memory coalescing:** neighbouring threads should read neighbouring addresses so the hardware can combine their loads.
- **Kernel fusion:** each separate kernel reads its inputs from HBM and writes its outputs back. Fusing `add → relu → scale` into one kernel cuts memory traffic 3×. Triton, `torch.compile` and inference engines fuse aggressively.

**Triton** (used throughout vLLM/SGLang) lets you write kernels in Python at the level of *blocks of data* rather than individual threads. The compiler handles much of the low-level work.

### 1.3 Number formats

| Format | Bytes | Range / precision | Inference use |
|---|---|---|---|
| FP32 | 4 | wide range, ~7 digits | almost never for LLM weights; some accumulations |
| **BF16** | 2 | FP32's range, ~3 digits | default format for weights and activations |
| FP16 | 2 | narrower range, ~3–4 digits | older default; overflow risk |
| **FP8** (E4M3 / E5M2) | 1 | needs per-tensor/channel scales | weights + activations + KV cache on Ada/Hopper/Blackwell |
| INT8 | 1 | integers + scale | weight (and activation) quantization |
| INT4 / FP4 | 0.5 | needs group scales | weight-only quantization (AWQ/GPTQ); FP4 on Blackwell |

Fewer bytes per weight means less memory to read per decode step (faster), more room for KV cache (more users) and faster tensor-core math. The cost is accuracy, which you have to measure (level 7).

### 1.4 The GPUs you'll meet on AWS

Approximate datasheet values. Check the exact SKU, since cloud variants can differ.

| GPU | Generation | Memory | Bandwidth | Dense BF16 | FP8 | AWS instances | Sweet spot |
|---|---|---|---|---|---|---|---|
| T4 | Turing (2018) | 16 GB | 320 GB/s | ~65 TFLOP/s | no | g4dn | tiny models, legacy |
| A10G | Ampere | 24 GB | 600 GB/s | ~70 | no | g5 | 7–8B in BF16, cheap |
| L4 | Ada | 24 GB | 300 GB/s | ~121 | **yes** | g6 | 7–8B (FP8 helps a lot), low power |
| L40S | Ada | 48 GB | 864 GB/s | ~362 | **yes** | g6e | 8–34B, or 70B with 4 GPUs |
| A100 40/80GB | Ampere | 40/80 GB | 1.6–2.0 TB/s | 312 | no | p4d / p4de | 70B with TP, NVLink |
| H100 | Hopper | 80 GB | 3.35 TB/s | ~989 | **yes** | p5 | large models, high throughput |
| H200 | Hopper | 141 GB | 4.8 TB/s | ~989 | **yes** | p5e / p5en | 70B on fewer GPUs, long context |
| B200 | Blackwell | ~180 GB | ~8 TB/s | ~2,250 | yes + FP4 | p6-b200 | frontier scale |

How to read the table for inference:
- **Does it fit?** Memory column (level 6 does the math).
- **How fast is decode?** Bandwidth column. The L4 has *more* FLOPs than the A10G but *half* the bandwidth. At small batch it decodes BF16 models more slowly, while its FP8 support (halving bytes per weight) can recover the difference.
- **How fast is prefill / large-batch work?** The FLOPs column.
- **Can I split a model across GPUs efficiently?** NVLink is only on the p-family here; g-family multi-GPU instances use PCIe.

### 1.5 Multi-GPU interconnects

| Link | Rough bandwidth per GPU | Where |
|---|---|---|
| PCIe Gen4 x16 / Gen5 x16 | ~32 / ~64 GB/s per direction | every GPU, g-family multi-GPU |
| NVLink (A100 / H100) | 600 / 900 GB/s total | p4d, p5 (8 GPUs, all-to-all via NVSwitch) |
| EFA (AWS networking) | 400–3,200 Gbps per instance | p4d/p5 multi-node |

Tensor parallelism does an all-reduce in every layer (level 7), so it wants NVLink. Pipeline parallelism sends much less data between stages and tolerates slower networks.

---

## 2. Worked example: reading `nvidia-smi`

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.xx       Driver Version: 550.xx       CUDA Version: 12.4                 |
| GPU  Name        Persistence-M | Bus-Id   Disp.A | Volatile Uncorr. ECC |               |
| Fan  Temp  Perf  Pwr:Usage/Cap |     Memory-Usage | GPU-Util  Compute M. |               |
|   0  NVIDIA L4             On  | 00000000:31:00.0 Off |               0 |               |
| N/A   54C    P0    68W /  72W  |  21500MiB / 23034MiB |    97%  Default |               |
```

- `CUDA Version: 12.4` is the **highest CUDA runtime this driver supports**, not what's installed (level 3).
- `21500MiB / 23034MiB` used: vLLM pre-allocates ~90% of memory for weights plus KV cache at startup, so this looks "full" even when idle. **Memory used is not a load signal for vLLM.**
- `GPU-Util 97%` means "a kernel was running during the sample window", not "97% of compute was used". A memory-bound decode can show 100% utilization while using 5% of tensor-core FLOPs. Use DCGM metrics such as SM activity and tensor-pipe activity for real utilization (level 12).

## 3. Labs

On a machine with an NVIDIA GPU (a g6.xlarge, Colab, or your own):

```bash
cd levels/02-what-is-a-gpu/labs
python3 gpu_info.py                                   # memory, SMs, compute capability, measured bandwidth
nvcc -O2 -o vector_add vector_add.cu && ./vector_add  # CUDA C kernel; compare copy vs kernel time
pip install torch triton && python3 vector_add_triton.py   # Triton kernel + kernel fusion speed-up
```

What to observe:
1. `gpu_info.py`: the measured copy bandwidth should be about 70–90% of the datasheet number. That fraction (**MBU**, memory bandwidth utilization) is what you'll plug into level 6's calculator.
2. `vector_add`: the kernel reaches hundreds of GB/s, while the host↔device copies run at PCIe speed and take far longer.
3. `vector_add_triton.py`: the fused add+relu+scale kernel is about 2–3× faster than three separate PyTorch ops, purely from less memory traffic.

## 4. Check yourself

<details><summary>1. Which spec decides whether Llama 3.1 70B in BF16 fits on one GPU, and which decides how fast one user's tokens stream?</summary>
Memory capacity (~141 GB of weights needs ≥ 2× 80 GB, or one H200 with little room left), and memory bandwidth (each token reads all weights).</details>

<details><summary>2. Why does nvidia-smi show a vLLM server at 95% memory even with zero traffic?</summary>
vLLM pre-allocates a fixed fraction of GPU memory (`--gpu-memory-utilization`, default 0.9) for weights plus the KV cache block pool at startup.</details>

<details><summary>3. What is kernel fusion and why does it speed up memory-bound work?</summary>
It combines several operations into one kernel so intermediate results stay in registers/SRAM instead of being written to and re-read from HBM. Memory-bound kernels are limited by those HBM round trips.</details>

<details><summary>4. Why is "GPU-Util 100%" misleading?</summary>
It only measures whether any kernel was running during the sample period, not how many SMs or tensor cores were busy.</details>

## 5. Going deeper
- NVIDIA H100 / L4 / L40S datasheets and the Hopper architecture whitepaper
- *Programming Massively Parallel Processors* (Hwu, Kirk, El Hajj)
- Triton tutorials: vector add → fused softmax → matmul
- Horace He, *Making Deep Learning Go Brrrr From First Principles*

**Next → [Level 3: The GPU software stack](../03-gpu-software-stack/README.md)**
