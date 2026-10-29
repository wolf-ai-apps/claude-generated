# Level 3: The GPU software stack

> Most "GPU doesn't work" incidents are version mismatches between layers. Learn the layers once and you can debug them for good.

**You will:** learn the layers from kernel driver to PyTorch, the compatibility rules between them, and how to profile where inference time goes.
**Time:** 2–3 hours · **Hardware:** a GPU machine for the labs (Tier 1)

---

## 1. Concepts

### 1.1 The layer cake

```
 your code:      vLLM / SGLang / TensorRT-LLM / transformers
 framework:      PyTorch (+ Triton compiler)            ┐
 libraries:      cuBLAS (matmul), cuDNN, NCCL (multi-GPU)│ shipped INSIDE the container
                 FlashAttention/FlashInfer kernels       │ or the pip wheel
 CUDA runtime:   libcudart (version X.Y)                 ┘
 ───────────────────────── container boundary ──────────────────────────
 container hook: NVIDIA Container Toolkit (exposes /dev/nvidia*, mounts driver libs)
 user-mode drv:  libcuda.so (comes with the driver)      ┐
 kernel driver:  nvidia.ko                               │ installed on the HOST
 hardware:       the GPU                                 ┘
```

| Layer | Where it lives | Who installs it | Example version |
|---|---|---|---|
| Kernel driver + `libcuda` | host OS | AMI / GPU Operator / node image | 550.x, 570.x |
| NVIDIA Container Toolkit | host | AMI / GPU Operator | 1.17.x |
| CUDA runtime, cuBLAS, cuDNN, NCCL | container image / wheel | you (via the base image) | CUDA 12.x |
| PyTorch | container image | you | 2.x+cu12x |
| Inference engine | container image | you | vLLM vX.Y |

### 1.2 The one compatibility rule to remember

> **Host driver version ≥ the minimum required by the CUDA runtime inside the container.**

- New driver + old runtime: works (drivers are backward compatible).
- Old driver + new runtime: `CUDA driver version is insufficient for CUDA runtime version`, or `RuntimeError: No CUDA GPUs are available`.
- The `CUDA Version` shown by `nvidia-smi` is the **maximum** runtime the driver supports.

On EKS, the AMI (or EKS Auto Mode) decides the driver. When you upgrade the vLLM image to one built on a newer CUDA, check the node driver first. This is the most common upgrade failure in GPU clusters.

### 1.3 How PyTorch runs a model on the GPU

1. Python calls `model(x)`. Each layer dispatches **kernels** (matmul via cuBLAS, attention via FlashAttention, elementwise ops via PyTorch or Triton).
2. Kernels go into a **CUDA stream** (a queue) and run asynchronously.
3. The CPU races ahead queuing work, unless something forces a **sync** (`.item()`, `.cpu()`, printing a tensor).
4. During decode, each kernel is tiny, so the CPU's Python overhead plus launch latency can become the bottleneck. Engines fix this with **CUDA graphs** (capture once, replay) and by moving scheduling off the critical path.

### 1.4 Multi-GPU: NCCL

**NCCL** implements collectives (all-reduce, all-gather) across GPUs over NVLink, PCIe or the network (EFA on AWS). Tensor parallelism calls all-reduce twice per transformer layer. When multi-GPU inference hangs, it's usually NCCL: mismatched versions, blocked network ports, missing `/dev/shm`, or a missing EFA plugin. `NCCL_DEBUG=INFO` is the first debugging step.

### 1.5 Profiling: where does the time go?

| Tool | What it shows | When |
|---|---|---|
| `nvidia-smi dmon -s pucm` | power, utilization, clocks, memory per second | is anything running at all? |
| DCGM (`dcgmi`, dcgm-exporter) | SM activity, tensor-core activity, memory bandwidth, Xid errors | production fleet metrics (level 12) |
| PyTorch profiler | per-op and per-kernel time, Chrome trace | "which part of my model is slow?" |
| Nsight Systems (`nsys`) | full timeline: CPU threads, kernel launches, gaps, NCCL | "why are there gaps between kernels?" |
| Nsight Compute (`ncu`) | one kernel's achieved bandwidth/FLOPs vs roofline | kernel tuning |

---

## 2. Worked example: debugging "CUDA driver version is insufficient"

Symptom: a new vLLM image crash-loops on a node that ran the old image fine.

```bash
kubectl logs vllm-xyz | head
# RuntimeError: ... CUDA driver version is insufficient for CUDA runtime version
kubectl debug node/ip-10-0-1-23 -it --image=ubuntu -- chroot /host nvidia-smi | head -3
# Driver Version: 535.x   CUDA Version: 12.2
docker run --rm --entrypoint python3 vllm/vllm-openai:<new> -c "import torch; print(torch.version.cuda)"
# 12.8
```

Diagnosis: the new image's runtime (12.8) needs a newer driver than the node's 535 (max CUDA 12.2). Fix: move GPU nodes to an AMI with a newer driver (e.g. the current EKS AL2023 NVIDIA AMI), or pin the image to a build that matches the driver. Prevention: put driver version on the dashboard, and canary image upgrades on one node first.

## 3. Labs

```bash
cd levels/03-gpu-software-stack/labs
./check_stack.sh                          # walk every layer; read the annotations
pip install torch transformers accelerate
python3 profile_generate.py               # profile one generation; open trace.json in ui.perfetto.dev
```

What to find in the trace:
1. **Prefill:** a few long GEMM (matmul) kernels. GPU busy, few gaps.
2. **Decode:** for each token, a repeating pattern of dozens of short kernels with gaps between them. Those gaps are CPU and launch overhead.
3. In the kernel table: the share of time in matmul vs attention vs elementwise ops. For small models at batch 1, "everything else" is surprisingly large.

**Stretch:** run the same generation with `model.generate(..., cache_implementation="static")` plus `torch.compile`, or in vLLM. Compare the decode gaps: CUDA graphs remove most of them.

## 4. Check yourself

<details><summary>1. nvidia-smi shows "CUDA Version: 12.4". Does that mean CUDA 12.4 is installed?</summary>
No. It's the highest CUDA runtime the installed driver supports. The runtime actually used comes from your container or wheel.</details>

<details><summary>2. Which layers belong in the container image, and which on the host?</summary>
Image: CUDA runtime, cuBLAS/cuDNN/NCCL, PyTorch, the engine. Host: kernel driver, libcuda, and the container toolkit that exposes them to containers.</details>

<details><summary>3. Your decode trace shows 60% of wall time as gaps between kernels. What are two fixes?</summary>
CUDA graphs (replay the whole decode step with one launch), and bigger batches or fused kernels so each launch does more work. Removing per-token Python overhead (async scheduling) also helps.</details>

## 5. Going deeper
- NVIDIA CUDA compatibility documentation (driver/runtime compatibility, forward-compatibility packages)
- PyTorch profiler recipes; Nsight Systems user guide
- NCCL environment variables (`NCCL_DEBUG`, `NCCL_SOCKET_IFNAME`) and AWS's EFA + NCCL guides

**Next → [Level 4: Models from first principles](../04-models-first-principles/README.md)**
