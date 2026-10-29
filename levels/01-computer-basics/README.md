# Level 1: Computer basics a GPU builds on

> Every inference optimization is about one of two limits: how fast you can compute, and how fast you can move data. Learn to tell them apart.

**You will:** understand why parallel hardware exists, the memory hierarchy, and the compute-bound vs memory-bound distinction, and measure both on your own machine.
**Time:** 2–3 hours · **Hardware:** laptop (Tier 0)

---

## 1. Concepts

### 1.1 One fast worker vs thousands of simple ones

A **CPU** has a few (4–128) powerful cores. Each one is built to run *one* stream of instructions as fast as possible: branch prediction, out-of-order execution, big caches. It's great for complicated logic with lots of `if` statements.

A **GPU** has thousands of simple cores grouped into units that run *the same instruction on different data* at the same time. It's terrible at branchy logic, and extraordinary at "do this math on these 10 million numbers".

**Analogy:** a CPU is a team of 16 senior engineers, each able to solve any problem quickly. A GPU is a stadium of 10,000 people with calculators who can each add two numbers. To add 10 million pairs of numbers, the stadium wins by a mile. To debug a race condition, you want the engineers.

Neural network inference is mostly **matrix multiplication**: millions of independent multiply-adds. It's the perfect stadium workload.

### 1.2 SIMD: parallelism inside one core

Even CPUs do some data parallelism. **SIMD** (Single Instruction, Multiple Data) instructions such as AVX-512 add 16 floats in one instruction. NumPy uses them, which is why `a + b` on arrays is hundreds of times faster than a Python loop. A GPU applies the same idea at a much larger scale: one instruction drives 32 threads at once (a **warp**), and thousands of threads are active.

### 1.3 The memory hierarchy

Data has to be close to the compute units to be used. The closer it is, the smaller and faster the storage:

| Level | Typical size | Typical bandwidth | Latency | GPU equivalent |
|---|---|---|---|---|
| Registers | KB per core | ~TB/s per core | ~1 cycle | registers (per thread) |
| L1 / shared memory (SRAM) | 32–256 KB per core/SM | many TB/s aggregate | ~few ns | shared memory / L1 per SM |
| L2 cache | MBs | TB/s | ~10s of ns | L2 (e.g. 50 MB on H100) |
| Main memory | GBs | CPU DDR5: ~50–500 GB/s | ~100 ns | **HBM: 1–8 TB/s**, 16–192 GB |
| Interconnect | — | PCIe Gen4 x16: ~32 GB/s per direction; NVLink: hundreds of GB/s | µs | GPU↔CPU, GPU↔GPU |
| Disk / network | TBs | NVMe: ~2–7 GB/s; network: 10–400 Gb/s | µs–ms | loading model weights |

Two lessons that come back in every level:
1. **Moving data is often more expensive than computing on it.** Copying a tensor from CPU to GPU over PCIe can take longer than the math done on it.
2. **Model weights live in GPU main memory (HBM).** Every decode step has to stream all of them through the compute units. HBM bandwidth is therefore the single most important number for LLM decode speed.

### 1.4 Compute-bound vs memory-bound

Every computation has an **arithmetic intensity**:

```
arithmetic intensity = FLOPs performed / bytes moved to and from memory
```

- **Elementwise add** `c = a + b` (FP32): 1 FLOP per 12 bytes (read 4 + 4, write 4). That's ~0.08 FLOPs/byte. The hardware waits on memory and the math units sit idle: **memory-bound**.
- **Matrix multiply** of two n×n matrices: 2n³ FLOPs over ~12n² bytes, so the intensity grows with n. For big n there's plenty of math per byte: **compute-bound**.

A machine has a peak compute rate (FLOP/s) and a peak bandwidth (bytes/s). Their ratio is the intensity at which a kernel switches from memory-bound to compute-bound. Level 6 turns this into the **roofline model** and applies it to LLMs. For now, remember:

> **LLM decode behaves like the elementwise add (memory-bound). LLM prefill behaves like a big matmul (compute-bound).**

### 1.5 Latency vs throughput

- **Latency:** how long one task takes.
- **Throughput:** how many tasks finish per second.

A GPU gives up per-thread latency for massive throughput. An inference server makes the same trade: processing 64 users together makes each user's tokens a little slower but the total much higher. **Little's Law** connects them, and you'll use it in levels 9 and 12:

```
concurrency (requests in system) = throughput (requests/s) × latency (s)
```

*Example:* a replica that holds 16 concurrent sequences, with each request taking 4 s end to end, can sustain at most 16 / 4 = **4 requests/s**. Send 6 req/s and the queue grows without bound.

---

## 2. Worked examples

**Example A: copy cost vs compute cost.** Adding two 256 MiB arrays on a GPU with 2 TB/s HBM moves 768 MiB, which takes ~0.4 ms. Copying the two inputs from the CPU over PCIe Gen4 (~25 GB/s in practice) moves 512 MiB, which takes ~21 ms. The copy costs **50× more than the computation**. That's why inference engines load weights onto the GPU once and keep everything there.

**Example B: why decode is memory-bound.** A Llama 3.1 8B model in BF16 is ~16 GB of weights. To generate one token for one user, the GPU reads all 16 GB and does about 2 FLOPs per weight, so ~16 GFLOPs over 16 GB, about **1 FLOP per byte**. An H100 can do ~989 TFLOP/s but read only ~3.35 TB/s, so it would need ~295 FLOPs per byte to keep its math units busy. At 1 FLOP/byte, more than 99% of its compute sits idle.

## 3. Labs

```bash
cd levels/01-computer-basics/labs
python3 loop_vs_vector.py      # Python loop vs NumPy SIMD (vs GPU if torch+CUDA present)
python3 memory_vs_compute.py   # measure GFLOP/s and GB/s for add vs matmul on YOUR machine
```

Sample output from a cloud VM (yours will differ):

```
kernel                   GFLOP/s      GB/s  FLOPs/byte
add 20M                      0.9      10.6       0.083
matmul 64x64                51.4      4.81        10.7
matmul 1024x1024           362.2      2.12       170.7
matmul 2048x2048           368.8      1.08       341.3
```

What to observe:
- The add reaches a decent **GB/s** but a tiny **GFLOP/s**: it's limited by memory.
- Matmul GFLOP/s climbs with size, then plateaus at the machine's compute peak: it's limited by compute.
- The plateau is 400× higher than the add's GFLOP/s, on the *same machine*. The only difference is the data reuse per byte.

**Stretch:** run `loop_vs_vector.py` on a machine with a GPU. You'll see the GPU add beat NumPy, *and* the PCIe copy dwarf the kernel time.

## 4. Check yourself

<details><summary>1. Why are GPUs good at neural networks but bad at running a web server?</summary>
Neural networks are huge numbers of identical, independent multiply-adds: data parallelism. Web servers are branchy, latency-sensitive logic with little data parallelism, which is exactly what CPU cores are optimized for.</details>

<details><summary>2. Doubling a GPU's FLOPs without changing its memory bandwidth: does single-user LLM decode get faster?</summary>
Barely. Single-user decode is memory-bound (~1 FLOP/byte), so it's limited by how fast the weights can be read, not by math. Prefill and large-batch decode would benefit.</details>

<details><summary>3. A replica serves at most 32 concurrent requests; each takes 8 s. What's its max sustainable request rate?</summary>
Little's Law: 32 / 8 = 4 requests/s. Beyond that, the queue (and TTFT) grows without bound.</details>

## 5. Going deeper
- *Computer Architecture: A Quantitative Approach* (Hennessy & Patterson), chapters on memory hierarchy and data-level parallelism
- "Latency Numbers Every Programmer Should Know", then think about where HBM and NVLink fit in it

**Next → [Level 2: What a GPU is](../02-what-is-a-gpu/README.md)**
