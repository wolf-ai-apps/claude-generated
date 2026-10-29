# Level 11: AWS foundations for GPU inference

> On AWS, the first obstacles aren't technical: GPU quota, GPU capacity, and choosing the right instance. Then come storage, networking and identity.

**You will:** pick GPU instances from first principles, get quota and capacity, store weights for fast starts, and set up networking and IAM the way production systems do.
**Time:** 3–4 hours · **Hardware:** an AWS account (no GPU spend needed for most of this level)

---

## 1. Concepts

### 1.1 GPU instance families

| Family | GPU | GPUs/instance | GPU memory | GPU interconnect | Good for |
|---|---|---|---|---|---|
| g4dn | T4 16 GB | 1–8 | 16 GB each | PCIe | small/legacy models, cheap experiments |
| **g5** | A10G 24 GB | 1, 4, 8 | 24 GB each | PCIe | 7–8B BF16; widely available |
| **g6** | L4 24 GB | 1, 4, 8 | 24 GB each | PCIe | 7–8B, **FP8**, cost-efficient |
| **g6e** | L40S 48 GB | 1, 4, 8 | 48 GB each | PCIe | 8–34B, 70B FP8 on 4 GPUs, long context |
| p4d / p4de | A100 40 / 80 GB | 8 | 320 / 640 GB | NVLink + EFA | 70B BF16 with TP |
| **p5** | H100 80 GB | 8 | 640 GB | NVLink + EFA | large models, high throughput |
| p5e / p5en | H200 141 GB | 8 | 1,128 GB | NVLink + EFA | 70B on fewer GPUs, 405B, long context |
| p6-b200 | B200 | 8 | ~1.4 TB | NVLink + EFA | frontier scale |
| **inf2** | AWS Inferentia2 | 1–12 chips | 32 GB per chip | NeuronLink | cost-efficient serving via the Neuron SDK |

Notes:
- Within one family, the instance size mostly changes vCPU, RAM and GPU count (e.g. g6.xlarge vs g6.2xlarge are both 1 L4). Give vLLM enough CPU for tokenization and scheduling: 4–8 vCPUs per GPU is a sensible floor.
- **SageMaker** uses the same hardware with an `ml.` prefix (`ml.g6.2xlarge`) and a different, usually higher, price.
- **Inferentia/Trainium** need models compiled with the AWS Neuron SDK. vLLM has a Neuron backend, and AWS provides Neuron containers. They're worth evaluating for steady, high-volume workloads on supported models.
- **Prices change and vary by region.** Get current numbers from the EC2 and SageMaker pricing pages and pass them to the course tools. Don't hard-code them.

### 1.2 Quotas: the day-1 blocker

New accounts often have **zero** GPU quota. EC2 quotas are measured in **vCPUs per instance-family group**:

| Quota (Service Quotas → EC2) | Covers | Code |
|---|---|---|
| Running On-Demand G and VT instances | g4dn, g5, g6, g6e | `L-DB2E81BA` |
| Running On-Demand P instances | p4d, p5, p5e ... | `L-417A185B` |
| All G and VT Spot Instance Requests / All P Spot Instance Requests | Spot | (separate codes) |

One `g6.2xlarge` = 8 vCPUs; one `p5.48xlarge` = 192 vCPUs. **Request quota days before you need it.** SageMaker has its **own** per-instance-type quotas (e.g. "ml.g6.2xlarge for endpoint usage").

```bash
aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA --query Quota.Value
```

### 1.3 Capacity: quota is not capacity

Quota is permission. Capacity is whether AWS has the GPUs in that AZ right now. For scarce GPUs (H100/H200):

| Option | What | Use |
|---|---|---|
| On-Demand | pay per second, no commitment | g-family usually fine; p-family can be unavailable |
| **On-Demand Capacity Reservations (ODCR)** | reserve capacity in an AZ, pay whether used or not | production baseline |
| **EC2 Capacity Blocks for ML** | book GPU clusters for a fixed window (days to weeks) | load tests, launches, multi-node experiments |
| Savings Plans / Reserved | price discount for commitment | steady baseline (discount, not a capacity guarantee) |
| Spot | spare capacity, deep discount, 2-minute interruption notice | stateless replicas behind a router with N+1 headroom; batch jobs |

Design for **multiple instance types and AZs**: let Karpenter pick from `g6`/`g6e`/`g5`. That flexibility is what makes Spot and scarce on-demand usable.

### 1.4 Storage for model weights

| Option | Read throughput (order of magnitude) | Notes |
|---|---|---|
| S3 (in-region, parallel ranged GETs) | GB/s with enough parallelism | the source of truth; versioned; use `aws s3 sync`, s5cmd, or model streamers |
| Mountpoint for Amazon S3 (CSI driver) | high for sequential reads | mount a bucket as a read-only filesystem |
| Instance store NVMe (g5/g6/g6e/p) | several GB/s | fastest local cache; erased when the instance stops |
| EBS gp3 | up to 1 GB/s with provisioned throughput | root volume; baking weights into a snapshot works |
| FSx for Lustre (S3-linked) | very high, shared | many nodes reading the same large models |

Rule: keep weights in **S3 in the same region as the GPUs**, and cache them on local NVMe or a shared filesystem. Never make production scale-out depend on the HF Hub.

### 1.5 Networking

- Run inference in **private subnets** and expose it through an **internal** load balancer behind your API layer.
- **VPC endpoints** (S3 gateway endpoint; interface endpoints for ECR, STS, CloudWatch) keep image pulls and weight downloads off NAT gateways. That's faster, and NAT data charges on 100 GB pulls add up.
- **Load balancer idle timeout:** the ALB default is 60 s. Streaming keeps the connection busy, but a slow first token or a non-streaming long generation can hit it. Raise it (level 12).
- **EFA** (Elastic Fabric Adapter): OS-bypass networking for NCCL across nodes on p4d/p5. You need it for multi-node tensor/pipeline parallelism, and you need placement groups (or Capacity Blocks) to keep nodes close.

### 1.6 Identity: no keys in pods

- **EKS Pod Identity** (or IRSA) gives a Kubernetes service account an IAM role. Pods get short-lived credentials automatically, so you never put access keys in a Secret.
- Minimum policies: read-only S3 on the model bucket, ECR pull, CloudWatch logs.
- Keep the Hugging Face token in **AWS Secrets Manager** (synced via External Secrets / the Secrets Store CSI driver), or avoid it entirely by mirroring weights to S3.

---

## 2. Worked example: choosing instances for the course's running example

```bash
python3 levels/11-aws-foundations/labs/pick_instance.py --params 8 --context 8192 --concurrency 32
python3 levels/11-aws-foundations/labs/pick_instance.py --params 70 --dtype fp8 --context 8192 --concurrency 64
```

For **Llama 3.1 8B, BF16, 8k context, 32 concurrent users per replica**: a single L4 or A10G fits the weights but holds only a few 8k-token sequences. Options: (a) FP8 weights + FP8 KV on g6; (b) g6e (48 GB) BF16; (c) shorter `max-model-len`. For **70B FP8, 64 concurrent users**: 4× L40S (g6e.12xlarge) fits but with little KV room. H100/H200 (p5/p5e) hold far more sequences per replica, so price per *sequence-hour* decides. Pass `--prices-file` with current prices to rank options by cost.

## 3. Labs

1. **Quota:** check your G and P quotas in two regions with the CLI command above. File a request for 16 G-family vCPUs in your course region.
2. **Sizing:** run `pick_instance.py` for 8B, 70B and 405B (FP8). For each, write one sentence explaining the winning shape and its runner-up.
3. **Prices:** create `prices.json` from the EC2 pricing page for your region (e.g. `{"g6.2xlarge": <price>, "g6e.2xlarge": <price>, ...}`) and rerun with `--prices-file prices.json`.
4. **Weights to S3:** mirror a model to your bucket once:
   ```bash
   pip install -U "huggingface_hub[cli]"
   huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir ./llama-3.1-8b
   aws s3 sync ./llama-3.1-8b s3://YOUR-BUCKET/models/Llama-3.1-8B-Instruct/
   ```
   Time a download back onto an EC2 instance in the same region (`time aws s3 sync ...`) and compare it with HF Hub download speed.
5. **IAM:** write the least-privilege IAM policy a vLLM pod needs to read only that prefix of the bucket.

## 4. Check yourself

<details><summary>1. You have 64 vCPUs of G quota. How many g6.2xlarge and how many g6.12xlarge can you run?</summary>
8 × g6.2xlarge (8 vCPU each), or 1 × g6.12xlarge (48 vCPU) plus 2 × g6.2xlarge.</details>

<details><summary>2. Your p5 launch fails with InsufficientInstanceCapacity although quota is fine. What are your options?</summary>
Try other AZs or regions, use a Capacity Block for the time window, buy an ODCR for steady needs, or re-plan onto available shapes (H200/L40S, or quantize to fit smaller GPUs).</details>

<details><summary>3. Why put an S3 gateway endpoint in the VPC of a GPU cluster?</summary>
Weight downloads (tens to hundreds of GB per scale-out) then go directly to S3 instead of through NAT gateways. That's faster and avoids per-GB NAT charges.</details>

## 5. Going deeper
- EC2 accelerated computing instance docs; EC2 Capacity Blocks for ML; On-Demand Capacity Reservations
- EKS best practices guide: AI/ML section (compute, networking, storage)
- AWS Neuron SDK documentation (Inferentia2/Trainium, vLLM on Neuron)

**Next → [Level 12: vLLM on EKS](../12-vllm-on-eks/README.md)**
