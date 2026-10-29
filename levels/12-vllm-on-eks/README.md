# Level 12: vLLM on EKS, end to end

> Everything so far comes together here: a production-shaped LLM service on Amazon EKS with GPU nodes that appear on demand, vLLM, a gateway with tenant budgets and prefix-aware routing, an internal ALB, autoscaling on the right signal, and SLO dashboards and alerts.

**You will:** build the cluster, run Llama 3.1 8B on L4 GPUs, put the academy gateway in front, autoscale pods *and* GPU nodes, observe everything, load-test to your SLO, then scale up to 70B and multi-node.
**Time:** 1–2 days · **Hardware:** AWS (Tier 1: g6 instances; Tier 2 for the 70B/multi-node sections) · **Cost guardrail:** run `./labs/cleanup.sh` at the end of every session.

---

## 1. Architecture

```
                         VPC (private subnets, 2-3 AZs)
 client ─► internal ALB ─► gateway pods (Go, CPU nodes) ──prefix-aware──► vLLM pods (GPU nodes)
           idle timeout     - API keys / token budgets      routing over     - Llama 3.1 8B, 1x L4 each
           600s             - DNS discovery of vLLM pods    headless Svc     - /metrics
                            - /metrics                                      ▲
                                                                            │ pending pods trigger
 Prometheus ◄── scrapes vLLM, gateway, DCGM ──► KEDA ──► HPA on vLLM ──────┘ new GPU nodes
     │                                                   (replicas)        (Auto Mode / Karpenter NodePool)
     └─► Grafana dashboards + PrometheusRule alerts (TTFT/ITL SLO, queueing, KV full, Xid)
 S3 (model weights, same region) ──► init container / streamer ──► node-local storage ──► GPU
```

Two autoscalers form a chain: **KEDA/HPA** adds vLLM *pods* when work per replica exceeds its target. The pods are Pending because no GPU is free, so the **NodePool** launches GPU *nodes* that fit them. When load drops, pods scale in and empty nodes are consolidated away.

## 2. Two ways to build the cluster

| | Path A: **EKS Auto Mode** (course default) | Path B: managed node groups (+ Karpenter) |
|---|---|---|
| Node provisioning | built-in (Karpenter-based), configured with `NodePool` | managed node groups or self-installed Karpenter |
| NVIDIA driver + device plugin | **included and managed** for GPU instances | AL2023 NVIDIA AMI has the driver; you install the device plugin (Helm) |
| Load balancer controller | built in (`eks.amazonaws.com/alb`) | install the AWS Load Balancer Controller |
| Control over the AMI/OS | less | full |
| Choose when | you want to focus on inference, not node plumbing | you need custom AMIs, specific drivers, or existing node tooling |

Path A files: `labs/cluster/auto-mode-cluster.yaml` + `labs/manifests/01-gpu-nodepool.yaml`. Path B: `labs/cluster/managed-nodegroup-cluster.yaml` (read the device-plugin note at the top).

## 3. Step by step

`labs/deploy.sh` runs every step below in order. Read the steps first and run the script once you understand them.

### Step 0: Prerequisites
- Tools: `aws` CLI v2, `eksctl` (≥ 0.195 for Auto Mode), `kubectl`, `helm`, `docker`.
- **Quota:** ≥ 8 vCPUs of "Running On-Demand G and VT instances" in your region (level 11).
- A Hugging Face token with the Llama 3.1 license accepted, or switch the model to `Qwen/Qwen2.5-7B-Instruct` (ungated).

### Step 1: Cluster

```bash
eksctl create cluster -f levels/12-vllm-on-eks/labs/cluster/auto-mode-cluster.yaml    # ~15-20 min
kubectl get nodepools        # general-purpose, system (CPU)
```

### Step 2: GPU NodePool

`01-gpu-nodepool.yaml` allows `g6` and `g6e` on-demand instances, taints them `nvidia.com/gpu:NoSchedule`, caps the pool at 8 GPUs (a **cost guardrail**: an autoscaling bug can't launch 100 GPUs), and only consolidates **empty** nodes so busy inference pods are never evicted to save money.

```bash
kubectl apply -f labs/manifests/00-namespace.yaml -f labs/manifests/01-gpu-nodepool.yaml
kubectl -n inference create secret generic hf-token --from-literal=token=$HF_TOKEN
```

### Step 3: Observability first

Install it before the workload, so you see the first cold start:

```bash
helm upgrade --install kps prometheus-community/kube-prometheus-stack -n monitoring --create-namespace \
  -f labs/monitoring/kps-values.yaml
# GPU metrics: NVIDIA DCGM exporter (on Path B, the GPU Operator can install it for you)
helm repo add gpu-helm-charts https://nvidia.github.io/dcgm-exporter/helm-charts
helm install dcgm gpu-helm-charts/dcgm-exporter -n monitoring \
  --set-json 'tolerations=[{"key":"nvidia.com/gpu","operator":"Exists","effect":"NoSchedule"}]' \
  --set serviceMonitor.enabled=true
```

### Step 4: vLLM

```bash
kubectl apply -f labs/manifests/02-vllm.yaml -f labs/monitoring/podmonitors.yaml -f labs/monitoring/alerts.yaml
kubectl get nodeclaims -w                    # watch a g6 node being launched for the pending pod
kubectl -n inference logs -f deploy/vllm     # weights download, KV cache size, CUDA graph capture
```

Things in `02-vllm.yaml` that come from earlier levels:

| Setting | Why (level) |
|---|---|
| `--max-model-len=8192` | fit enough sequences in the L4's KV budget (6, 8) |
| `--max-num-seqs=64`, `--enable-prefix-caching` | batching ceiling; reuse of system prompts (7, 8) |
| `nodeSelector: instance-family g6` | an 8B BF16 model at 8k fits a 24 GB L4 (11) |
| `startupProbe` with a 15-minute budget | cold start = node + image + weights + graphs (10) |
| `maxUnavailable: 0, maxSurge: 1` + PDB | never lose capacity during rollouts or drains (10) |
| `/dev/shm` memory volume | NCCL needs it once you use TP > 1 (3, 10) |
| `preStop` sleep, 120 s grace period | in-flight streams finish before shutdown (10) |
| headless Service `vllm-headless` | the gateway discovers pod IPs for prefix routing (10) |

**Write down the cold-start breakdown** from the events and logs: node launch → image pull → weights → ready. It goes in your report, and step 7 shrinks it.

### Step 5: Gateway and internal ALB

```bash
# build/push the gateway to ECR (deploy.sh does this), then:
kubectl apply -f labs/manifests/03-gateway.yaml -f labs/manifests/04-ingress.yaml
kubectl -n inference get ingress gateway     # internal ALB DNS name (reachable from inside the VPC)
```

The ALB's **idle timeout is raised to 600 s**. With the 60 s default, any request whose first byte takes longer than a minute (long prompts under load, or non-streaming requests) gets a 504 from the load balancer, while vLLM keeps generating for a client that has gone. The gateway also sets `X-Accel-Buffering: no` and flushes every chunk, so no proxy buffers the token stream.

### Step 6: Autoscaling

`05-keda-autoscaling.yaml` scales on two signals and the HPA takes the higher:
1. **Requests in the system per replica** (`running + waiting`) against a target of 48 (out of `max-num-seqs` 64). That leaves headroom so bursts queue briefly instead of breaking the TTFT SLO.
2. **KV cache usage** against 80%, for long-context traffic that fills memory before it fills batch slots.

It scales up fast (2 pods per minute) and down slowly (1 pod per 5 minutes after a 10-minute window). A new replica takes minutes, so scaling in too eagerly causes cold-start storms.

### Step 7: Faster cold starts (pick at least one)

| Technique | How | Saves |
|---|---|---|
| Weights from S3 in-region | `07-s3-weights-initcontainer.yaml` + Pod Identity | HF download time and rate limits |
| Model streaming | vLLM `--load-format runai_streamer` with an `s3://` model path (check your version's docs) | overlaps download and GPU load |
| Image pre-pull / lazy loading | DaemonSet that pre-pulls the vLLM image on GPU nodes; SOCI lazy loading | image pull time |
| Warm headroom | `minReplicaCount` ≥ expected base load + 1; over-provisioning "pause" pods to keep a spare GPU node warm | node launch time |

Pod Identity for the S3 pattern:

```bash
aws iam create-role --role-name vllm-s3-read --assume-role-policy-document '{
  "Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"pods.eks.amazonaws.com"},
  "Action":["sts:AssumeRole","sts:TagSession"]}]}'
aws iam put-role-policy --role-name vllm-s3-read --policy-name read-models --policy-document '{
  "Version":"2012-10-17","Statement":[
   {"Effect":"Allow","Action":"s3:ListBucket","Resource":"arn:aws:s3:::YOUR-BUCKET","Condition":{"StringLike":{"s3:prefix":"models/*"}}},
   {"Effect":"Allow","Action":"s3:GetObject","Resource":"arn:aws:s3:::YOUR-BUCKET/models/*"}]}'
kubectl -n inference create serviceaccount vllm
aws eks create-pod-identity-association --cluster-name inference-academy --namespace inference \
  --service-account vllm --role-arn arn:aws:iam::<ACCOUNT>:role/vllm-s3-read
```

### Step 8: Load test to your SLO

From a pod or an instance in the VPC (or through `kubectl port-forward`):

```bash
go run ./platform/cmd/loadgen -url http://<gateway> -api-key team-support -model llama-3.1-8b \
  -rate 4 -n 400 -prompt-min 256 -prompt-max 2048 -output-min 64 -output-max 512 \
  -prefix-tokens 1500 -prefix-groups 6 -slo-ttft 1s -slo-tpot 60ms
```

Run `levels/09-benchmarking-and-slos/labs/sweep.sh` against one replica to find **max goodput per replica**, then set the KEDA target from it. Then run a **5× ramp** and watch Grafana (`labs/monitoring/grafana-dashboard.json`): queue → new pods Pending → NodeClaims → nodes Ready → pods Ready → queue drains. **How long did the SLO burn?** That's the number warm headroom has to cover.

## 4. Scaling the model up

**Llama 3.1 70B on one node.** Options (check with `infercalc` and `pick_instance.py`):

| Shape | Config | Notes |
|---|---|---|
| g6e.12xlarge (4× L40S, PCIe) | FP8 checkpoint, `--tensor-parallel-size 4`, `--max-model-len 8192` | fits; little KV room; PCIe all-reduce tax |
| p4de.24xlarge (8× A100 80 GB) | BF16, TP=8, or 2 replicas × TP=4 | NVLink; FP8 weights run as weight-only on Ampere |
| p5.48xlarge (8× H100) | FP8, 4 replicas × TP=2 | usually the best $/token at scale; capacity is scarce (Capacity Blocks) |

Changes from `02-vllm.yaml`: request `nvidia.com/gpu: "4"` (or 2/8), add `--tensor-parallel-size`, raise `/dev/shm` and memory limits, widen the NodePool's instance families, and allow a longer startup budget (~140 GB to load).

**Multi-node (405B).** `06-multinode-lws.yaml` uses a **LeaderWorkerSet**: one group = leader + workers on separate nodes, started together and restarted together, with Ray coordinating vLLM's workers. TP=8 inside each node over NVLink, PP=2 across nodes over **EFA**. Prerequisites: p5-class capacity (usually a Capacity Block), the EFA device plugin, and a NodePool for those instances. This is a Tier 2 lab; a recorded walkthrough is fine if you lack capacity.

## 5. Operations checklist

- [ ] GPU NodePool has a `limits.nvidia.com/gpu` cap, and consolidation is `WhenEmpty`
- [ ] vLLM image and model revision pinned; upgrades canaried on one replica
- [ ] startupProbe budget > worst-case cold start; `maxUnavailable: 0`; PDB present
- [ ] ALB internal, idle timeout raised; streaming verified end to end through every proxy
- [ ] KEDA targets derived from measured goodput per replica (level 9), not guessed
- [ ] Dashboards: TTFT/ITL p50/p95, running/waiting, KV usage, prefix hit rate, tokens/s, GPU util/memory, per-tenant traffic
- [ ] Alerts: SLO burn, sustained queueing, KV near full, GPU Xid errors, pods Pending > N minutes (capacity)
- [ ] Weights in in-region S3; Pod Identity, no static keys; HF token not needed at runtime
- [ ] `cleanup.sh` tested; budget alarms set in AWS Budgets

## 6. Check yourself

<details><summary>1. Load rises, KEDA raises desired replicas from 2 to 4, but TTFT stays bad for 6 minutes. Walk through what's happening.</summary>
The new pods are Pending until GPU nodes launch (1–3 min). Then come image pull, weights load and CUDA graph capture (several minutes) before they're Ready. Meanwhile the original replicas are saturated. Fixes: warm headroom, faster cold starts (step 7), and scaling earlier (lower targets).</details>

<details><summary>2. Why not let Karpenter consolidate under-utilized GPU nodes?</summary>
Consolidation evicts pods and moves them elsewhere. For vLLM that means killing in-flight generations and paying a multi-minute cold start to save a little money. Consolidate only empty nodes.</details>

<details><summary>3. Why does the gateway discover pods through a headless Service instead of using the ClusterIP Service?</summary>
A ClusterIP Service load-balances connections randomly (kube-proxy), which destroys prefix-cache affinity. The gateway needs individual pod endpoints to route a prefix to the same replica.</details>

<details><summary>4. A non-streaming request with max_tokens=4000 returns 504 after exactly 60 s. What is it?</summary>
The ALB idle timeout (default 60 s). Raise it, and prefer streaming for long generations.</details>

## 7. Going deeper
- EKS user guide: Auto Mode, accelerated workloads, Pod Identity; EKS best practices: AI/ML
- Karpenter docs (NodePools, disruption); KEDA Prometheus scaler
- AWS Load Balancer Controller / Auto Mode ingress annotations
- vLLM docs: Kubernetes deployment, production-stack Helm chart; LeaderWorkerSet docs
- AI on EKS (awslabs) blueprints for inference

**Next → [Level 13: SageMaker and managed options](../13-sagemaker/README.md)**
