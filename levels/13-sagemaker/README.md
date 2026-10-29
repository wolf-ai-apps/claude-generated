# Level 13: SageMaker and managed options

> SageMaker runs the cluster for you. You give up some control and pay a premium per instance-hour, and in return there are no nodes, no autoscaler plumbing and no load balancer to operate. Learn exactly what it does so you can choose deliberately.

**You will:** deploy the same vLLM + Llama 3.1 8B service to a SageMaker real-time endpoint with plain boto3, stream from it, autoscale it on concurrency, pack several models onto one endpoint with inference components, and know when to use Bedrock instead.
**Time:** 4–6 hours · **Hardware:** AWS with SageMaker quota for `ml.g6.2xlarge` (or `ml.g5.2xlarge`) endpoint usage

---

## 1. Concepts

### 1.1 SageMaker inference options

| Option | How it behaves | GPU? | Use for |
|---|---|---|---|
| **Real-time endpoint** | always-on instances behind an HTTPS API; autoscaling | yes | interactive LLM serving (this level) |
| Real-time + **inference components** | one endpoint, many models/copies bin-packed onto instances; per-model scaling; scale to zero | yes | many models or fine-tunes, spiky traffic |
| **Asynchronous inference** | requests queued (S3 in/out), long timeouts, scale to zero | yes | long generations, batch-like interactive work |
| **Batch transform** | offline job over an S3 dataset | yes | bulk offline inference |
| Serverless inference | per-request, scale to zero | **no** (CPU only) | small non-LLM models |

### 1.2 What a SageMaker deployment is

```
Model           = container image (ECR) + environment variables + optional model data (S3 -> /opt/ml/model)
EndpointConfig  = production variant(s): instance type, count, health-check & download timeouts
Endpoint        = running instances + managed load balancer + HTTPS API (SigV4-authenticated)
```

**The container contract:** SageMaker runs your image with the argument `serve`, health-checks `GET /ping` and sends traffic to `POST /invocations`, both on **port 8080**. vLLM's OpenAI server implements `/ping` and `/invocations` itself. The academy's BYOC image (`labs/byoc/`) only sets the port, the host and the model location.

### 1.3 Three ways to get vLLM onto SageMaker

| Approach | What you manage | Notes |
|---|---|---|
| **BYOC: stock vLLM image + tiny entrypoint** (this level) | the image (pin the vLLM version) | full control and transparency; same engine and flags as EKS |
| AWS **LMI** (Large Model Inference) containers | environment variables (`HF_MODEL_ID`, `OPTION_*`) | AWS-maintained; vLLM (and other) backends; check the LMI docs for the current image tags and option names |
| AWS **vLLM Deep Learning Container** for SageMaker | environment variables (vLLM flags exposed as `SM_VLLM_*`) | AWS-maintained vLLM build; check the AWS DLC docs for image URIs and variables |

The BYOC route teaches the contract and keeps EKS and SageMaker on the **same engine version**, so benchmark comparisons are fair.

### 1.4 Scaling

- **Instance autoscaling** via Application Auto Scaling, target tracking on **concurrent requests per model/copy**. The high-resolution variants report every 10 s, so scaling reacts faster than with 1-minute invocation metrics.
- **Inference components** scale *copies* of each model and (with managed instance scaling) the instances under them, including **scale to zero** for components.
- New instances still take minutes (instance launch, image pull, weights, CUDA graphs). Startup tuning (container caching, model data in S3, smaller images) matters as much as on EKS.

### 1.5 SageMaker vs EKS vs Bedrock

| | EKS + vLLM | SageMaker + vLLM | Amazon Bedrock |
|---|---|---|---|
| You operate | cluster, nodes, autoscaling, LB, observability | container + config | nothing (API) |
| Billing | EC2 instance-hours | ml.* instance-hours (premium) | per token (or provisioned throughput) |
| Control over engine and flags | full | full (BYOC) | none |
| Custom/fine-tuned models | any | any | Custom Model Import for supported architectures; fine-tuning for some models |
| Routing / gateway customization | full (level 10) | limited (built-in routing strategies) | none |
| Best when | high volume, platform team in place, need control | small team, a handful of models, want managed ops | standard models, variable volume, fastest time to production |

**SageMaker HyperPod** sits between EKS and SageMaker. It's resilient GPU clusters orchestrated with EKS (or Slurm), with tooling for deploying inference on them. Consider it when you want EKS-style control with managed cluster health and node recovery.

---

## 2. Worked example: the same service, two platforms

| Step | EKS (level 12) | SageMaker (this level) |
|---|---|---|
| Capacity | quota + NodePool | quota for `ml.g6.2xlarge` endpoint usage |
| Engine | `vllm/vllm-openai:v0.11.0` | the same image + 8-line entrypoint |
| Config | container `args` | `VLLM_ARGS` env var |
| Exposure | internal ALB + gateway | SageMaker HTTPS endpoint (IAM/SigV4) |
| Autoscaling | KEDA on running+waiting per replica | target tracking on concurrent requests per model |
| Observability | Prometheus/Grafana (vLLM metrics) | CloudWatch endpoint metrics + container logs (vLLM `/metrics` isn't scraped for you) |
| Cold start | node + image + weights | instance + image + weights (similar) |

## 3. Labs

Prerequisites: a SageMaker execution role with ECR pull, CloudWatch Logs and (optionally) S3 read on your model bucket, plus endpoint quota.

```bash
cd levels/13-sagemaker/labs
# 13a. image
IMAGE=$(./byoc/build_push.sh | tail -1)
# 13b. deploy (10-20 min for the first InService)
python3 deploy.py --image-uri "$IMAGE" --role-arn arn:aws:iam::<acct>:role/<SageMakerExecRole> \
  --model-id meta-llama/Llama-3.1-8B-Instruct --hf-token "$HF_TOKEN"
#   better: ship weights from S3 (mirrored in level 11) and drop the token:
#   --model-data-s3 s3://YOUR-BUCKET/models/Llama-3.1-8B-Instruct/
# 13c. call it
python3 invoke.py --name academy-llama-8b
# 13d. autoscale on concurrency
python3 autoscale.py --name academy-llama-8b --min 1 --max 3 --target 32
# 13e. (optional, 4-GPU instance) two models packed on one endpoint
python3 inference_components.py --image-uri "$IMAGE" --role-arn ... --hf-token "$HF_TOKEN"
# ALWAYS
python3 cleanup.py --prefix academy-
```

Assignments:
1. **Cold start:** from the CloudWatch logs (`/aws/sagemaker/Endpoints/academy-llama-8b`), break down time to InService. Compare it with level 12's EKS breakdown.
2. **Latency overhead:** compare TTFT and ITL measured by `invoke.py` with the same model on EKS (through the ALB) at low load. How much does SageMaker's routing layer add?
3. **Load test:** SageMaker endpoints need SigV4-signed requests, so the academy loadgen can't call them directly. Write a small Python load generator with boto3 and threads (open loop, Poisson arrivals, streaming TTFT), or run the gateway with a SigV4-signing backend as a stretch goal. Find max goodput per instance and compare with EKS.
4. **Cost:** plug both platforms' measured goodput and current prices into `levels/14-production-operations/labs/cost_compare.py`.

## 4. Check yourself

<details><summary>1. What three resources make up a SageMaker deployment, and which one do you change to roll out a new model version?</summary>
Model, EndpointConfig, Endpoint. Create a new Model and EndpointConfig, then `update_endpoint` to point at the new config. SageMaker performs a blue/green swap.</details>

<details><summary>2. Your endpoint fails with "health check did not pass" after ~10 minutes, but the same image works on EKS. What should you check?</summary>
`ContainerStartupHealthCheckTimeoutInSeconds` (big models need longer), that vLLM listens on 0.0.0.0:8080, that `/ping` exists in your vLLM version, and that the model download had time to finish (`ModelDataDownloadTimeoutInSeconds`). Then read the CloudWatch container logs.</details>

<details><summary>3. When would you choose Bedrock over both EKS and SageMaker?</summary>
When a supported foundation model meets your quality needs, volume is variable or modest, and time to production and zero operations matter more than per-token cost at high volume or deep control over the engine.</details>

## 5. Going deeper
- SageMaker docs: real-time endpoints, inference components, autoscaling (including scale to zero), asynchronous inference
- LMI container docs (DJL Serving); AWS Deep Learning Containers for vLLM
- SageMaker HyperPod docs; Amazon Bedrock Custom Model Import

**Next → [Level 14: Production operations](../14-production-operations/README.md)**
