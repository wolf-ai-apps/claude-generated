#!/usr/bin/env bash
# Level 12 end-to-end deploy (Path A: EKS Auto Mode). Read each step before running it:
# GPU instances bill by the second from launch. Run ./cleanup.sh when you are done.
set -euo pipefail
: "${HF_TOKEN:?export HF_TOKEN=hf_... (accept the Llama 3.1 license on Hugging Face first)}"
REGION=${REGION:-us-east-1}
CLUSTER=${CLUSTER:-inference-academy}
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)

step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

step "1. Quota check: G-family on-demand vCPUs (need >= 8 for one g6.2xlarge)"
aws service-quotas get-service-quota --region "$REGION" --service-code ec2 --quota-code L-DB2E81BA \
  --query 'Quota.Value' --output text || echo "could not read quota; check the Service Quotas console"

step "2. Cluster (15-20 minutes)"
aws eks describe-cluster --name "$CLUSTER" --region "$REGION" >/dev/null 2>&1 \
  || eksctl create cluster -f "$HERE/cluster/auto-mode-cluster.yaml"
aws eks update-kubeconfig --name "$CLUSTER" --region "$REGION"

step "3. GPU NodePool + namespace + Hugging Face secret"
kubectl apply -f "$HERE/manifests/00-namespace.yaml" -f "$HERE/manifests/01-gpu-nodepool.yaml"
kubectl -n inference create secret generic hf-token --from-literal=token="$HF_TOKEN" \
  --dry-run=client -o yaml | kubectl apply -f -

step "4. Monitoring (Prometheus + Grafana + DCGM GPU metrics) and KEDA"
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts >/dev/null
helm repo add kedacore https://kedacore.github.io/charts >/dev/null
helm repo add gpu-helm-charts https://nvidia.github.io/dcgm-exporter/helm-charts >/dev/null
helm repo update >/dev/null
helm upgrade --install kps prometheus-community/kube-prometheus-stack -n monitoring --create-namespace \
  -f "$HERE/monitoring/kps-values.yaml" --wait
helm upgrade --install dcgm gpu-helm-charts/dcgm-exporter -n monitoring \
  --set-json 'tolerations=[{"key":"nvidia.com/gpu","operator":"Exists","effect":"NoSchedule"}]' \
  --set serviceMonitor.enabled=true
helm upgrade --install keda kedacore/keda -n keda --create-namespace --wait

step "5. vLLM (first start downloads ~16 GB of weights; watch: kubectl -n inference logs -f deploy/vllm)"
kubectl apply -f "$HERE/manifests/02-vllm.yaml" -f "$HERE/monitoring/podmonitors.yaml" -f "$HERE/monitoring/alerts.yaml"
kubectl -n inference rollout status deploy/vllm --timeout=30m

step "6. Gateway image -> ECR, then deploy gateway + internal ALB"
REPO="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/academy/gateway"
aws ecr describe-repositories --repository-names academy/gateway --region "$REGION" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name academy/gateway --region "$REGION" >/dev/null
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "${REPO%%/*}"
docker build --platform linux/amd64 -t "$REPO:v1" --build-arg CMD=gateway "$ROOT/platform"
docker push "$REPO:v1"
sed "s|ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/academy/gateway:v1|$REPO:v1|" "$HERE/manifests/03-gateway.yaml" | kubectl apply -f -
kubectl apply -f "$HERE/manifests/04-ingress.yaml"

step "7. Autoscaling"
kubectl apply -f "$HERE/manifests/05-keda-autoscaling.yaml"

step "Done. Smoke test from your laptop:"
echo "  kubectl -n inference port-forward svc/gateway 9000:9000 &"
echo "  curl -s localhost:9000/v1/chat/completions -H 'Authorization: Bearer team-support' -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"llama-3.1-8b\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":32}'"
echo "  Grafana: kubectl -n monitoring port-forward svc/kps-grafana 3000:80  (import monitoring/grafana-dashboard.json)"
