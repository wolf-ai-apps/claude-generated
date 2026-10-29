#!/usr/bin/env bash
# Tear everything down. Deleting the cluster removes Auto Mode nodes, the ALB and the EBS volumes it created.
set -euo pipefail
REGION=${REGION:-us-east-1}
CLUSTER=${CLUSTER:-inference-academy}
kubectl delete ingress -n inference gateway --ignore-not-found --wait=true   # let the ALB be deleted first
kubectl delete nodepool gpu --ignore-not-found
eksctl delete cluster --name "$CLUSTER" --region "$REGION" --wait
echo "Check the EC2 console for leftover GPU instances and the ECR repo academy/gateway (delete it if unused)."
