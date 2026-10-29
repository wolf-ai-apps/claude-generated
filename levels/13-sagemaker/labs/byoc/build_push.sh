#!/usr/bin/env bash
# Build the BYOC image and push it to ECR.  ./build_push.sh  -> prints the image URI
set -euo pipefail
REGION=${REGION:-us-east-1}
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REPO=academy/vllm-sagemaker
TAG=${TAG:-v0.11.0}
URI="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPO:$TAG"
aws ecr describe-repositories --repository-names "$REPO" --region "$REGION" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$REPO" --region "$REGION" >/dev/null
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "${URI%%/*}"
docker build --platform linux/amd64 --build-arg VLLM_TAG="$TAG" -t "$URI" "$(dirname "$0")"
docker push "$URI"
echo "$URI"
