"""Lab 13c: autoscale the endpoint on concurrent requests.

    python3 autoscale.py --name academy-llama-8b --min 1 --max 3 --target 32

SageMaker endpoints scale through Application Auto Scaling. For LLMs, scale on
concurrency (requests in flight per instance), not invocations per minute: one
long generation and one short one are both "one invocation" but very different load.
The high-resolution variant of the metric is emitted every 10 seconds, so
scale-out reacts faster than with the classic 1-minute metrics.
"""

import argparse

import boto3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="academy-llama-8b")
    p.add_argument("--variant", default="AllTraffic")
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--min", type=int, default=1)
    p.add_argument("--max", type=int, default=3)
    p.add_argument("--target", type=float, default=32, help="concurrent requests per instance (below --max-num-seqs)")
    a = p.parse_args()

    aas = boto3.client("application-autoscaling", region_name=a.region)
    resource = f"endpoint/{a.name}/variant/{a.variant}"
    aas.register_scalable_target(ServiceNamespace="sagemaker", ResourceId=resource,
                                 ScalableDimension="sagemaker:variant:DesiredInstanceCount",
                                 MinCapacity=a.min, MaxCapacity=a.max)
    aas.put_scaling_policy(
        PolicyName=f"{a.name}-concurrency",
        ServiceNamespace="sagemaker",
        ResourceId=resource,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        PolicyType="TargetTrackingScaling",
        TargetTrackingScalingPolicyConfiguration={
            "TargetValue": a.target,
            "PredefinedMetricSpecification": {
                "PredefinedMetricType": "SageMakerVariantConcurrentRequestsPerModelHighResolution"},
            "ScaleOutCooldown": 120,   # a new GPU instance takes minutes to become InService
            "ScaleInCooldown": 600,    # scale in slowly: cold starts are expensive
        },
    )
    print(f"autoscaling {resource}: {a.min}-{a.max} instances, target {a.target} concurrent requests/instance")


if __name__ == "__main__":
    main()
