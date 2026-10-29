"""Lab 13a: deploy vLLM to a SageMaker real-time endpoint with plain boto3.

    pip install boto3
    python3 deploy.py --image-uri <from byoc/build_push.sh> --role-arn arn:aws:iam::<acct>:role/<SageMakerExecRole> \
        --model-id meta-llama/Llama-3.1-8B-Instruct --hf-token $HF_TOKEN

Using boto3 directly (instead of a higher-level SDK) shows the three resources
every SageMaker deployment is made of:
    Model           = container image + environment (+ optional S3 model data)
    EndpointConfig  = which instance type, how many, timeouts
    Endpoint        = the running, billable thing behind an HTTPS API

Note: environment variables are visible to anyone who can describe the model.
For production, ship weights as S3 model data (no token needed) or fetch secrets at startup.
"""

import argparse
import time

import boto3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image-uri", required=True)
    p.add_argument("--role-arn", required=True, help="SageMaker execution role (ECR pull, CloudWatch logs, S3 read)")
    p.add_argument("--model-id", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--hf-token", default="")
    p.add_argument("--model-data-s3", default="", help="s3://bucket/prefix/ with uncompressed weights (optional)")
    p.add_argument("--instance-type", default="ml.g6.2xlarge", help="1 x L4 24GB; ml.g5.2xlarge = 1 x A10G 24GB")
    p.add_argument("--name", default="academy-llama-8b")
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--vllm-args", default="--max-model-len 8192 --gpu-memory-utilization 0.90 "
                                          "--max-num-seqs 64 --enable-prefix-caching --served-model-name llm")
    a = p.parse_args()

    sm = boto3.client("sagemaker", region_name=a.region)
    suffix = time.strftime("%Y%m%d-%H%M%S")
    model_name = f"{a.name}-{suffix}"
    env = {"MODEL_ID": a.model_id, "VLLM_ARGS": a.vllm_args}
    if a.hf_token:
        env["HF_TOKEN"] = a.hf_token

    container = {"Image": a.image_uri, "Environment": env}
    if a.model_data_s3:
        # Uncompressed S3 prefix: SageMaker copies it to /opt/ml/model (no tar.gz extraction step).
        container["ModelDataSource"] = {"S3DataSource": {
            "S3Uri": a.model_data_s3, "S3DataType": "S3Prefix", "CompressionType": "None"}}

    sm.create_model(ModelName=model_name, ExecutionRoleArn=a.role_arn, PrimaryContainer=container)
    print("created model", model_name)

    config_name = f"{a.name}-{suffix}"
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            "VariantName": "AllTraffic",
            "ModelName": model_name,
            "InstanceType": a.instance_type,
            "InitialInstanceCount": 1,
            # Large models take minutes to download + load + capture CUDA graphs.
            # The default health-check window is too short for that.
            "ContainerStartupHealthCheckTimeoutInSeconds": 1800,
            "ModelDataDownloadTimeoutInSeconds": 1800,
        }],
    )
    print("created endpoint config", config_name)

    existing = sm.list_endpoints(NameContains=a.name)["Endpoints"]
    if any(e["EndpointName"] == a.name for e in existing):
        sm.update_endpoint(EndpointName=a.name, EndpointConfigName=config_name)  # blue/green under the hood
        print("updating endpoint", a.name)
    else:
        sm.create_endpoint(EndpointName=a.name, EndpointConfigName=config_name)
        print("creating endpoint", a.name)

    t = time.time()
    sm.get_waiter("endpoint_in_service").wait(EndpointName=a.name, WaiterConfig={"Delay": 30, "MaxAttempts": 80})
    print(f"InService after {(time.time() - t) / 60:.1f} min. Logs: CloudWatch /aws/sagemaker/Endpoints/{a.name}")


if __name__ == "__main__":
    main()
