"""Lab 13d: inference components - several models (or copies) packed on one endpoint.

    python3 inference_components.py --image-uri ... --role-arn ... --instance-type ml.g6.12xlarge

An endpoint with inference components owns a pool of instances; each component
declares the accelerators and memory one copy needs, and SageMaker places copies
onto instances (bin-packing). Each component scales its copy count independently,
and the endpoint can scale its instances (down to zero, with managed instance scaling).

Here: one ml.g6.12xlarge (4 x L4) hosting a Llama 8B component and a Qwen 7B component,
each using one GPU per copy.
"""

import argparse
import time

import boto3

MODELS = {
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image-uri", required=True)
    p.add_argument("--role-arn", required=True)
    p.add_argument("--hf-token", default="")
    p.add_argument("--instance-type", default="ml.g6.12xlarge")
    p.add_argument("--name", default="academy-ic")
    p.add_argument("--region", default="us-east-1")
    a = p.parse_args()
    sm = boto3.client("sagemaker", region_name=a.region)

    sm.create_endpoint_config(
        EndpointConfigName=a.name,
        ExecutionRoleArn=a.role_arn,
        ProductionVariants=[{
            "VariantName": "AllTraffic",
            "InstanceType": a.instance_type,
            "InitialInstanceCount": 1,
            "ManagedInstanceScaling": {"Status": "ENABLED", "MinInstanceCount": 1, "MaxInstanceCount": 2},
            "RoutingConfig": {"RoutingStrategy": "LEAST_OUTSTANDING_REQUESTS"},
        }],
    )
    sm.create_endpoint(EndpointName=a.name, EndpointConfigName=a.name)
    sm.get_waiter("endpoint_in_service").wait(EndpointName=a.name, WaiterConfig={"Delay": 30, "MaxAttempts": 60})
    print("endpoint ready (no models yet)")

    for short, hf_id in MODELS.items():
        model_name = f"{a.name}-{short}"
        env = {"MODEL_ID": hf_id,
               "VLLM_ARGS": "--max-model-len 8192 --gpu-memory-utilization 0.90 --served-model-name llm"}
        if a.hf_token:
            env["HF_TOKEN"] = a.hf_token
        sm.create_model(ModelName=model_name, ExecutionRoleArn=a.role_arn,
                        PrimaryContainer={"Image": a.image_uri, "Environment": env})
        sm.create_inference_component(
            InferenceComponentName=f"{model_name}-ic",
            EndpointName=a.name,
            VariantName="AllTraffic",
            Specification={
                "ModelName": model_name,
                "StartupParameters": {"ContainerStartupHealthCheckTimeoutInSeconds": 1800,
                                      "ModelDataDownloadTimeoutInSeconds": 1800},
                "ComputeResourceRequirements": {"NumberOfAcceleratorDevicesRequired": 1,
                                                "MinMemoryRequiredInMb": 16384},
            },
            RuntimeConfig={"CopyCount": 1},
        )
        print("creating component", f"{model_name}-ic")

    for short in MODELS:
        ic = f"{a.name}-{short}-ic"
        while sm.describe_inference_component(InferenceComponentName=ic)["InferenceComponentStatus"] != "InService":
            time.sleep(30)
        print(ic, "InService ->  python3 invoke.py --name", a.name, "--inference-component", ic)


if __name__ == "__main__":
    main()
