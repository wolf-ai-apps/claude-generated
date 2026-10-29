"""Lab 13e: delete endpoints, inference components, configs and models created by this level.

    python3 cleanup.py --prefix academy-
An InService endpoint bills per instance-hour even with zero traffic.
"""

import argparse
import time

import boto3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", default="academy-")
    p.add_argument("--region", default="us-east-1")
    a = p.parse_args()
    sm = boto3.client("sagemaker", region_name=a.region)
    aas = boto3.client("application-autoscaling", region_name=a.region)

    for ic in sm.list_inference_components(NameContains=a.prefix)["InferenceComponents"]:
        sm.delete_inference_component(InferenceComponentName=ic["InferenceComponentName"])
        print("deleting inference component", ic["InferenceComponentName"])
    while sm.list_inference_components(NameContains=a.prefix)["InferenceComponents"]:
        time.sleep(15)

    for ep in sm.list_endpoints(NameContains=a.prefix)["Endpoints"]:
        name = ep["EndpointName"]
        targets = aas.describe_scalable_targets(ServiceNamespace="sagemaker",
                                                ResourceIds=[f"endpoint/{name}/variant/AllTraffic"])
        for t in targets["ScalableTargets"]:
            aas.deregister_scalable_target(ServiceNamespace="sagemaker", ResourceId=t["ResourceId"],
                                           ScalableDimension=t["ScalableDimension"])
        sm.delete_endpoint(EndpointName=name)
        print("deleting endpoint", name)
    for cfg in sm.list_endpoint_configs(NameContains=a.prefix)["EndpointConfigs"]:
        sm.delete_endpoint_config(EndpointConfigName=cfg["EndpointConfigName"])
        print("deleted endpoint config", cfg["EndpointConfigName"])
    for m in sm.list_models(NameContains=a.prefix)["Models"]:
        sm.delete_model(ModelName=m["ModelName"])
        print("deleted model", m["ModelName"])


if __name__ == "__main__":
    main()
