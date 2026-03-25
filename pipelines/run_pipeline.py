"""
run_pipeline.py – Submit the Azure ML end-to-end pipeline from the CLI.

Usage:
    python pipelines/run_pipeline.py
    python pipelines/run_pipeline.py --wait   # block until pipeline completes
"""

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, load_job
from azure.ai.ml.entities import PipelineJob


def submit_pipeline(wait: bool = False) -> None:
    from src.config import SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME

    if not SUBSCRIPTION_ID:
        raise EnvironmentError(
            "AZURE_SUBSCRIPTION_ID environment variable not set. "
            "Export it before running this script."
        )

    print(f"[pipeline] Connecting to Azure ML workspace '{WORKSPACE_NAME}' …")
    ml_client = MLClient(
        DefaultAzureCredential(), SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME
    )

    pipeline_job: PipelineJob = load_job(
        source=os.path.join(os.path.dirname(__file__), "pipeline.yml")
    )

    print("[pipeline] Submitting pipeline job …")
    submitted = ml_client.jobs.create_or_update(pipeline_job)
    print(f"[pipeline] Job submitted: {submitted.name}")
    print(f"[pipeline] Studio URL  : {submitted.studio_url}")

    if wait:
        print("[pipeline] Waiting for pipeline to complete …")
        ml_client.jobs.stream(submitted.name)
        print("[pipeline] Pipeline finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", action="store_true", help="Block until pipeline finishes")
    args = parser.parse_args()
    submit_pipeline(wait=args.wait)
