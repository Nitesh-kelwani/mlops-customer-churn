"""
azure_pipeline.py – Single-file Azure ML Pipeline definition.
Defines and submits the data ingestion and training job to Azure ML.
"""

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment

from src.config import SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME, EXPERIMENT_NAME

def submit_pipeline():
    if not SUBSCRIPTION_ID:
        raise EnvironmentError("AZURE_SUBSCRIPTION_ID not set.")

    print(f"[pipeline] Connecting to workspace '{WORKSPACE_NAME}' …")
    ml_client = MLClient(
        DefaultAzureCredential(), SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME
    )

    # 1. Define the command job (Pipeline Step)
    train_job = command(
        experiment_name=EXPERIMENT_NAME,
        display_name="Customer Churn - Train Ensemble",
        description="Downloads data from Blob, applies SMOTE, trains Random Forest/AdaBoost, and logs to MLflow.",
        command="python src/train.py",
        code="./",
        environment=Environment(
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
            conda_file="environment.yml"
        ),
        compute="cpu-cluster",
    )

    # 2. Submit the job
    print("[pipeline] Submitting pipeline job …")
    submitted = ml_client.jobs.create_or_update(train_job)
    print(f"[pipeline] Job submitted: {submitted.name}")
    print(f"[pipeline] Studio URL  : {submitted.studio_url}")


if __name__ == "__main__":
    submit_pipeline()
