"""
deploy_aci.py – Deploy the churn prediction Docker image to Azure Container Instances (ACI).

Prerequisites:
  - Docker image must be built & pushed to ACR (done by CI/CD pipeline)
  - Env vars: AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP (or ACI_RESOURCE_GROUP),
              ACR_NAME, ACR_IMAGE, ACR_PASSWORD

Usage:
    python deploy/deploy_aci.py
    python deploy/deploy_aci.py --delete   # tear down existing ACI group
"""

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from azure.identity import DefaultAzureCredential
from azure.mgmt.containerinstance import ContainerInstanceManagementClient
from azure.mgmt.containerinstance.models import (
    ContainerGroup,
    Container,
    ContainerPort,
    EnvironmentVariable,
    ImageRegistryCredential,
    IpAddress,
    OperatingSystemTypes,
    Port,
    ResourceRequests,
    ResourceRequirements,
)

from src.config import (
    SUBSCRIPTION_ID,
    ACI_RESOURCE_GROUP,
    ACI_CONTAINER_GROUP,
    ACI_LOCATION,
    ACR_NAME,
    ACR_IMAGE,
)

ACR_PASSWORD = os.getenv("ACR_PASSWORD", "")
ACR_LOGIN_SERVER = f"{ACR_NAME}.azurecr.io"
FULL_IMAGE = f"{ACR_LOGIN_SERVER}/{ACR_IMAGE}"


def deploy_aci() -> None:
    print(f"[deploy] Deploying '{FULL_IMAGE}' to ACI group '{ACI_CONTAINER_GROUP}' …")

    credential = DefaultAzureCredential()
    client = ContainerInstanceManagementClient(credential, SUBSCRIPTION_ID)

    container_group = ContainerGroup(
        location=ACI_LOCATION,
        containers=[
            Container(
                name="churn-api",
                image=FULL_IMAGE,
                resources=ResourceRequirements(
                    requests=ResourceRequests(memory_in_gb=1.5, cpu=1.0)
                ),
                ports=[ContainerPort(port=80)],
                environment_variables=[
                    EnvironmentVariable(name="PORT", value="80"),
                ],
            )
        ],
        os_type=OperatingSystemTypes.LINUX,
        ip_address=IpAddress(
            ports=[Port(protocol="TCP", port=80)],
            type="Public",
            dns_name_label=ACI_CONTAINER_GROUP,
        ),
        image_registry_credentials=[
            ImageRegistryCredential(
                server=ACR_LOGIN_SERVER,
                username=ACR_NAME,
                password=ACR_PASSWORD,
            )
        ],
    )

    poller = client.container_groups.begin_create_or_update(
        ACI_RESOURCE_GROUP, ACI_CONTAINER_GROUP, container_group
    )
    result = poller.result()
    fqdn = result.ip_address.fqdn
    print(f"[deploy] ✅ ACI deployed!")
    print(f"[deploy] Endpoint : http://{fqdn}/predict")
    print(f"[deploy] Health   : http://{fqdn}/health")


def delete_aci() -> None:
    print(f"[deploy] Deleting ACI group '{ACI_CONTAINER_GROUP}' …")
    credential = DefaultAzureCredential()
    client = ContainerInstanceManagementClient(credential, SUBSCRIPTION_ID)
    client.container_groups.begin_delete(ACI_RESOURCE_GROUP, ACI_CONTAINER_GROUP).result()
    print("[deploy] ACI group deleted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete", action="store_true", help="Delete the ACI container group")
    args = parser.parse_args()

    if not SUBSCRIPTION_ID:
        raise EnvironmentError("AZURE_SUBSCRIPTION_ID not set.")

    if args.delete:
        delete_aci()
    else:
        deploy_aci()
