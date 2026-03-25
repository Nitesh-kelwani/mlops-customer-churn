import os

# ── Local paths ──────────────────────────────────────────────────────────────
DATA_PATH       = "data/raw/churn.csv"
MODEL_PATH      = "model/churn_pipeline.pkl"
TARGET_COL      = "Churn"

# ── Azure Blob Storage ────────────────────────────────────────────────────────
BLOB_CONN_STR   = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
BLOB_CONTAINER  = os.getenv("AZURE_BLOB_CONTAINER", "mlops-churn")
BLOB_FILE       = os.getenv("AZURE_BLOB_FILE", "churn.csv")

# ── Azure ML Workspace ────────────────────────────────────────────────────────
SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID", "")
RESOURCE_GROUP  = os.getenv("AZURE_RESOURCE_GROUP", "mlops-rg")
WORKSPACE_NAME  = os.getenv("AZURE_ML_WORKSPACE", "mlops-workspace")

# ── Azure Container Registry ──────────────────────────────────────────────────
ACR_NAME        = os.getenv("ACR_NAME", "mlopsacr")
ACR_IMAGE       = os.getenv("ACR_IMAGE", "churn-api:latest")

# ── Azure Container Instances ─────────────────────────────────────────────────
ACI_RESOURCE_GROUP = os.getenv("ACI_RESOURCE_GROUP", RESOURCE_GROUP)
ACI_CONTAINER_GROUP = os.getenv("ACI_CONTAINER_GROUP", "churn-api-aci")
ACI_LOCATION     = os.getenv("ACI_LOCATION", "eastus")

# ── Model Registry ────────────────────────────────────────────────────────────
MODEL_NAME      = "customer-churn-ensemble"
EXPERIMENT_NAME = "churn-prediction"
