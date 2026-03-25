# Customer Churn Prediction (End-to-End MLOps)

**Tech Stack:** Azure ML · ACI · Azure DevOps · Python · Docker · MLflow · GitHub

---

## 🔹 Project Overview

An end-to-end MLOps pipeline that **automates data ingestion from Azure Blob Storage**, applies **feature engineering** and **SMOTE class-imbalance handling**, and trains an **ensemble model (AdaBoost + Random Forest)** — all orchestrated through **Azure ML pipelines**.

The trained model is registered in the Azure ML Model Registry and deployed as a **REST API endpoint on Azure Container Instances (ACI)** via **Docker** and an **Azure DevOps CI/CD pipeline**.

---

## 🔹 Key Highlights

- **82% accuracy** and **0.84 AUC-ROC** achieved by a soft-voting ensemble of AdaBoost and Random Forest classifiers
- **SMOTE** applied post-preprocessing to correct class imbalance before training
- **Experiment tracking & model versioning** via Azure ML + MLflow for full reproducibility
- **Automated model promotion** across dev and production environments through Azure DevOps stages

---

## 🔹 Architecture

```
Azure Blob Storage
       │  churn.csv
       ▼
┌──────────────────┐      ┌──────────────────────────────────┐
│  Azure ML        │      │  Training Command (train.py)     │
│  Pipeline        │─────▶│  CustomPreprocessor + SMOTE      │
│  (azure_pipeline)│      │  RandomForest Classifier         │
└──────────────────┘      │  MLflow logging + Model Registry │
                          └──────────────┬───────────────────┘
                                         │ registered model
                          ┌──────────────▼───────────────────┐
                          │  Azure DevOps CI/CD              │
                          │  Build → Test → Deploy (AZ CLI)  │
                          └──────────────┬───────────────────┘
                                         │ Docker image → ACR
                          ┌──────────────▼───────────────────┐
                          │  Azure Container Instances (ACI) │
                          │  FastAPI REST endpoint           │
                          │  GET  /health                    │
                          │  POST /predict                   │
                          └──────────────────────────────────┘
```

---

## 🔹 Project Structure

```
mlops project/
├── src/
│   ├── config.py              # Configuration & Environment Variables
│   ├── preprocess.py          # CustomPreprocessor (sklearn compatible)
│   ├── train.py               # Ingestion, SMOTE, Training, and MLflow logging
│   └── evaluate.py            # Model evaluation utilities
├── preprocess_pipeline.py     # Standalone preprocessor (used by Docker)
├── pipelines/
│   └── azure_pipeline.py      # Azure ML Pipeline SDK script
├── docker/
│   ├── app.py                 # FastAPI inference service
│   └── Dockerfile             # Container image definition
├── azure-pipelines.yml        # Azure DevOps CI/CD pipeline
└── requirements.txt
```

---

## 🔹 Pipeline Steps

### 1. Data Ingestion & Preprocessing (`src/train.py`)
Downloads `churn.csv` directly from **Azure Blob Storage** into `data/raw/` at the start of the training run.
Preprocessing includes:
- Fixing `TotalCharges` (numeric coercion + median imputation)
- Binary encoding with `LabelEncoder`
- Multi-class encoding with `pd.get_dummies`
- Feature scaling with `StandardScaler`

### 2. SMOTE (`src/train.py`)
Applied **after** train/test split on the training set only, to prevent data leakage.

### 3. Model Training & Selection (`src/train.py`)
Independently trains:
- `AdaBoostClassifier(n_estimators=100)`
- `RandomForestClassifier(n_estimators=100)`

Selects the best performing model. The chosen **Random Forest** achieves **~82% accuracy** and **0.84 AUC-ROC** on the hold-out test set.

### 4. Experiment Tracking & Model Versioning
Metrics and parameters are logged to **Azure ML via MLflow**. The best model is registered in the **Azure ML Model Registry** with automatic versioning after each successful training run.

### 5. CI/CD – Azure DevOps (`azure-pipelines.yml`)
Triggered on every merge to `main`:
1. **Build** – `docker build` + push image to ACR
2. **Test** – import smoke test + local training dry-run
3. **Deploy** – deploy Docker image to ACI using native **Azure CLI** commands

---

## 🔹 Quickstart (Local)

```powershell
# Install dependencies
pip install -r requirements.txt

# Train locally (using data/raw/churn.csv — no Azure credentials needed)
python src/train.py --local

# Start the API
uvicorn docker.app:app --port 8000

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": {"customerID": "0000-AAAAA", "gender": "Female", ...}}'
```

---

## 🔹 Required Environment Variables

| Variable | Description |
|---|---|
| `AZURE_STORAGE_CONNECTION_STRING` | Blob Storage connection string |
| `AZURE_SUBSCRIPTION_ID` | Azure subscription ID |
| `AZURE_RESOURCE_GROUP` | Resource group for AML + ACI |
| `AZURE_ML_WORKSPACE` | Azure ML workspace name |
| `ACR_NAME` | Azure Container Registry name |
| `ACR_PASSWORD` | ACR admin password |

---

## 🔹 Automated Model Promotion

The Azure DevOps pipeline enforces environment gates:
- **Dev** – runs on every PR; trains model, validates metrics
- **Production** – deploys only from `main`; requires the Test stage to pass

This ensures only validated, reproducible models reach production.
