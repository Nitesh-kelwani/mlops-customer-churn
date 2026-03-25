"""
train.py – Train ensemble model (AdaBoost + Random Forest) with SMOTE.
Logs metrics to Azure ML via MLflow and registers the model.

Usage:
    python src/train.py            # uses Azure Blob ingestion (if configured)
    python src/train.py --local    # skips Blob download, uses data/raw/churn.csv
"""

import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from preprocess_pipeline import CustomPreprocessor
from src.config import (
    DATA_PATH, MODEL_PATH, TARGET_COL,
    SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME,
    MODEL_NAME, EXPERIMENT_NAME,
)


def get_mlflow_tracking_uri() -> str:
    """Return Azure ML MLflow tracking URI if credentials present, else local."""
    if SUBSCRIPTION_ID:
        try:
            from azure.ai.ml import MLClient
            from azure.identity import DefaultAzureCredential
            ml_client = MLClient(
                DefaultAzureCredential(), SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME
            )
            return ml_client.workspaces.get(WORKSPACE_NAME).mlflow_tracking_uri
        except Exception as exc:
            print(f"[train] Could not connect to Azure ML: {exc}. Using local MLflow tracking.")
    return "mlruns"


def register_model_azure(run_id: str, model_path: str) -> None:
    """Register the model in the Azure ML Model Registry after a successful run."""
    if not SUBSCRIPTION_ID:
        print("[train] Skipping Azure ML model registration (no subscription configured).")
        return
    try:
        from azure.ai.ml import MLClient
        from azure.ai.ml.entities import Model
        from azure.ai.ml.constants import AssetTypes
        from azure.identity import DefaultAzureCredential

        ml_client = MLClient(
            DefaultAzureCredential(), SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME
        )
        model = Model(
            path=model_path,
            name=MODEL_NAME,
            description="Ensemble (AdaBoost + Random Forest) churn prediction model",
            type=AssetTypes.CUSTOM_MODEL,
        )
        registered = ml_client.models.create_or_update(model)
        print(f"[train] Model registered: {registered.name} v{registered.version}")
    except Exception as exc:
        print(f"[train] Model registration failed: {exc}")


def train_model(local: bool = False) -> None:
    # ── Data ingestion ────────────────────────────────────────────────────────
    if not local:
        from src.ingest import download_from_blob
        download_from_blob()

    df = pd.read_csv(DATA_PATH)

    y = df[TARGET_COL].map({"No": 0, "Yes": 1})
    X = df.drop(TARGET_COL, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Preprocessing (fit only on train) ─────────────────────────────────────
    preprocessor = CustomPreprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)

    # ── SMOTE (applied after preprocessing) ───────────────────────────────────
    print("[train] Applying SMOTE to balance classes …")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_proc, y_train)

    # ── Train Individual Models ───────────────────────────────────────────────
    adaboost = AdaBoostClassifier(n_estimators=100, random_state=42)
    rf       = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    models = {
        "AdaBoost": adaboost,
        "RandomForest": rf
    }
    
    best_model_name = None
    best_model = None
    best_auc = 0.0
    best_acc = 0.0

    # ── MLflow tracking ───────────────────────────────────────────────────────
    tracking_uri = get_mlflow_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        print("[train] Training AdaBoost and Random Forest to select the best …")
        for name, model in models.items():
            model.fit(X_train_res, y_train_res)
            
            preds = model.predict(X_test_proc)
            proba = model.predict_proba(X_test_proc)[:, 1]
            acc   = accuracy_score(y_test, preds)
            auc   = roc_auc_score(y_test, proba)
            
            print(f"[train] {name} - Accuracy: {acc:.4f}, AUC-ROC: {auc:.4f}")
            
            if auc > best_auc:
                best_auc = auc
                best_acc = acc
                best_model = model
                best_model_name = name

        print(f"\n[train] Best Model: {best_model_name}")
        print(f"[train] Best Accuracy: {best_acc:.4f}")
        print(f"[train] Best AUC-ROC : {best_auc:.4f}")

        # Log hyper-params
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("smote_enabled", True)

        # Log metrics
        mlflow.log_metric("accuracy", best_acc)
        mlflow.log_metric("auc_roc", best_auc)

        # Save full pipeline (preprocessor + best model)
        full_pipeline = {"preprocessor": preprocessor, "model": best_model}
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(full_pipeline, MODEL_PATH)

        mlflow.sklearn.log_model(best_model, artifact_path="best_model")
        print(f"[train] Model saved -> {MODEL_PATH}")

        run_id = run.info.run_id

    # ── Register in Azure ML ───────────────────────────────────────────────────
    register_model_azure(run_id, MODEL_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Skip Blob download")
    args = parser.parse_args()
    train_model(local=args.local)
