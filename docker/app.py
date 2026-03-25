"""
app.py – FastAPI inference service for ACI deployment.
Endpoints:
  GET  /health   -> liveness probe
  POST /predict  -> churn prediction
"""

import sys
import os

sys.path.append(os.getcwd())

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Customer Churn Prediction API",
    description="End-to-end MLOps – Ensemble model deployed on ACI",
    version="1.0.0",
)

# Load the persisted pipeline dict {preprocessor, model}
try:
    artifact = joblib.load("model/churn_pipeline.pkl")
    preprocessor = artifact["preprocessor"]
    model        = artifact["model"]
except Exception as e:
    preprocessor = None
    model        = None
    print(f"[app] Warning: could not load model – {e}")


class ChurnInput(BaseModel):
    data: dict


@app.get("/health")
def health():
    """Liveness / readiness probe used by ACI and Azure DevOps."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(input_data: ChurnInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    df         = pd.DataFrame([input_data.data])
    df_proc    = preprocessor.transform(df)
    pred       = model.predict(df_proc)
    proba      = model.predict_proba(df_proc)[:, 1]
    return {
        "prediction": int(pred[0]),
        "churn_probability": round(float(proba[0]), 4),
    }
