import sys
import os

sys.path.append(os.getcwd()) 
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Churn Prediction API")

pipeline = joblib.load("model/churn_pipeline.pkl")

class ChurnInput(BaseModel):
    data: dict

@app.post("/predict")
def predict(input_data: ChurnInput):
    df = pd.DataFrame([input_data.data])
    pred = pipeline.predict(df)
    return {"prediction": pred.tolist()}
