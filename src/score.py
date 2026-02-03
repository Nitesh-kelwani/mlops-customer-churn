
import json
import joblib
import numpy as np
import os

MODEL_PATH = os.path.join("model", "churn_model.pkl")

def init():
    global model
    model = joblib.load(MODEL_PATH)

def run(raw_data):
    try:
        data = json.loads(raw_data)

        # Convert input to numpy array
        input_data = np.array(data["data"])

        preds = model.predict(input_data)

        return {"prediction": preds.tolist()}
    except Exception as e:
        return {"error": str(e)}
