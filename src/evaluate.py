import joblib
from sklearn.metrics import accuracy_score, classification_report

from preprocess import load_data, preprocess_data
from config import DATA_PATH, MODEL_PATH, TARGET_COL

def evaluate_model():
    df = preprocess_data(load_data(DATA_PATH))

    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    model = joblib.load(MODEL_PATH)
    preds = model.predict(X)

    print("Accuracy:", accuracy_score(y, preds))
    print("\nClassification Report:\n")
    print(classification_report(y, preds))

if __name__ == "__main__":
    evaluate_model()
