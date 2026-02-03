import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from preprocess_pipeline import CustomPreprocessor
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/raw/churn.csv")

y = df["Churn"].map({"No": 0, "Yes": 1})
X = df.drop("Churn", axis=1)

pipeline = Pipeline(steps=[
    ("preprocess", CustomPreprocessor()),
    ("model", AdaBoostClassifier(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)

joblib.dump(pipeline, "model/churn_pipeline.pkl")

print("Full pipeline saved as churn_pipeline.pkl")
