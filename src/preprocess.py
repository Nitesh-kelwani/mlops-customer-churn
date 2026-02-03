import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler

class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.binary_encoders = {}
        self.scaler = StandardScaler()
        self.num_cols = None
        self.ohe_cols = None

    def fit(self, X, y=None):
        X = X.copy()

        # Fix TotalCharges
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
        X["TotalCharges"].fillna(X["TotalCharges"].median(), inplace=True)

        # Binary columns
        binary_cols = [
            col for col in X.columns
            if X[col].dtype == "object" and X[col].nunique() == 2
        ]

        for col in binary_cols:
            le = LabelEncoder()
            le.fit(X[col])
            self.binary_encoders[col] = le
            X[col] = le.transform(X[col])

        # One-hot columns
        self.ohe_cols = [
            col for col in X.columns
            if 30 >= X[col].nunique() > 2 and X[col].dtype == "object"
        ]

        X = pd.get_dummies(X, columns=self.ohe_cols, drop_first=True)

        # Numeric columns
        self.num_cols = [
            col for col in X.columns
            if col not in ["customerID"] and X[col].dtype in [int, float]
        ]

        self.scaler.fit(X[self.num_cols])

        return self

    def transform(self, X):
        X = X.copy()

        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
        X["TotalCharges"].fillna(X["TotalCharges"].median(), inplace=True)

        for col, le in self.binary_encoders.items():
            X[col] = le.transform(X[col])

        X = pd.get_dummies(X, columns=self.ohe_cols, drop_first=True)

        # Align missing columns
        for col in self.scaler.feature_names_in_:
            if col not in X.columns:
                X[col] = 0

        X = X[self.scaler.feature_names_in_]

        X[self.num_cols] = self.scaler.transform(X[self.num_cols])

        return X
