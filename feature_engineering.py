"""Prepare features for machine learning."""

from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "outputs" / "cleaned_buyer_data.csv"


def build_preprocessor(df):
    numeric_features = ["age", "satisfaction_score"]
    categorical_features = [
        "client_type", "gender", "country", "region",
        "acquisition_purpose", "loan_applied", "referral_channel"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             categorical_features),
        ],
        remainder="drop"
    )
    return preprocessor, numeric_features, categorical_features


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    preprocessor, _, _ = build_preprocessor(df)
    X = preprocessor.fit_transform(df)
    print("Feature matrix shape:", X.shape)
