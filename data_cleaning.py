"""Load, clean, and engineer the real-estate buyer dataset."""

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "real_estate_buyer_segmentation_dataset.csv"
OUTPUT_PATH = BASE_DIR / "outputs" / "cleaned_buyer_data.csv"


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df


def clean_data(df):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # Remove duplicate customer records.
    df = df.drop_duplicates(subset=["client_id"])

    # Parse date and calculate age.
    df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce")
    reference_date = pd.Timestamp.today().normalize()
    df["age"] = ((reference_date - df["date_of_birth"]).dt.days / 365.25).round(1)

    # Basic missing-value handling.
    categorical_cols = [
        "client_type", "gender", "country", "region",
        "acquisition_purpose", "loan_applied", "referral_channel"
    ]
    for col in categorical_cols:
        if col in df.columns:
            mode = df[col].mode(dropna=True)
            if not mode.empty:
                df[col] = df[col].fillna(mode.iloc[0])

    if "satisfaction_score" in df.columns:
        df["satisfaction_score"] = pd.to_numeric(
            df["satisfaction_score"], errors="coerce"
        )
        df["satisfaction_score"] = df["satisfaction_score"].fillna(
            df["satisfaction_score"].median()
        )

    if "age" in df.columns:
        df["age"] = df["age"].fillna(df["age"].median())

    # Normalize common text fields.
    text_cols = [
        "client_type", "gender", "country", "region",
        "acquisition_purpose", "loan_applied", "referral_channel"
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    return df


if __name__ == "__main__":
    data = clean_data(load_data())
    print(f"Cleaned dataset saved to: {OUTPUT_PATH}")
    print(f"Rows: {len(data)}, Columns: {len(data.columns)}")
    print(data.head())
