"""Exploratory Data Analysis and visualization."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "outputs" / "cleaned_buyer_data.csv"
FIG_DIR = BASE_DIR / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def save_countplot(df, column, filename, title):
    plt.figure(figsize=(9, 5))
    sns.countplot(data=df, x=column)
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=150)
    plt.close()


def run_eda(path=DATA_PATH):
    df = pd.read_csv(path)

    save_countplot(
        df, "client_type", "client_type.png",
        "Buyer Type Distribution"
    )
    save_countplot(
        df, "acquisition_purpose", "acquisition_purpose.png",
        "Acquisition Purpose"
    )
    save_countplot(
        df, "loan_applied", "loan_applied.png",
        "Loan Application Distribution"
    )
    save_countplot(
        df, "region", "region.png",
        "Regional Buyer Distribution"
    )
    save_countplot(
        df, "referral_channel", "referral_channel.png",
        "Customer Acquisition Channels"
    )

    plt.figure(figsize=(9, 5))
    sns.histplot(df["age"], bins=12, kde=True)
    plt.title("Buyer Age Distribution")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "age_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.histplot(df["satisfaction_score"], bins=10, discrete=True)
    plt.title("Satisfaction Score Distribution")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "satisfaction_distribution.png", dpi=150)
    plt.close()

    print(f"EDA figures saved in: {FIG_DIR}")


if __name__ == "__main__":
    run_eda()
