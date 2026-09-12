"""Create business-readable profiles for discovered buyer clusters."""

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "outputs" / "cluster_results.csv"
OUTPUT_PATH = BASE_DIR / "outputs" / "investment_profiles.csv"


def profile_clusters(df):
    profiles = []

    for cluster, group in df.groupby("cluster"):
        investment_rate = (
            group["acquisition_purpose"].eq("Investment").mean() * 100
        )
        corporate_rate = (
            group["client_type"].eq("Corporate").mean() * 100
        )
        loan_rate = (
            group["loan_applied"].eq("Yes").mean() * 100
        )
        avg_age = group["age"].mean()
        avg_satisfaction = group["satisfaction_score"].mean()

        if investment_rate >= 65:
            profile = "Investment-focused buyers"
        elif corporate_rate >= 40:
            profile = "Corporate-oriented buyers"
        elif loan_rate >= 65:
            profile = "Financing-oriented buyers"
        else:
            profile = "Personal-use / mixed buyers"

        profiles.append({
            "cluster": cluster,
            "profile_name": profile,
            "buyer_count": len(group),
            "average_age": round(avg_age, 1),
            "investment_buyer_percentage": round(investment_rate, 1),
            "corporate_buyer_percentage": round(corporate_rate, 1),
            "loan_application_percentage": round(loan_rate, 1),
            "average_satisfaction": round(avg_satisfaction, 2),
            "top_region": group["region"].mode().iloc[0],
            "top_country": group["country"].mode().iloc[0],
            "top_referral_channel": group["referral_channel"].mode().iloc[0],
        })

    return pd.DataFrame(profiles)


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    profiles = profile_clusters(df)
    profiles.to_csv(OUTPUT_PATH, index=False)
    print(f"Investment profiles saved to: {OUTPUT_PATH}")
    print(profiles.to_string(index=False))
