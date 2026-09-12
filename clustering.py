"""K-Means buyer segmentation."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "outputs" / "cleaned_buyer_data.csv"
RESULT_PATH = BASE_DIR / "outputs" / "cluster_results.csv"
FIG_DIR = BASE_DIR / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def prepare_features(df):
    numeric = ["age", "satisfaction_score"]
    categorical = [
        "client_type", "gender", "country", "region",
        "acquisition_purpose", "loan_applied", "referral_channel"
    ]

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ]
    )
    X = preprocessor.fit_transform(df)
    return X


def find_best_k(X, min_k=2, max_k=8):
    rows = []
    for k in range(min_k, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        rows.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette_score": silhouette_score(X, labels)
        })
    return pd.DataFrame(rows)


def run_clustering():
    df = pd.read_csv(DATA_PATH)
    X = prepare_features(df)

    evaluation = find_best_k(X)
    evaluation.to_csv(BASE_DIR / "outputs" / "cluster_evaluation.csv", index=False)

    # Select the best silhouette score as a simple, reproducible baseline.
    best_k = int(evaluation.loc[
        evaluation["silhouette_score"].idxmax(), "k"
    ])

    model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["cluster"] = model.fit_predict(X)

    # 2D visualization with PCA.
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    plt.figure(figsize=(9, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=df["cluster"], alpha=0.8)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(f"Buyer Segments using K-Means (K={best_k})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "buyer_clusters_pca.png", dpi=150)
    plt.close()

    df.to_csv(RESULT_PATH, index=False)

    print(f"Best K: {best_k}")
    print(f"Results saved to: {RESULT_PATH}")
    print(evaluation.to_string(index=False))


if __name__ == "__main__":
    run_clustering()
