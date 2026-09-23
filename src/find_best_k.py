from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_segmentation_scaled.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load scaled data
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

X = df[
    [
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
]


# --------------------------------------------------
# 3. Test different K values
# --------------------------------------------------

k_values = range(2, 11)

inertia_values = []
silhouette_values = []


print("========== K-MEANS K SELECTION ==========")

for k in k_values:

    # Create K-Means model
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    # Train model
    labels = kmeans.fit_predict(X)

    # Elbow metric
    inertia_values.append(kmeans.inertia_)

    # Silhouette metric
    score = silhouette_score(X, labels)

    silhouette_values.append(score)

    print(
        f"K = {k} | "
        f"Inertia = {kmeans.inertia_:.2f} | "
        f"Silhouette Score = {score:.4f}"
    )


# --------------------------------------------------
# 4. Find highest silhouette score
# --------------------------------------------------

best_index = silhouette_values.index(
    max(silhouette_values)
)

best_k = list(k_values)[best_index]

print("\nBest K according to highest silhouette score:")
print(f"K = {best_k}")


# --------------------------------------------------
# 5. Elbow plot
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.plot(
    list(k_values),
    inertia_values,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for K-Means")

plt.grid(alpha=0.3)

elbow_path = RESULTS_DIR / "elbow_method.png"

plt.savefig(
    elbow_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# --------------------------------------------------
# 6. Silhouette plot
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.plot(
    list(k_values),
    silhouette_values,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for K-Means")

plt.grid(alpha=0.3)

silhouette_path = RESULTS_DIR / "silhouette_scores.png"

plt.savefig(
    silhouette_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# --------------------------------------------------
# 7. Save K evaluation results
# --------------------------------------------------

results_df = pd.DataFrame(
    {
        "K": list(k_values),
        "Inertia": inertia_values,
        "Silhouette Score": silhouette_values
    }
)

results_path = RESULTS_DIR / "k_selection_results.csv"

results_df.to_csv(
    results_path,
    index=False
)


print("\nSaved files:")
print(elbow_path)
print(silhouette_path)
print(results_path)
