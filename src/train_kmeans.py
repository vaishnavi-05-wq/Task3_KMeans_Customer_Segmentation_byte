from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Input and output paths
INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_segmentation_scaled.csv"
)

CLEANED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_segmentation_cleaned.csv"
)

CLUSTERED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_segmentation_clustered.csv"
)

CENTROIDS_PATH = (
    PROJECT_ROOT
    / "results"
    / "cluster_centroids.csv"
)

CLUSTER_SIZES_PATH = (
    PROJECT_ROOT
    / "results"
    / "cluster_sizes.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "kmeans_customer_segmentation.joblib"
)


# Load scaled features
scaled_df = pd.read_csv(INPUT_PATH)

features = [
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

X = scaled_df[features]


# Final number of clusters
k = 5

print("========== FINAL K-MEANS MODEL ==========")
print(f"Selected K : {k}")


# Create K-Means model
kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)


# Train model and generate cluster labels
cluster_labels = kmeans.fit_predict(X)


# Load original cleaned dataset
cleaned_df = pd.read_csv(CLEANED_DATA_PATH)

# Add cluster labels
cleaned_df["Cluster"] = cluster_labels


# Save clustered dataset
cleaned_df.to_csv(
    CLUSTERED_DATA_PATH,
    index=False
)


# Create centroid table
centroids = pd.DataFrame(
    kmeans.cluster_centers_,
    columns=features
)

centroids.index.name = "Cluster"

centroids.to_csv(
    CENTROIDS_PATH
)


# Calculate cluster sizes
cluster_sizes = (
    cleaned_df["Cluster"]
    .value_counts()
    .sort_index()
    .reset_index()
)

cluster_sizes.columns = [
    "Cluster",
    "Customer_Count"
]

cluster_sizes.to_csv(
    CLUSTER_SIZES_PATH,
    index=False
)


# Save trained model
joblib.dump(
    kmeans,
    MODEL_PATH
)


# Display results
print("\nCluster Sizes:")
print(cluster_sizes)

print("\nCluster Centroids (Scaled):")
print(centroids)

print("\nModel inertia:")
print(kmeans.inertia_)

print("\nFiles saved successfully:")
print(CLUSTERED_DATA_PATH)
print(CENTROIDS_PATH)
print(CLUSTER_SIZES_PATH)
print(MODEL_PATH)