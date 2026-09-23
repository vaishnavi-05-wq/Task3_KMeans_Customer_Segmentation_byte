from pathlib import Path
import pandas as pd


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Input dataset
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_segmentation_clustered.csv"
)

# Output file
OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "cluster_profiles.csv"
)


# Load clustered dataset
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print(f"Rows: {len(df)}")


# Calculate cluster statistics
cluster_profiles = (
    df.groupby("Cluster")
    .agg(
        Customer_Count=("CustomerID", "count"),
        Average_Age=("Age", "mean"),
        Average_Income=("Annual Income (k$)", "mean"),
        Average_Spending_Score=(
            "Spending Score (1-100)",
            "mean"
        )
    )
    .reset_index()
)


# Round values
cluster_profiles["Average_Age"] = (
    cluster_profiles["Average_Age"].round(2)
)

cluster_profiles["Average_Income"] = (
    cluster_profiles["Average_Income"].round(2)
)

cluster_profiles["Average_Spending_Score"] = (
    cluster_profiles["Average_Spending_Score"].round(2)
)


# Overall medians
income_median = df["Annual Income (k$)"].median()
spending_median = df["Spending Score (1-100)"].median()


# Income classification
def get_income_level(income):
    if income >= income_median:
        return "High Income"
    else:
        return "Low Income"


# Spending classification
def get_spending_level(score):
    if score >= spending_median:
        return "High Spending"
    else:
        return "Low Spending"


cluster_profiles["Income_Level"] = (
    cluster_profiles["Average_Income"]
    .apply(get_income_level)
)

cluster_profiles["Spending_Level"] = (
    cluster_profiles["Average_Spending_Score"]
    .apply(get_spending_level)
)


# Business-oriented profile
cluster_profiles["Customer_Profile"] = (
    cluster_profiles["Income_Level"]
    + " / "
    + cluster_profiles["Spending_Level"]
)


# Save CSV
cluster_profiles.to_csv(
    OUTPUT_PATH,
    index=False
)


# Display result
print()
print("========== CUSTOMER CLUSTER PROFILES ==========")
print()
print(cluster_profiles.to_string(index=False))

print()
print("Income median:", income_median)
print("Spending score median:", spending_median)

print()
print("SUCCESS!")
print("Profile file saved at:")
print(OUTPUT_PATH)