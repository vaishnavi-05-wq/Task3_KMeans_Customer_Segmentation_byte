from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_segmentation_cleaned.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_segmentation_scaled.csv"
)


# --------------------------------------------------
# 2. Load cleaned dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)


# --------------------------------------------------
# 3. Select K-Means features
# --------------------------------------------------

features = [
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

X = df[features].copy()


print("========== FEATURE SCALING ==========")

print("\nOriginal features:")
print(X.head())


# --------------------------------------------------
# 4. Create StandardScaler
# --------------------------------------------------

scaler = StandardScaler()


# --------------------------------------------------
# 5. Fit and transform features
# --------------------------------------------------

X_scaled = scaler.fit_transform(X)


# --------------------------------------------------
# 6. Convert scaled data back to DataFrame
# --------------------------------------------------

scaled_df = pd.DataFrame(
    X_scaled,
    columns=features
)


# --------------------------------------------------
# 7. Save scaled dataset
# --------------------------------------------------

scaled_df.to_csv(OUTPUT_PATH, index=False)


# --------------------------------------------------
# 8. Display results
# --------------------------------------------------

print("\nScaled features:")
print(scaled_df.head())

print("\nScaled feature means:")
print(scaled_df.mean())

print("\nScaled feature standard deviations:")
print(scaled_df.std())

print("\nSaved scaled dataset:")
print(OUTPUT_PATH)