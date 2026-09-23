from pathlib import Path
import pandas as pd


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "Mall_Customers.csv"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_PATH = PROCESSED_DIR / "customer_segmentation_cleaned.csv"

SAMPLE_PATH = PROCESSED_DIR / "customer_segmentation_sample.csv"


# --------------------------------------------------
# 2. Create processed-data folder if needed
# --------------------------------------------------

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 3. Load original dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)


print("========== DATA PREPROCESSING ==========")

print(f"\nOriginal rows    : {len(df)}")
print(f"Original columns : {len(df.columns)}")


# --------------------------------------------------
# 4. Keep the columns required for this project
# --------------------------------------------------

required_columns = [
    "CustomerID",
    "Genre",
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

df = df[required_columns].copy()


# --------------------------------------------------
# 5. Check missing values
# --------------------------------------------------

missing_values = df.isnull().sum().sum()

print(f"\nMissing values: {missing_values}")


# --------------------------------------------------
# 6. Remove rows with missing values if any
# --------------------------------------------------

df = df.dropna().copy()


# --------------------------------------------------
# 7. Remove duplicate rows if any
# --------------------------------------------------

duplicates = df.duplicated().sum()

print(f"Duplicate rows found: {duplicates}")

df = df.drop_duplicates().copy()


# --------------------------------------------------
# 8. Validate numerical columns
# --------------------------------------------------

numeric_columns = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")


# Remove rows that became invalid after conversion
df = df.dropna(subset=numeric_columns).copy()


# --------------------------------------------------
# 9. Save cleaned/used dataset
# --------------------------------------------------

df.to_csv(OUTPUT_PATH, index=False)


# --------------------------------------------------
# 10. Save a sample for the AVIP deliverable
# --------------------------------------------------

sample = df.head(20)

sample.to_csv(SAMPLE_PATH, index=False)


# --------------------------------------------------
# 11. Display final information
# --------------------------------------------------

print("\n========== FINAL DATA ==========")

print(f"Final rows    : {len(df)}")
print(f"Final columns : {len(df.columns)}")

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 cleaned rows:")
print(df.head())

print("\nSaved cleaned dataset:")
print(OUTPUT_PATH)

print("\nSaved dataset sample:")
print(SAMPLE_PATH)