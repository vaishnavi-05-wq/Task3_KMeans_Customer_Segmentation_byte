from pathlib import Path
import pandas as pd


# Project root folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Dataset path
DATASET_PATH = PROJECT_ROOT / "data" / "Mall_Customers.csv"


# Load dataset
df = pd.read_csv(DATASET_PATH)


print("========== DATASET INSPECTION ==========")

# Dataset shape
print("\n1. Dataset Shape:")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")


# Column names
print("\n2. Column Names:")
print(df.columns.tolist())


# Data types
print("\n3. Data Types:")
print(df.dtypes)


# Missing values
print("\n4. Missing Values:")
print(df.isnull().sum())


# Duplicate rows
print("\n5. Duplicate Rows:")
print(df.duplicated().sum())


# First 5 rows
print("\n6. First 5 Rows:")
print(df.head())


# Statistical summary
print("\n7. Statistical Summary:")
print(df.describe())