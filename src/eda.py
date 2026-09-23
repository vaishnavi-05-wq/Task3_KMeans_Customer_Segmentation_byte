from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_segmentation_cleaned.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load cleaned dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)


print("========== EXPLORATORY DATA ANALYSIS ==========")

print(f"\nNumber of customers: {len(df)}")


# --------------------------------------------------
# 3. Basic statistics
# --------------------------------------------------

print("\nAnnual Income statistics:")
print(df["Annual Income (k$)"].describe())

print("\nSpending Score statistics:")
print(df["Spending Score (1-100)"].describe())


# --------------------------------------------------
# 4. Create income vs spending scatter plot
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    alpha=0.7
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Annual Income vs Spending Score")

plt.grid(alpha=0.3)

plot_path = RESULTS_DIR / "income_vs_spending.png"

plt.savefig(plot_path, dpi=300, bbox_inches="tight")

plt.show()

print(f"\nPlot saved to:")
print(plot_path)