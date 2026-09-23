import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "models" / "kmeans_customer_segmentation.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "customer_segmentation_clustered.csv"


# --------------------------------------------------
# 2. Load trained model and clustered dataset
# --------------------------------------------------

model = joblib.load(MODEL_PATH)
clustered_data = pd.read_csv(DATA_PATH)


# --------------------------------------------------
# 3. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="📊",
    layout="centered"
)


# --------------------------------------------------
# 4. Application title
# --------------------------------------------------

st.title("📊 K-Means Customer Segmentation")

st.write(
    "Enter a customer's annual income and spending score "
    "to identify the corresponding customer segment."
)


# --------------------------------------------------
# 5. User inputs
# --------------------------------------------------

annual_income = st.number_input(
    "Annual Income (k$)",
    min_value=0.0,
    max_value=200.0,
    value=50.0,
    step=1.0
)

spending_score = st.number_input(
    "Spending Score (1-100)",
    min_value=1.0,
    max_value=100.0,
    value=50.0,
    step=1.0
)


# --------------------------------------------------
# 6. Prediction button
# --------------------------------------------------

if st.button("Predict Customer Segment"):

    input_data = pd.DataFrame(
        [[annual_income, spending_score]],
        columns=[
            "Annual Income (k$)",
            "Spending Score (1-100)"
        ]
    )

    cluster = model.predict(input_data)[0]

    cluster_profile = clustered_data[
        clustered_data["Cluster"] == cluster
    ]

    st.success(f"Customer belongs to Cluster {cluster}")

    st.subheader("Customer Segment Details")

    if not cluster_profile.empty:

        average_age = cluster_profile["Age"].mean()
        average_income = cluster_profile["Annual Income (k$)"].mean()
        average_spending = cluster_profile[
            "Spending Score (1-100)"
        ].mean()

        customer_count = len(cluster_profile)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Customers in Cluster", customer_count)
            st.metric("Average Age", f"{average_age:.1f}")

        with col2:
            st.metric(
                "Average Income",
                f"{average_income:.1f} k$"
            )

            st.metric(
                "Average Spending Score",
                f"{average_spending:.1f}"
            )

        st.info(
            f"This segment contains {customer_count} customers "
            f"with an average income of {average_income:.1f} k$ "
            f"and an average spending score of "
            f"{average_spending:.1f}."
        )


# --------------------------------------------------
# 7. Project information
# --------------------------------------------------

st.divider()

st.caption(
    "Model: K-Means clustering | "
    "Features: Annual Income and Spending Score | "
    "K = 5"
)