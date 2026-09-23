K-Means Customer Segmentation

1. Project Overview

This project implements customer segmentation using the K-Means clustering algorithm.

The objective is to group customers into meaningful segments based on:

Annual Income

Spending Score

Age

The project follows a complete machine learning workflow:

Dataset -> Data Inspection -> Data Cleaning -> Feature Scaling -> K Selection -> K-Means Training -> Cluster Analysis -> Business Recommendations
## 🚀 Live Demo

The deployed Streamlit application is available here:

👉 [Open Customer Segmentation Live Demo](https://vaishnavi-05-wq-task3-kmeans-customer-segmentation-b-app-przvys.streamlit.app/)

The live application allows users to:

- Enter Annual Income (k$)
- Enter Spending Score (1-100)
- Predict the customer's K-Means cluster
- View cluster-level customer statistics
- Understand the identified customer segment

2. Dataset

Dataset Name

Mall Customers Dataset

Dataset Source

Kaggle Mall Customers dataset.

The dataset contains customer information including:

CustomerID

Genre

Age

Annual Income (k$)

Spending Score (1-100)

The original dataset contains 200 customer records.

The dataset was checked for missing values and duplicate records before modelling.

Dataset Files

data/
├── Mall_Customers.csv
└── processed/
    ├── customer_segmentation_cleaned.csv
    ├── customer_segmentation_clustered.csv
    ├── customer_segmentation_sample.csv
    └── customer_segmentation_scaled.csv

The raw dataset is excluded from GitHub through .gitignore.

3. Technologies Used

Python

Pandas

NumPy

Scikit-learn

Matplotlib

Joblib

VS Code

Conda

Git

GitHub

4. Machine Learning Method

Algorithm

K-Means Clustering

K-Means is an unsupervised machine learning algorithm that groups data points into a predefined number of clusters.

Each customer is assigned to a cluster based on its position in the feature space used by the model.

5. Data Preprocessing

The following preprocessing steps were performed:

Loaded the Mall Customers dataset.

Inspected dataset shape, columns and data types.

Checked for missing values.

Checked for duplicate rows.

Converted numerical features to numeric data types.

Saved the cleaned dataset.

Standardized Annual Income and Spending Score using StandardScaler.

The clustering model used:

Annual Income (k$)

Spending Score (1-100)

Age was retained for cluster profiling and interpretation.

6. Selecting the Number of Clusters

K values from 2 to 10 were evaluated.

Two evaluation methods were used:

Elbow Method

Silhouette Score

The silhouette scores obtained were:

K

Silhouette Score

2

0.3213

3

0.4666

4

0.4939

5

0.5547

6

0.5399

7

0.5281

8

0.4552

9

0.4571

10

0.4432

K=5 produced the highest silhouette score among the tested values, so K=5 was selected for the final K-Means model.

The selection results are stored in:

results/k_selection_results.csv

Visualizations:

results/elbow_method.png

results/silhouette_scores.png

7. Final K-Means Model

The final model was trained using:

Number of clusters: 5
Random state: 42
n_init: 10

The trained model is saved as:

models/kmeans_customer_segmentation.joblib

The clustered dataset is saved as:

data/processed/customer_segmentation_clustered.csv

8. Cluster Results

Cluster 0 - Low Income / Low Spending

Customers: 81

Average Age: 42.72

Average Income: 55.30 k$

Average Spending Score: 49.52

This is the largest customer segment. Customers have relatively lower average income and lower spending compared with the high-income/high-spending segment.

Cluster 1 - High Income / High Spending

Customers: 39

Average Age: 32.69

Average Income: 86.54 k$

Average Spending Score: 82.13

These customers have high average income and high spending scores, representing a segment with strong purchasing activity.

Cluster 2 - Low Income / High Spending

Customers: 22

Average Age: 25.27

Average Income: 25.73 k$

Average Spending Score: 79.36

These customers have relatively low income but high spending scores, showing strong spending activity despite lower average income.

Cluster 3 - High Income / Low Spending

Customers: 35

Average Age: 41.11

Average Income: 88.20 k$

Average Spending Score: 17.11

These customers have high average income but a low spending score, representing a segment with purchasing capacity but comparatively low spending activity.

Cluster 4 - Low Income / Low Spending

Customers: 23

Average Age: 45.22

Average Income: 26.30 k$

Average Spending Score: 20.91

These customers have low average income and low spending scores. This is a smaller low-income/low-spending segment distinct from Cluster 0.

Detailed cluster information is available in:

results/cluster_profiles.csv
results/cluster_sizes.csv
results/cluster_centroids.csv
results/cluster_profile_summary.txt

9. Cluster Visualization

The final customer segmentation visualization is:

results/customer_clusters.png

The visualization shows customers according to:

Annual Income

Spending Score

Assigned K-Means cluster

10. Business Recommendations

1. Premium Customer Strategy

Cluster 1 has high average income and high spending scores.

Possible business actions:

Premium product campaigns

Personalized recommendations

Loyalty benefits

Exclusive offers

2. Customer Engagement Strategy

Cluster 3 has high average income but a low spending score.

Possible actions:

Personalized promotions

Targeted recommendations

Engagement campaigns

Offers designed to encourage additional purchases

3. Value and Retention Strategy

Clusters 0, 2 and 4 can be approached with strategies appropriate to their income and spending characteristics.

Possible actions:

Value-oriented offers

Product bundles

Discounts

Customer retention campaigns

These recommendations are business-oriented interpretations of the observed cluster characteristics.

11. Project Structure

Task3_KMeans_Customer_Segmentation/
│
├── data/
│   ├── Mall_Customers.csv
│   └── processed/
│       ├── customer_segmentation_cleaned.csv
│       ├── customer_segmentation_clustered.csv
│       ├── customer_segmentation_sample.csv
│       └── customer_segmentation_scaled.csv
│
├── models/
│   └── kmeans_customer_segmentation.joblib
│
├── results/
│   ├── cluster_centroids.csv
│   ├── cluster_profiles.csv
│   ├── cluster_profile_summary.txt
│   ├── cluster_sizes.csv
│   ├── customer_clusters.png
│   ├── elbow_method.png
│   ├── k_selection_results.csv
│   └── silhouette_scores.png
│
├── src/
│   ├── analyze_clusters.py
│   ├── eda.py
│   ├── find_best_k.py
│   ├── inspect_dataset.py
│   ├── preprocess_dataset.py
│   ├── scale_features.py
│   ├── train_kmeans.py
│   └── visualize_clusters.py
│
├── .gitignore
├── requirements.txt
└── README.md

12. How to Run the Project

Step 1: Activate the environment

conda activate avip-ml

Step 2: Install dependencies

python -m pip install -r requirements.txt

Step 3: Inspect the dataset

python .\src\inspect_dataset.py

Step 4: Preprocess the dataset

python .\src\preprocess_dataset.py

Step 5: Scale the features

python .\src\scale_features.py

Step 6: Evaluate possible K values

python .\src\find_best_k.py

Step 7: Train the final K-Means model

python .\src\train_kmeans.py

Step 8: Analyze clusters

python .\src\analyze_clusters.py

Step 9: Generate cluster visualization

python .\src\visualize_clusters.py

13. Saved Model and Inference

The trained K-Means model is saved using Joblib:

import joblib

model = joblib.load(
    "models/kmeans_customer_segmentation.joblib"
)

For new customer data, the same feature preprocessing used during training should be applied before calling the model's prediction method.

The clustering features are:

Annual Income (k$)

Spending Score (1-100)

Example:

cluster = model.predict(new_customer_features)

The input features must be prepared in the same order and scaling format used during training.

14. Results Summary

The final K-Means model uses five customer segments.

The highest silhouette score among K=2 to K=10 was obtained at K=5 with a score of 0.5547.

The resulting clusters provide different combinations of income and spending behavior that can be used for customer profiling and business-oriented segmentation.

15. AVIP Task 3 Deliverables

The project includes:

Dataset source documentation

Cleaned dataset

Cleaned dataset sample

K selection using elbow method and silhouette score

Final K-Means model

Clustered customer dataset

Cluster centroids CSV

Cluster sizes CSV

Cluster visualization

Cluster profiles

Business-oriented cluster summary

Three actionable business recommendations

Saved model and inference instructions

Python source scripts

16. Author

Vaishnavi

AI/ML Engineering Internship Project

Arithmatrix Virtual Internship Program (AVIP) 2026