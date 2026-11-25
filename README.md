# Experiment 5: Cluster Analysis for Market Segmentation (K-Means Algorithm)

## Aim
To apply the K-Means Clustering algorithm to segment a dataset into distinct clusters and visualize how similar data points are grouped together for market segmentation.

## Dataset
The dataset consists of 16 manually entered customer data points representing two numerical features:
* **Feature 1 ($X$):** Annual Income (in Lakhs)
* **Feature 2 ($Y$):** Spending Score (`1–100`)

| Point | Annual Income ($X$) | Spending Score ($Y$) | Point | Annual Income ($X$) | Spending Score ($Y$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 20 | 30 | **9** | 32 | 33 |
| **2** | 22 | 35 | **10** | 24 | 31 |
| **3** | 25 | 40 | **11** | 28 | 45 |
| **4** | 70 | 80 | **12** | 75 | 70 |
| **5** | 75 | 85 | **13** | 71 | 83 |
| **6** | 80 | 90 | **14** | 83 | 89 |
| **7** | 30 | 70 | **15** | 28 | 60 |
| **8** | 35 | 75 | **16** | 27 | 67 |

## Algorithm / Steps Followed
1. Import required Python libraries (`numpy`, `matplotlib`, `scikit-learn`).
2. Input data points as separate $X$ and $Y$ feature arrays.
3. Combine both variables into a single 2D feature matrix using `np.column_stack()`.
4. Initialize the `KMeans` model with $K = 3$ clusters.
5. Fit the model and assign each data point to a cluster using `fit_predict()`.
6. Retrieve cluster centroids (`cluster_centers_`) and assigned labels.
7. Visualize the clusters and their centroids using a Matplotlib scatter plot.
8. Interpret the resulting customer segments and centroid coordinates.

## Functions and Methods Used

| Category | Function / Method | Purpose / Description |
| :--- | :--- | :--- |
| **Data Handling** | `np.array()` | Creates arrays for storing numerical feature values ($X$ and $Y$) |
| | `np.column_stack()` | Combines multiple 1D arrays into a single 2D dataset for clustering |
| **Model Initialization** | `KMeans()` | Initializes the K-Means clustering model with $K$ clusters |
| **Model Training & Prediction** | `fit_predict()` | Fits the K-Means model to the dataset and assigns cluster labels |
| **Cluster Properties** | `cluster_centers_` | Retrieves the coordinates (centroids) of the cluster centers |
| | `labels_` | Stores the cluster labels assigned to each data point |
| **Visualization (Matplotlib)** | `plt.scatter()` | Plots the data points and centroids in a 2D scatter plot |
| | `plt.title()`, `plt.xlabel()`, `plt.ylabel()`, `plt.legend()`, `plt.grid()`, `plt.show()` | Configures labels, legend, gridlines, and displays the final graph |

## Program & Output
# 1. Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 2. Enter the customer dataset
annual_income = np.array([
    20, 22, 25, 70, 75, 80, 30, 35,
    32, 24, 28, 75, 71, 83, 28, 27
])

spending_score = np.array([
    30, 35, 40, 80, 85, 90, 70, 75,
    33, 31, 45, 70, 83, 89, 60, 67
])

# 3. Combine the two features into one 2D matrix
X = np.column_stack((annual_income, spending_score))

print("Customer feature matrix (Income, Spending Score):")
print(X)
print("Dataset shape:", X.shape)

# 4. Initialize and fit the K-Means model with K = 3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X)

# 5. Retrieve cluster centroids and assigned labels
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print("Cluster labels assigned to each customer:")
print(labels)

print("\\nCluster centroids (Income, Spending Score):")
for cluster_id, centroid in enumerate(centroids):
    print(
        f"Cluster {cluster_id}: "
        f"Income = {centroid[0]:.2f} Lakhs, "
        f"Spending Score = {centroid[1]:.2f}"
    )
# 6. Visualize the clusters and their centroids
plt.figure(figsize=(9, 6))

for cluster_id in range(3):
    cluster_points = X[cluster_labels == cluster_id]
    plt.scatter(
        cluster_points[:, 0],
        cluster_points[:, 1],
        s=90,
        label=f"Cluster {cluster_id}"
    )

plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    s=250,
    marker="X",
    color="black",
    edgecolor="white",
    linewidth=1.5,
    label="Centroids"
)

plt.title("K-Means Customer Segmentation (K = 3)")
plt.xlabel("Annual Income (Lakhs)")
plt.ylabel("Spending Score (1–100)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


## Result
