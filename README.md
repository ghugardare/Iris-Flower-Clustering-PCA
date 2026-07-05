# Iris Flower Clustering & Dimensionality Reduction (Week 3)

This repository features an **Unsupervised Machine Learning Project** focused on clustering the classic Iris dataset. The project demonstrates the complete process of partitioning unlabeled continuous data using **K-Means Clustering** and optimizing computational dimensions via **PCA (Principal Component Analysis)**.

---

## Project Overview
The objective of this project is twofold:
1. To group Iris flowers into natural clusters based on physical dimensions (Sepal/Petal lengths and widths) without using predefined labels.
2. To compress the 4-dimensional dataset into a 2D space for optimal interpretation and mathematical visualization.

---

## Repository Architecture & Dual-Notebook Structure
To maintain professional separation of concerns, the implementation is divided into two target execution files:

1. **`iris_kmeans_clustering.ipynb`**: Handles dataset synthesis, clusters constraint definition via the Elbow Curve, mathematical execution of the K-Means algorithm, and centroid diagnostics.
2. **`iris_pca_dimensionality_reduction.ipynb`**: Transforms the feature space using orthogonal linear projections to lower dimensional constraints and computes geometric preservation scores.
3. **`iris_dataset.csv`**: The clean source data extracted from the environmental pipeline for reproducibility.

---

## Phase 1: Unsupervised Partitioning (K-Means Clustering)
* **The Elbow Method**: Implemented the within-cluster sum of squares (Inertia) verification across multiple hyperparameter settings ($K = 1 \text{ to } 10$). The continuous mathematical drop-rate confirmed that **$K=3$** is the mathematically optimal cluster matrix.
* **Model Parameters**: Configured Scikit-Learn's `KMeans` with explicit boundaries (`n_init=10`, `random_state=42`) to group the features into 3 distinct taxonomic variations.
* **Model Serialization**: Persisted the final fit parameters as a binary artifact (`kmeans_iris_model.pkl`) using `joblib` for immediate inference deployment.

---

## Phase 2: Dimensionality Reduction (PCA)
Because visualizing data in a 4-dimensional space is impossible, **Principal Component Analysis (PCA)** was used to collapse the structure down to 2 dimensions ($PC_1$ and $PC_2$).

### Mathematical Variance Preservation:
| Feature Transformation component | Explained Variance Ratio (%) |
| :--- | :--- |
| **Principal Component 1 (PC1)** | **92.46%** |
| **Principal Component 2 (PC2)** | **5.31%** |
| **Total Preserved Information** | **97.77%** |

> **Key Insight:** Even after eliminating half of the dimensions (reducing from 4D to 2D), PCA successfully retained **97.77% of the original information/variance** of the dataset!

---

## Visualizations and Diagnostics
* **Clustering Analysis**: Visualized continuous cluster boundaries along with their absolute coordinates via custom **Matplotlib** scatter mappings.
* **Centroid Tracking**: Plotted exact dynamic coordinate averages (represented as a **Red 'X'** marker) to track cluster convergence behaviors.

---

## 🛠️ Technology Stack Used
* **Core Language:** Python
* **Data Management:** Pandas, NumPy
* **Scientific Machine Learning:** Scikit-Learn (`cluster.KMeans`, `decomposition.PCA`)
* **Visualization Engine:** Matplotlib
