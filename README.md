# Iris Flower Analysis: Unsupervised Clustering, PCA & Model Optimization

This repository contains a comprehensive, two-part Machine Learning workflow using the classic Iris dataset. It covers **Unsupervised Learning (Week 3)** and advanced **Model Evaluation & Hyperparameter Tuning (Week 4)**.

---

## 📁 Repository Structure
* **`iris_kmeans_clustering.ipynb`**: Week 3 task implementing K-Means Clustering.
* **`iris_pca_dimensionality_reduction.ipynb`**: Week 3 task focusing on PCA for dimensions reduction.
* **`model_optimization_svm.ipynb`**: Week 4 task focusing on Cross-Validation and GridSearchCV tuning for SVM.
* **`model_optimization_iris_data.csv`**: Automated dataset export generated during the evaluation pipeline.
* **`best_svm_iris_model.pkl`**: The final optimized and serialized SVM model.

---

## 🧠 Part 1: Unsupervised Learning & Dimensionality Reduction

### 1. K-Means Clustering
* **Objective**: Group iris flowers into distinct structural clusters without using pre-existing labels.
* **Methodology**: Applied the K-Means algorithm to automatically segment data based on petal and sepal measurements.
* **Outcome**: Successfully grouped the flowers into operational clusters representing biological species boundaries.

### 2. Principal Component Analysis (PCA)
* **Objective**: Reduce the multi-dimensional feature space of the Iris dataset down to 2 principal components for seamless visual charting.
* **Methodology**: Computed orthogonal transformation vectors to capture maximum dataset variance.
* **Outcome**: Created clear 2D cluster visualizations, demonstrating how dimensionality reduction simplifies complex feature spaces.

---

## ⚡ Part 2: Model Evaluation & Hyperparameter Tuning 

### 1. Robust Cross-Validation
* Applied a strict **5-Fold Cross-Validation (`cross_val_score`)** using a Linear Support Vector Classifier (SVC) to establish an authentic baseline accuracy and avoid overfitting issues.

### 2. Automated Hyperparameter Optimization (`GridSearchCV`)
* Conducted an exhaustive parameter sweep to find the absolute best structural configuration for the Support Vector Machine (SVM).
* **Grid Search Workspace**:
  * `C`: `[0.1, 1, 10, 100]`
  * `gamma`: `[1, 0.1, 0.01, 0.001]`
  * `kernel`: `['rbf', 'linear']`
* **Result**: Successfully extracted the `best_estimator_` config parameters to guarantee peak prediction consistency.

### 3. Model Serialization & Production Inference
* **Persistence**: Saved the fully trained, optimized model using `joblib` as a reusable binary asset (`best_svm_iris_model.pkl`).
* **Inference Pipeline**: Successfully reloaded the production binary to execute real-time class predictions on completely unseen dummy feature matrices.

---

## 🛠️ Technical Stack Used
* **Language**: Python
* **Data Processing**: Pandas, NumPy
* **Machine Learning Engine**: Scikit-Learn
* **Model Serialization**: Joblib
