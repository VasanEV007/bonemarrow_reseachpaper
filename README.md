# Cascade-Based Input-Doubling Classifier & Linear Regression Pipeline
### Predicting Survival in Allogeneic Bone Marrow Transplants: Small Data Case

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.3%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 1. Project Overview & Research Context

This repository implements a complete, production-ready machine learning framework for predicting post-transplant survival (`survival_status`: 0 = Survived, 1 = Deceased) in pediatric allogeneic bone marrow transplantation (BMT), based on the methodology established in:

> **Izonin, I., Tkachenko, R., Hovdysh, N., Berezsky, O., Yemets, K., & Tsmots, I. (2025).**
> *Cascade-Based Input-Doubling Classifier for Predicting Survival in Allogeneic Bone Marrow Transplants: Small Data Case.*
> **Computation**, 13(3), 80. [MDPI Open Access].

In clinical oncology and transplantology, datasets are often strictly limited in size ("Small Data Problem") and exhibit marked class imbalance. Standard deep networks and complex non-linear classifiers frequently overfit or fail to generalize under these constraints.

To resolve these limitations, this project implements **Algorithm 1** from the research paper: a multi-stage **Cascade-Based Input-Doubling Classifier** with a **Probabilistic Neural Network (PNN)** probability expansion layer, paired with a **Random Forest meta-classifier**, and benchmarked against classical **Ordinary Least Squares (OLS) Linear Regression** with **100-iteration bootstrapping**.

---

## 2. Pipeline Architecture

```text
+-------------------------------------------------------------------------------+
|                       RAW CLINICAL BMT DATASETS                               |
|        remaining_dataset.csv (157 samples)   |  fully_filled_30.csv (30 test) |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
|              STEP A: ARTIFICIAL DATA CORRUPTION ("DIRTY DATA")                |
|  - Random 10-15% NaN missingness injection in numerical features              |
|  - Synthetic duplicate row generation                                         |
|  - Non-standard string casing & whitespace injection in Disease               |
|  - Extreme numerical outliers inserted (e.g. CD34, weights, ages)             |
|  --> Exports: dirty_dataset.csv                                               |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
|                 STEP B: DATA CLEANING & PREPROCESSING PIPELINE                |
|  - Strict isolation: Fitted on Train, Applied to Test (No Data Leakage)       |
|  - Deduplication (.drop_duplicates())                                         |
|  - Outlier Capping via IQR (Interquartile Range) Clipping [Q1-3*IQR, Q3+3*IQR]|
|  - Numerical Median Imputation & Categorical Mode Imputation                  |
|  - Standardization (StandardScaler) & Categorical One-Hot Encoding            |
|  --> Exports: cleaned_dataset.csv                                             |
+-------------------------------------------------------------------------------+
                 |                                              |
                 v                                              v
+-------------------------------------+   +-------------------------------------+
|   MODEL 1: OLS LINEAR REGRESSION    |   | MODEL 2: CASCADE INPUT-DOUBLING     |
|   (Baseline Diagnostic Modeling)    |   | (Algorithm 1, Izonin et al. 2025)   |
|-------------------------------------|   |-------------------------------------|
| - Ordinary Least Squares (OLS) Fit  |   | Stage 1: PNN Probability Expansion  |
| - 100-Iteration Bootstrapping       |   | - Distance: Euclidean, Chebyshev,   |
| - Parameter Standard Errors & 95% CI|   |   Cosine to support vectors         |
| - Residual Diagnostics & Error Tabs |   | - Gaussian Kernel (Smooth factor σ) |
| - Thresholded Binary Classification |   | - Augments space: [x, P_E, P_C, P_Cos]|
|                                     |   | Stage 2: Quadratic Input-Doubling   |
|                                     |   | - Sample pairs (x_i, x_j) concatenated|
|                                     |   | - Target difference z = y_i - y_j   |
|                                     |   | Stage 3: Random Forest Meta-Classifier|
|                                     |   | - Trained on doubled feature space  |
|                                     |   | - Plurality Voting across references|
|                                     |   | Hyperparameter Optimization:        |
|                                     |   | - Dual Annealing for σ, n_est, depth|
+-------------------------------------+   +-------------------------------------+
                 \                                              /
                  \                                            /
                   v                                          v
+-------------------------------------------------------------------------------+
|                         COMPREHENSIVE BENCHMARK EVALUATION                    |
|  - Performance Metrics: Accuracy, Precision, Recall, F1, MCC, Cohen's Kappa   |
|  - Error Metrics: RMSE, MAE, Residual Error Distributions                     |
|  - Diagnostic Plots: Missingness Maps, Correlation Heatmaps, 3D Scatters,     |
|    Side-by-side Confusion Matrices, ROC-AUC & Precision-Recall Curves,         |
|    Linear Regression with 95% Bootstrapped Confidence Interval Bands          |
+-------------------------------------------------------------------------------+
```

---

## 3. Repository Structure

```text
.
├── src/
│   ├── __init__.py                     # Package initialization
│   ├── data_corruption.py              # Artificial noise & corruption generator
│   ├── data_cleaning.py                # Preprocessing, imputation, IQR capping, scaling
│   ├── linear_regression.py            # Baseline OLS & 100-iteration bootstrapping
│   ├── cascade_classifier.py           # Algorithm 1: PNN expansion, doubling, RF, voting
│   ├── evaluation.py                   # Metrics calculation & formatted markdown tables
│   └── visualizations.py               # Publication-grade plotting routines
├── data/
│   ├── remaining_dataset.csv           # Original raw training dataset (157 rows)
│   ├── fully_filled_30.csv             # Original clean test dataset (30 rows)
│   ├── dirty_dataset.csv               # Artificially corrupted dataset
│   ├── cleaned_dataset.csv             # Cleaned training dataset
│   └── cleaned_test_dataset.csv        # Preprocessed test dataset
├── outputs/
│   ├── final_predictions.csv           # Predictions per sample with match status
│   ├── model_metrics.csv               # Benchmark comparison across all 9 metrics
│   ├── confusion_matrix_metrics.csv    # TP, FP, TN, FN, Sensitivity, Specificity, PPV, NPV
│   ├── regression_errors.csv           # OLS residuals, absolute errors, squared errors
│   ├── bootstrapped_coefficients.csv   # OLS coefficients, std errors, 95% CI, p-values
│   ├── hyperparameter_tuning_results.csv # Dual Annealing trial history
│   └── feature_importance.csv          # Random Forest feature importance ranking
├── plots/
│   ├── missing_values_dirty_vs_clean.png # Missingness heatmaps before vs after cleaning
│   ├── eda_feature_distributions.png     # Histograms & KDE for clinical features
│   ├── eda_correlation_heatmap.png       # Correlation matrix of clinical attributes
│   ├── 3d_scatter_features.png           # 3D feature space (Donor/Recipient Age vs CD34)
│   ├── confusion_matrix_comparison.png   # Side-by-side confusion matrices
│   ├── roc_pr_curves.png                 # ROC-AUC and Precision-Recall curves
│   └── linear_regression_bootstrapped_fits.png # OLS regression with 95% bootstrap band
├── pipeline.py                         # Master executable pipeline script
├── build_notebook.py                   # Generator for reproducible Jupyter Notebook
├── bone_marrow_cascade_pipeline.ipynb  # Interactive, fully executed Jupyter Notebook
└── README.md                           # Comprehensive documentation
```

---

## 4. Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.10, 3.11, 3.12, 3.14)
- Recommended: Dedicated virtual environment (`venv` or `conda`)

### Install Dependencies
```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn statsmodels nbformat nbconvert ipykernel
```

---

## 5. Instructions to Run

### Run the End-to-End Master Pipeline
Executes the data dirtying, cleaning, model training, hyperparameter optimization, and exports all CSVs and high-resolution figures:
```bash
python pipeline.py
```

### Run the Interactive Jupyter Notebook
```bash
jupyter notebook bone_marrow_cascade_pipeline.ipynb
```
Or rebuild and execute programmatically:
```bash
python build_notebook.py
```

---

## 6. Summary of Benchmark Results

All models were evaluated on the independent, clean test cohort (`fully_filled_30.csv`, 30 patients: 20 Survived, 10 Deceased).

### Comprehensive Model Performance Comparison

| Model Name | Accuracy | Precision | Recall | F1-Score | MCC | Cohen's Kappa | RMSE | MAE | Training Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression (OLS Baseline)** | 0.9333 | 0.8333 | **1.0000** | **0.9091** | **0.8660** | **0.8571** | 0.3186 | 0.2597 | 0.061 s |
| **Hybrid Cascade Input-Doubling (Algorithm 1)** | **0.9667** | **1.0000** | 0.9000 | **0.9474** | **0.9258** | **0.9211** | **0.1661** | **0.0276** | 0.434 s |

### Confusion Matrix Breakdown & Diagnostic Metrics

| Model | TP | FP | TN | FN | Sensitivity | Specificity | PPV (Precision) | NPV |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression (OLS)** | 10 | 2 | 18 | 0 | 1.0000 | 0.9000 | 0.8333 | 1.0000 |
| **Hybrid Cascade (Algorithm 1)** | 9 | **0** | **20** | 1 | 0.9000 | **1.0000** | **1.0000** | **0.9524** |

### Parameter Optimization Results (Algorithm 1)

| Parameter | Optimal Value | Search / Optimization Method |
| :--- | :---: | :--- |
| **Smooth factor ($\sigma$)** | `3.2183` | Dual Annealing Global Optimization (`scipy.optimize.dual_annealing`) |
| **Distance Metrics** | Euclidean, Chebyshev, Cosine | Stage 1 PNN Multi-Metric Expansion Layer |
| **Number of Estimators (`n_estimators`)** | `80` | Dual Annealing Global Optimization |
| **Maximum Tree Depth (`max_depth`)** | `9` | Dual Annealing Global Optimization |
| **Ensemble Inference Method** | Plurality Voting | Pairwise Comparison over Support References |

---

## 7. Key Findings & Insights

1. **Error Reduction**: The Cascade-Based Input-Doubling model reduced the Mean Absolute Error (MAE) from **0.2597** down to **0.0276** (an **89.4% reduction in prediction error**) and RMSE from **0.3186** to **0.1661**.
2. **Zero False Positives (100% Specificity & Precision)**: In high-stakes BMT clinical decisions, false positives can lead to unnecessary invasive interventions. The Cascade classifier achieved **0 False Positives** (Specificity = 1.0, Precision = 1.0).
3. **PNN Feature Expansion Impact**: In the feature importance ranking of the meta-classifier, `PNN_Euclid` emerged as the #1 most influential feature (importance = 35.6%), verifying the core thesis of *Izonin et al. (2025)* that neural probability projection enriches low-dimensional data spaces.
4. **Input Doubling Mitigation of Small Data**: Generating pairwise differences ($z = y_i - y_j$) effectively transforms a small dataset of $N=157$ into an augmented problem of thousands of informative vector comparisons, mitigating class imbalance and small sample variance.

---

## 8. License & Attribution
- Research Reference: Izonin et al. (2025), *Computation*, MDPI.
- Dataset: Pediatric Bone Marrow Transplantation Benchmark Dataset.
- License: MIT License.
