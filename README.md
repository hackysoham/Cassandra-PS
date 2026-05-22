# Cassandra Veritas — Auto Insurance Claim Prediction

A complete end-to-end machine learning pipeline for predicting auto insurance claim probability using anonymized driver and vehicle data from Veritas Risk Corp. The project combines exploratory data analysis, structural discovery, representation learning, feature engineering, and ensemble-based predictive modeling optimized using the **Normalized Gini Coefficient**.

---

## Overview

This project was developed as part of **Cassandra Veritas — Signals Under the Surface** under the Electronics Engineering Society, IIT BHU.

The workflow is divided into three major parts:

- **Part A — Data Investigation**
  - Feature understanding
  - Target imbalance analysis
  - Missing value analysis
  - Correlation studies

- **Part B — Structural Discovery**
  - PCA & UMAP representation learning
  - Feature correlation networks
  - Driver risk clustering
  - SHAP interaction analysis
  - Synthetic data generation pipeline

- **Part C — Predictive Modeling**
  - Feature engineering
  - Gradient boosting models
  - 5-Fold stratified cross-validation
  - Optuna-optimized ensemble

---

## Dataset Characteristics

- **58 predictive features**
- Mixed feature types:
  - Binary
  - Categorical
  - Continuous

### Feature Groups

- `ps_ind` → Individual driver attributes
- `ps_reg` → Regional information
- `ps_car` → Vehicle-related attributes
- `ps_calc` → Engineered/calculated features

### Class Distribution

- **96.4%** → No Claim
- **3.6%** → Claim Filed

Class ratio: **26.7 : 1**

---

## Key Findings

### Missing Values Carry Predictive Signal

- Missing values encoded as `-1`
- Converted to `NaN` during preprocessing
- Missingness indicator flags added as features
- Certain missing patterns strongly correlated with claim probability

---

### Representation Learning

#### PCA

- ~35 principal components required for 90% variance retention
- Dataset not linearly separable

#### UMAP

UMAP embeddings revealed distinct driver risk clusters and non-linear structure within the data.

---

### Risk Clustering

K-Means clustering identified multiple driver segments:

| Cluster | Risk Level |
|---|---|
| C0 | Low Risk |
| C1 | Moderate Risk |
| C2 | High Risk |
| C3 | Anomalous High Risk |

Applications:
- Premium pricing
- Fraud detection
- Underwriting strategies

---

## Feature Engineering

Engineered features included:

- Missingness indicator flags
- Group-level aggregate statistics
- Row-wise missing counts
- Binary feature sums
- Pairwise interaction terms

Final feature count: **~85 features**

---

## Models Used

### 1. LightGBM
- Fast and memory efficient
- Custom Gini evaluation
- Handles imbalanced tabular data effectively

### 2. XGBoost
- Strong regularization
- Robust bias-variance tradeoff

### 3. CatBoost
- Native categorical feature handling
- Reduced target leakage
- Best standalone performance

Training strategy:
- 5-Fold Stratified Cross Validation
- Early stopping
- Class weighting using `scale_pos_weight`

---

## Ensemble Strategy

Predictions from all three models were combined using an **Optuna-optimized weighted ensemble**.

### Final OOF Gini Scores

| Model | OOF Gini |
|---|---|
| LightGBM | 0.23539 |
| XGBoost | 0.23971 |
| CatBoost | 0.25730 |
| **Ensemble** | **0.25859** |

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- XGBoost
- CatBoost
- Optuna
- SHAP
- UMAP
- NetworkX

---

## End-to-End Pipeline

```text
Raw CSV Data
    ↓
Data Cleaning & Missing Value Handling
    ↓
EDA & Statistical Analysis
    ↓
PCA + UMAP Embeddings
    ↓
Risk Clustering & SHAP Analysis
    ↓
Feature Engineering
    ↓
LightGBM + XGBoost + CatBoost
    ↓
5-Fold Stratified Cross Validation
    ↓
Optuna Weighted Ensemble
    ↓
Final Predictions