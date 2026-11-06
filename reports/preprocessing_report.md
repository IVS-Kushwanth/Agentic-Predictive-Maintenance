# Preprocessing & Pipeline Architecture Report: Phase 1

**Framework:** Agentic Predictive Maintenance  
**Document:** Preprocessing Pipeline, Splitting Protocols & Model-Ready Artifacts  
**Scope:** NASA C-MAPSS (FD001–FD004) & UCI AI4I 2020  

---

## 1. Splitting Methodology & Leakage Prevention

### 1.1 NASA C-MAPSS Engine-Level Split Protocol
Time-series degradation datasets possess strong temporal autocorrelation. Random row-wise train/test splitting distributes consecutive flight cycles and overlapping sliding windows from the same engine across both training and validation sets, resulting in severe data leakage and artificially inflated performance metrics.

**Adopted Protocol:**
1. Splitting is executed strictly at the **Engine ID level (`unit_id`)**.
2. **Train / Validation Ratio:** 80% Training Engines, 20% Validation Engines.
3. **Reproducibility:** Seed fixed to `42` (`numpy.random.RandomState(42)`).
4. **Official Test Set:** Maintained 100% isolated and untouched for final out-of-sample evaluation.

#### Engine Group Allocation (FD001 Benchmark):
- **Training Fleet (80 Engines):** Units 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 65, 66, 67, 68, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 81, 82, 83, 84, 85, 86.
- **Validation Fleet (20 Engines):** Units 5, 20, 39, 59, 69, 80, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100.
- **Set Intersection:** $\text{Train} \cap \text{Val} = \emptyset$ (Zero engine overlap).

### 1.2 UCI AI4I 2020 Stratified Tabular Split
- **Methodology:** Stratified Shuffle Split (80:20 ratio, `random_state=42`).
- **Stratification Target:** `Machine failure` (0 vs. 1).
- **Result:**
  - Training Set: 8,000 samples, 271 failures (3.39% failure rate).
  - Test Set: 2,000 samples, 68 failures (3.40% failure rate).
  - Exact preservation of minority class prevalence across splits.

---

## 2. Normalization and Scaling Pipelines

### 2.1 NASA Scaling Architecture
- **Scaler:** `MinMaxScaler(feature_range=(-1, 1))`
- **Fitting Scope:** Fitted **strictly on the 80 training engines**. Validation and test engines are transformed using the pre-fitted training parameters $(\min_{\text{train}}, \max_{\text{train}})$.
- **Feature Order Preservation:** Input column ordering ($D=153$ features for FD001) is strictly maintained across train, val, and test matrices.
- **Persistence:** Serialized to `models/preprocessing/nasa_scaler_FD001.joblib`.

### 2.2 UCI AI4I Preprocessing Pipeline
- **Architecture:** `sklearn.compose.ColumnTransformer`
  - **Numerical Features ($D_{\text{num}}=12$):** Standardized using `StandardScaler` (zero mean, unit variance).
  - **Categorical Feature (`Type`):** One-hot encoded using `OneHotEncoder(drop='first', sparse_output=False)` producing indicators `Type_L` and `Type_M` with `Type_H` as reference baseline.
  - **Excluded Columns:** `UDI`, `Product ID`, `Machine failure`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF`.
- **Fitting Scope:** Fitted **strictly on `X_train`**.
- **Persistence:** Serialized to `models/preprocessing/ai4i_preprocessor.joblib`.

---

## 3. Preprocessed Data Artifacts Inventory

### 3.1 NASA Processed Artifacts (`data/processed/nasa/FD001/`)

| File Name | Format | Dimensions | Purpose |
|:---|:---:|:---:|:---|
| `train_tabular.csv` | CSV | $(16,340 \times 159)$ | Scaled tabular training records with RUL and early warning targets. |
| `val_tabular.csv` | CSV | $(4,291 \times 159)$ | Scaled tabular validation records. |
| `test_tabular.csv` | CSV | $(13,096 \times 159)$ | Scaled tabular test records with reconstructed RUL. |
| `sequences_w30_train.npz` | NPZ | $X: (14020, 30, 153)$ | 30-cycle sliding window sequence training set. |
| `sequences_w30_val.npz` | NPZ | $X: (3711, 30, 153)$ | 30-cycle sliding window sequence validation set. |
| `sequences_w30_test.npz` | NPZ | $X: (10196, 30, 153)$ | 30-cycle sliding window sequence test set. |
| `sequences_w50_train.npz` | NPZ | $X: (12420, 50, 153)$ | 50-cycle sliding window sequence training set. |
| `sequences_w50_val.npz` | NPZ | $X: (3311, 50, 153)$ | 50-cycle sliding window sequence validation set. |
| `sequences_w50_test.npz` | NPZ | $X: (8262, 50, 153)$ | 50-cycle sliding window sequence test set. |
| `sequences_w60_train.npz` | NPZ | $X: (11620, 60, 153)$ | 60-cycle sliding window sequence training set. |
| `sequences_w60_val.npz` | NPZ | $X: (3111, 60, 153)$ | 60-cycle sliding window sequence validation set. |
| `sequences_w60_test.npz` | NPZ | $X: (7363, 60, 153)$ | 60-cycle sliding window sequence test set. |
| `metadata.json` | JSON | N/A | Engine split lists, active feature names, dropped sensors. |

### 3.2 AI4I Processed Artifacts (`data/processed/ai4i/`)

| File Name | Format | Dimensions | Purpose |
|:---|:---:|:---:|:---|
| `ai4i_cleaned.csv` | CSV | $(10,000 \times 14)$ | Validated raw data. |
| `ai4i_engineered.csv` | CSV | $(10,000 \times 21)$ | Full dataset with 7 physical engineered features. |
| `X_train.csv` | CSV | $(8,000 \times 14)$ | Standardized model input features (train split). |
| `X_test.csv` | CSV | $(2,000 \times 14)$ | Standardized model input features (test split). |
| `y_train.csv` | CSV | $(8,000 \times 1)$ | Binary ground-truth target (`Machine failure`). |
| `y_test.csv` | CSV | $(2,000 \times 1)$ | Binary ground-truth target (`Machine failure`). |
| `metadata.json` | JSON | N/A | Feature schema, column categorizations, split statistics. |

---

## 4. Explainable AI (XAI) Readiness

To support Phase 2 model interpretability (SHAP, Permutation Importance, Integrated Gradients, Attention Weight attribution):
1. **Feature Metadata Persistence:** All feature names and transformation indices are saved both in `.npz` headers and JSON metadata files.
2. **Physical Semantic Integrity:** Raw physical units are preserved prior to scaling; inverse transform functions can reconstruct human-interpretable values (e.g., Temperature in Kelvin or Pressure in psia) for decision support explanations.
3. **Temporal Step Indexing:** Sequence metadata tracks each window's exact `engine_id` and `end_cycle`, allowing temporal explanations to pinpoint the exact flight cycle at which degradation began.
