# 04. Trained Model Artifacts & Checkpoints Audit
**Project:** Agentic Predictive Maintenance Framework  
**Document ID:** AUDIT-04-MODEL-ARTIFACTS  
**Auditor:** Senior AI/ML Deployment Specialist  
**Audit Date:** November 2026  
**Status:** PASS — ALL ARTIFACTS VERIFIED & CLOUD-COMPATIBLE  

---

## 1. Model Artifact Inventory & Disk Footprint

The `models/` directory contains 121 serialized artifacts spanning baseline regressors, early-warning classifiers, PyTorch deep recurrent neural networks, anomaly detectors, and feature scalers:

| Subdirectory | Model Count | Total Size | Largest Artifact | Format |
| :--- | :---: | :---: | :--- | :---: |
| **`models/baseline/`** | 68 files | 152.4 MB | `nasa_rul_random_forest_regressor_FD004.joblib` (46.87 MB) | Scikit-Learn / XGBoost `.joblib` |
| **`models/deep_learning/`**| 18 files | 1.8 MB | `bilstm_rul_w60.pt` (0.12 MB) | PyTorch `state_dict` `.pt` |
| **`models/anomaly_detection/`**| 24 files | 14.8 MB | `lof_FD001.joblib` (5.03 MB) | Scikit-Learn `.joblib` & PyTorch `.pt` |
| **`models/scalers/`** | 11 files | 0.8 MB | `nasa_minmax_scaler.joblib` (0.15 MB) | Scikit-Learn `.joblib` |
| **Total `models/`** | **121 files** | **169.84 MB** | **All files under 50 MB** | — |

---

## 2. Artifact Verification Findings

### 2.1 File Size & Version Control Compliance: PASS
- **GitHub Compatibility**: Every single model file is well below GitHub's 100 MB hard reject limit. The largest model is 46.87 MB.
- **Git LFS Requirement**: Git Large File Storage (LFS) is *not strictly required* for the models, as all files are $< 50\text{ MB}$, though Git LFS is recommended if model tracking history grows.

### 2.2 Model Deserialization & Inference Test: PASS
All model families were subjected to automated deserialization and inference verification in `tests/test_phase9_e2e_resilience.py`:
- Traditional ML baselines (XGBoost, Random Forest, Gradient Boosting) loaded cleanly via `joblib.load()` and executed inference within 1.5 ms.
- PyTorch recurrent networks (`LSTMRegressor`, `BiLSTMRegressor`, `GRURegressor`) deserialized into their respective classes on CPU with `torch.load(..., map_location="cpu")` and executed forward passes in $< 65\text{ ms}$.
- Isolation Forest and LSTM Autoencoders loaded cleanly and produced continuous anomaly scores and severity classifications.

### 2.3 Preprocessing Scaler Alignment: PASS
- Feature scalers (`MinMaxScaler`, `StandardScaler`) are serialized alongside models in `models/scalers/`.
- Input feature dimensions and column ordering are strictly maintained:
  - NASA C-MAPSS: Exactly 153 features (14 active sensors + 136 rolling statistics + 3 settings).
  - UCI AI4I 2020: Exactly 14 engineered physical features.

---

## 3. Cloud Memory Footprint
- Peak memory consumption when loading all active production models into the FastAPI backend service (`ModelService`) is **under 350 MB RAM**.
- This enables smooth operation on entry-level cloud container tiers (512 MB to 1 GB RAM) without risk of Out-Of-Memory (OOM) termination.
