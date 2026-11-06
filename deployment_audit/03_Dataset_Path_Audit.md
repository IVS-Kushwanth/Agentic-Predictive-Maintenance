# 03. Dataset Paths & Data Portability Audit
**Project:** Agentic Predictive Maintenance Framework  
**Document ID:** AUDIT-03-DATASETS  
**Auditor:** Senior Cloud Infrastructure & Data Engineer  
**Audit Date:** November 2026  
**Status:** AUDITED — HIGH RISK IDENTIFIED (DATASET SIZE & GITHUB LIMITS)  

---

## 1. Dataset Path Portability Audit

Every dataset-loading function across `app/frontend/components/data_loader.py` and `src/` was scanned for hardcoded local file paths (e.g., `C:\Users\HP\...`):

### Audit Result: PASS
- **Zero hardcoded Windows absolute paths** exist in the application code.
- All modules utilize relative project root anchors:
  ```python
  BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
  data_path = os.path.join(BASE_DIR, "data", "processed", ...)
  ```
- This confirms that dataset loading executes correctly regardless of the host operating system, username, or directory installation path.

---

## 2. Dataset Size & Cloud Storage Footprint Audit

A complete disk usage audit was executed across the `data/` directory:

| Directory Partition | Disk Space Occupied | Primary Contents | Cloud Impact |
| :--- | :---: | :--- | :--- |
| **`data/raw/`** | 185.4 MB | NASA text files & raw AI4I CSV | Non-essential for runtime inference |
| **`data/processed/nasa/`** | 1,642.1 MB | Rolling tabular CSVs & sequence NPZ tensors | **CRITICAL: Several files exceed 100 MB** |
| **`data/processed/ai4i/`** | 1.8 MB | Cleaned tabular test CSVs | Lightweight & cloud-compatible |
| **`data/maintenance_history.db`** | 0.04 MB | SQLite audit database | Lightweight |
| **Total `data/` Directory** | **1,838.98 MB (1.84 GB)** | Complete benchmark training archives | **BLOCKING FOR GITHUB & STREAMLIT CLOUD** |

### 2.1 Critical GitHub 100 MB File Limit Violation
GitHub strictly blocks pushes containing any individual file larger than **100 MB**:

| File Path | Actual File Size | GitHub Status |
| :--- | :---: | :---: |
| `data/processed/nasa/FD001/sequences_w60_train.npz` | **314.28 MB** | **REJECTED (Hard Block)** |
| `data/processed/nasa/FD001/sequences_w60_test.npz` | **198.23 MB** | **REJECTED (Hard Block)** |
| `data/processed/nasa/FD004/train_tabular.csv` | **192.91 MB** | **REJECTED (Hard Block)** |
| `data/processed/nasa/FD002/train_tabular.csv` | **170.18 MB** | **REJECTED (Hard Block)** |
| `data/processed/nasa/FD004/test_tabular.csv` | **163.78 MB** | **REJECTED (Hard Block)** |
| `data/processed/nasa/FD002/test_tabular.csv` | **134.67 MB** | **REJECTED (Hard Block)** |

If a developer executes `git add .` and `git push`, GitHub will abort the entire transaction.

---

## 3. Runtime Dependency Analysis (What Data Does the Web App Actually Need?)

An architectural code audit was conducted on `app/frontend/components/data_loader.py` to identify which specific data files are strictly required for runtime dashboard operation:

```python
# Exact files loaded by Streamlit for sample telemetry:
1. data/processed/nasa/FD001/sequences_w30_val.npz (2.57 MB)
2. data/processed/nasa/FD001/val_tabular.csv        (12.14 MB)
3. data/processed/ai4i/X_test.csv                  (0.47 MB)
4. data/processed/ai4i/y_test.csv                  (0.01 MB)
5. data/processed/ai4i/metadata.json               (0.00 MB)
```

### Key Discovery:
The running application requires **ONLY 15.19 MB of validation data** to deliver 100% of its interactive capabilities! The remaining **1.82 GB** consists of training-phase sequence archives and raw logs that are completely untouched during live inference.

---

## 4. Remediation Strategy
1. Exclude all heavy training files (`train_tabular.csv`, `*train.npz`, `*test.npz`) in `.gitignore`.
2. Keep only the lightweight validation runtime files (`val_tabular.csv`, `sequences_w30_val.npz`, `X_test.csv`, `y_test.csv`, `metadata.json`) tracked in version control.
3. Total deployable repository size drops from **2.1 GB to under 200 MB**, fully resolving GitHub and cloud container size constraints.
