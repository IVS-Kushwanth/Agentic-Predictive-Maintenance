# End-User Installation & Setup Guide
## Agentic Predictive Maintenance Framework

This guide provides step-by-step instructions to install, configure, verify, and run the complete Agentic Predictive Maintenance system on a clean host machine.

---

## 1. System Prerequisites

Before beginning installation, ensure your host environment meets these requirements:

### Hardware Requirements
- **CPU**: Quad-Core 64-bit x86/x64 processor (Intel Core i5/i7 or AMD Ryzen 5/7).
- **RAM**: Minimum 8 GB DDR4 (16 GB recommended for deep sequence batching).
- **Disk Storage**: At least 5 GB free SSD space for datasets, serialized models, and logs.
- **Operating System**: Windows 10/11 (64-bit), Ubuntu Linux 20.04/22.04 LTS, or macOS 12+.

### Software Requirements
- **Python Runtime**: Python 3.10, 3.11, or 3.12 (Validated on **Python 3.12.3** 64-bit).
- **Git**: Version 2.30 or higher for repository cloning.
- **Package Manager**: Python Pip version 24.0 or higher.
- **Internet Connection**: Required for initial dependency downloads and dataset fetching.

---

## 2. Step-by-Step Installation Procedure

### Step 2.1: Clone the Repository
Open a terminal (PowerShell on Windows, Bash on Linux/macOS) and clone the project:
```bash
git clone https://github.com/organization/agentic-predictive-maintenance.git
cd agentic-predictive-maintenance
```

### Step 2.2: Create and Activate Virtual Environment
Always install dependencies in an isolated virtual environment to prevent package version conflicts:

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell displays an Execution Policy restriction, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and re-run activate).*

**On Linux / macOS (Bash):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2.3: Upgrade Package Installer
Ensure `pip`, `setuptools`, and `wheel` are up to date:
```bash
python -m pip install --upgrade pip setuptools wheel
```

### Step 2.4: Install Core Dependencies
Install all required libraries specified in `requirements.txt`:
```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing or you require explicit package pinning, install the verified versions directly:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn xgboost shap langgraph langchain-core fastapi uvicorn streamlit plotly pandas numpy python-multipart httpx python-docx
```

---

## 3. Dataset Setup & Automated Ingestion

The framework includes automated ingestion scripts that retrieve, unpack, and validate both the NASA C-MAPSS and UCI AI4I 2020 benchmark datasets.

Execute the download and validation pipeline:
```bash
# 1. Download raw data archives
python src/data_download.py

# 2. Audit schemas, nulls, and duplicate values
python src/data_validation.py

# 3. Generate feature engineered tabular matrices and sliding sequence tensors
python src/feature_engineering.py
python src/sequence_generation.py
```

### Expected Directory Layout After Data Setup:
```
data/
|-- raw/
|   |-- CMAPSS/
|   |   |-- train_FD001.txt ... train_FD004.txt
|   |   |-- test_FD001.txt ... test_FD004.txt
|   |   |-- RUL_FD001.txt ... RUL_FD004.txt
|   `-- ai4i/
|       `-- ai4i2020.csv
`-- processed/
    |-- train_tabular.csv
    |-- val_tabular.csv
    |-- sequences_w30_train.npz
    |-- sequences_w30_val.npz
    |-- sequences_w60_train.npz
    `-- sequences_w60_val.npz
```

---

## 4. Model Artifact Setup & Verification

Pretrained baseline models, PyTorch checkpoints, and anomaly detectors are serialized in the `models/` directory:
- `models/baseline/`: Contains 32 Scikit-learn and XGBoost `.joblib` model pipelines.
- `models/deep_learning/`: Contains 18 PyTorch `.pt` model weights (LSTM, Bi-LSTM, GRU).
- `models/anomaly/`: Contains LSTM Autoencoder checkpoints and Isolation Forest binaries.
- `models/scalers/`: Contains fitted `MinMaxScaler` and `StandardScaler` objects.

Verify all model artifacts using the automated verification harness:
```bash
python src/final_validation_check.py
```
*Expected Output:* `[INFO] All datasets, sequences, scalers, and model checkpoints VERIFIED.`

---

## 5. Database Initialization

The human-in-the-loop review system operates on a zero-configuration SQLite database (`data/maintenance_history.db`).
Initialize and verify the database schema:
```bash
python -c "from src.recommendations.recommendation_engine import RecommendationEngine; engine = RecommendationEngine(); print('Database initialized at:', engine.db_path)"
```

---

## 6. Installation Verification Smoke Test

Run the automated regression test suite to confirm complete system integrity:
```bash
python tests/test_phase8_backend.py
python tests/test_phase9_e2e_resilience.py
```
*Verification Assertion:* All 21 tests must return `OK` / `[PASS]` status.
