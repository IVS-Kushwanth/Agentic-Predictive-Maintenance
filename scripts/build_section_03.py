"""
Builder script for Section 03: User Instruction Manual
- Installation_Guide.md
- Configuration_Guide.md
- Application_User_Manual.md
- Troubleshooting_Guide.md
"""

import os

DEST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "documentation", "03_User_Instruction_Manual")
os.makedirs(DEST_DIR, exist_ok=True)

INSTALL_GUIDE_CONTENT = r"""# End-User Installation & Setup Guide
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
"""

CONFIG_GUIDE_CONTENT = r"""# System Configuration Guide
## Agentic Predictive Maintenance Framework

This guide details all configuration options, environment variables, network port bindings, model paths, and operational thresholds across the framework.

---

## 1. Environment Variables

The application reads configuration parameters from operating system environment variables or an optional `.env` file located in the project root directory.

| Variable Name | Default Value | Permitted Values | Architectural Purpose |
| :--- | :--- | :--- | :--- |
| `FASTAPI_HOST` | `127.0.0.1` | Valid IPv4 / `0.0.0.0` | Host IP interface for FastAPI REST server. |
| `FASTAPI_PORT` | `8000` | Port `1024 - 65535` | Listening port for FastAPI microservice. |
| `STREAMLIT_PORT` | `8501` | Port `1024 - 65535` | Listening port for Streamlit web dashboard. |
| `DATABASE_URL` | `sqlite:///data/maintenance_history.db` | SQLite URI | Path to SQLite audit and review database. |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | System logging verbosity across all modules. |
| `DEVICE` | `cpu` | `cpu`, `cuda` | PyTorch inference hardware device. |
| `MAX_RUL_CLIP` | `125` | Positive Integer | Piecewise linear RUL target threshold (cycles). |
| `AI4I_THRESHOLD` | `0.485` | Float $[0.0, 1.0]$ | Calibrated decision threshold for machine failure. |
| `ANOMALY_THRESHOLD` | `0.0370` | Positive Float | Calibrated nominal reconstruction error threshold. |
| `CORS_ORIGINS` | `*` | Comma-separated URLs | Permitted Cross-Origin Resource Sharing origins. |

---

## 2. Configuration Profiles

### 2.1 Development / Local Workstation Profile
Create a `.env` file in the project root:
```ini
FASTAPI_HOST=127.0.0.1
FASTAPI_PORT=8000
STREAMLIT_PORT=8501
LOG_LEVEL=DEBUG
DEVICE=cpu
DATABASE_URL=sqlite:///data/maintenance_history.db
```

### 2.2 Production Industrial Node Profile
For deployment on a dedicated plant reliability server:
```ini
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000
STREAMLIT_PORT=8501
LOG_LEVEL=INFO
DEVICE=cuda
DATABASE_URL=sqlite:///data/maintenance_history.db
CORS_ORIGINS=http://192.168.1.50:8501,http://mes.plant.internal
```

---

## 3. Model & Artifact Paths Configuration

All default file paths are defined in `src/models/registry.py` and can be overridden programmatically:

```python
MODEL_PATHS = {
    "nasa_rul_xgboost": "models/baseline/nasa_rul_xgboost_regressor.joblib",
    "nasa_rul_bilstm_w60": "models/deep_learning/bilstm_rul_w60.pt",
    "nasa_rul_lstm_w30": "models/deep_learning/lstm_rul_w30.pt",
    "ai4i_failure_gbdt": "models/baseline/ai4i_gradient_boosting_classifier.joblib",
    "ai4i_failure_xgboost": "models/baseline/ai4i_xgboost_classifier.joblib",
    "nasa_lstm_autoencoder": "models/anomaly/nasa_lstm_autoencoder.pt",
    "ai4i_isolation_forest": "models/anomaly/ai4i_isolation_forest.joblib",
    "nasa_minmax_scaler": "models/scalers/nasa_minmax_scaler.joblib",
    "ai4i_standard_scaler": "models/scalers/ai4i_standard_scaler.joblib"
}
```

---

## 4. Priority Classification & Threshold Rules

Priority classification thresholds can be adjusted in `src/recommendations/priority_engine.py`:

```python
# Industrial Decision Thresholds (ISO 13381-1 Aligned)
PRIORITY_THRESHOLDS = {
    "RUL_CRITICAL": 20.0,      # Cycles remaining for emergency stand-down
    "RUL_HIGH": 50.0,          # Cycles remaining for scheduled 48-hour service
    "RUL_MEDIUM": 80.0,        # Cycles remaining for next-shift inspection
    "FAILURE_PROB_CRITICAL": 0.75, # Machine failure probability triggering shutdown
    "FAILURE_PROB_HIGH": 0.40,     # Machine failure probability triggering inspection
    "FAILURE_PROB_MEDIUM": 0.20,   # Machine failure probability triggering monitoring
    "ANOMALY_SEVERITY_TIERS": {
        "NOMINAL": 0.00,
        "LOW": 0.50,
        "MEDIUM": 0.65,
        "HIGH": 0.80
    }
}
```
"""

USER_MANUAL_CONTENT = r"""# End-User Application Operation Manual
## Agentic Predictive Maintenance Web Control Center

This manual explains how plant engineers, reliability technicians, and maintenance supervisors operate the web application without requiring access to the underlying Python source code.

---

## 1. Starting the Application

The entire platform (FastAPI REST backend and Streamlit web dashboard) is launched with a single unified command:

```powershell
python run_dashboard.py
```

### Console Startup Output:
```
======================================================================
  AGENTIC PREDICTIVE MAINTENANCE PLATFORM
======================================================================
  [INFO] Starting FastAPI Backend on http://127.0.0.1:8000
  [INFO] FastAPI documentation available at http://127.0.0.1:8000/docs
  [INFO] Starting Streamlit Dashboard on http://127.0.0.1:8501
  [INFO] Press CTRL+C to terminate both servers cleanly.
======================================================================
```

Once launched, open your web browser (Google Chrome, Microsoft Edge, Mozilla Firefox) and navigate to:
**`http://127.0.0.1:8501`**

---

## 2. Navigating the Web Dashboard

The interface is organized into 7 specialized operational pages accessible via the left sidebar navigation menu:

```
[Sidebar Navigation]
|-- 1. Fleet Monitoring Overview
|-- 2. Turbofan RUL Estimator (NASA C-MAPSS)
|-- 3. CNC Machine Failure Predictor (UCI AI4I)
|-- 4. Multivariate Anomaly Detection
|-- 5. Explainable AI Diagnostics (SHAP)
|-- 6. Multi-Agent Orchestrator Studio
`-- 7. Maintenance Work Orders & History
```

---

## 3. Operational Walkthrough by Page

### Page 1: Fleet Monitoring Overview
- **Purpose**: High-level operational health dashboard displaying plant-wide asset status.
- **Key Metrics Displayed**:
  - Total Assets Monitored (100 Turbofans, 2,000 CNC shifts).
  - Active High-Risk Warnings.
  - Critical Maintenance Interventions Pending Human Sign-Off.
- **Recent Decision Ledger**: Displays the last 10 recommendations generated across the plant with color-coded priority badges (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`).

### Page 2: Turbofan RUL Estimator (NASA C-MAPSS)
- **Step 1 (Select Asset)**: In the sidebar, select **Dataset: NASA C-MAPSS** and choose an **Engine ID** (e.g., `Engine #24`).
- **Step 2 (Select Operational Cycle)**: Use the cycle slider to select the observation point (e.g., Cycle `130`).
- **Step 3 (Inspect Degradation Curve)**: View the interactive Plotly multi-sensor degradation trajectory. Sensor lines represent moving averages of HPC outlet temperature (`sensor_11`), LPT exhaust temperature (`sensor_4`), and core speed.
- **Step 4 (View Predictions)**:
  - **Predicted RUL**: Displayed on a high-contrast circular gauge dial (e.g., `18.2 Cycles Remaining`).
  - **Early Warning Indicators**: Multi-horizon warning flags indicating whether failure is imminent within 10, 20, or 30 flight cycles.

### Page 3: CNC Machine Failure Predictor (UCI AI4I 2020)
- **Step 1 (Select Asset)**: In the sidebar, select **Dataset: UCI AI4I 2020**.
- **Step 2 (Configure Operating Conditions)**: Adjust machine parameter sliders:
  - Air Temperature [K] (e.g., `300.5 K`)
  - Process Temperature [K] (e.g., `311.2 K`)
  - Rotational Speed [RPM] (e.g., `1380 RPM`)
  - Torque [Nm] (e.g., `58.4 Nm`)
  - Tool Wear [min] (e.g., `215 min`)
- **Step 3 (Execute Failure Risk Assessment)**: Click **"Run Machine Failure Prediction"**.
- **Step 4 (Interpret Risk)**:
  - **Failure Probability Dial**: Displays calibrated risk percentage (e.g., `84.2% Risk — BREAKDOWN WARNING`).
  - **Failure Mode Breakdown**: Displays likelihood of specific mechanical causes (Tool Wear Failure, Heat Dissipation Failure, Overstrain Failure).

### Page 4: Multivariate Anomaly Detection
- **Purpose**: Continuous unsupervised sensor health evaluation without ground-truth labels.
- **Features**:
  - **Reconstruction Error Chart**: Displays sequence reconstruction error plotted against nominal threshold $\tau = 0.0370$.
  - **Severity Tier Badge**: Categorizes asset state into `NOMINAL`, `LOW ANOMALY`, `MEDIUM ANOMALY`, or `HIGH ANOMALY`.
  - **Sensor Anomaly Radar**: Multi-axis radar chart showing which specific sensors deviate most severely from healthy baselines.

### Page 5: Explainable AI Diagnostics (SHAP)
- **Purpose**: Game-theoretic causal attribution explaining why an asset received an alert.
- **Interactive Waterfall Chart**:
  - Shows base expected value $E[f(x)]$ (e.g., average RUL = 108 cycles).
  - Red bars represent sensors that accelerate wear (e.g., `sensor_11 +24.2 cycles reduction`).
  - Blue bars represent sensors operating within safe parameters.
- **Feature Ranking Table**: Lists top 5 physical sensor drivers with their exact percentage attributions.

### Page 6: Multi-Agent Orchestrator Studio
- **Purpose**: Step-by-step transparency into the LangGraph multi-agent decision process.
- **Execution**: Click **"Run Multi-Agent Diagnostic Analysis"**.
- **Agent Workflow Steps Visualized**:
  1. `SupervisorAgent`: Ingests telemetry and selects routing pipeline.
  2. `PredictionAgent`: Computes RUL / failure probability.
  3. `AnomalyAgent`: Computes reconstruction error and anomaly tier.
  4. `ExplanationAgent`: Runs TreeSHAP and extracts top sensor attributions.
  5. `MaintenanceAgent`: Synthesizes evidence into an ISO 13381-1 action plan.
  6. `ResultValidation`: Asserts schema completeness and persists to SQLite.

### Page 7: Maintenance Work Orders & History
- **Reviewing Work Orders**: Maintenance engineers review all generated recommendations in a table format.
- **Human-in-the-Loop Actions**:
  - **Approve**: Confirms work order and dispatches work order to plant crew.
  - **Reject**: Cancels work order with mandatory text justification.
  - **Request Further Analysis**: Schedules non-destructive vibration or oil analysis.
- **Exporting Reports**: Click **"Download Maintenance Audit CSV"** to export records for plant ERP / compliance audits.

---

## 4. Application Shutdown Procedure
To cleanly stop both the FastAPI backend and Streamlit dashboard:
1. In the terminal running `run_dashboard.py`, press **`CTRL + C`**.
2. Wait 2 seconds for both child processes to terminate and release ports 8000 and 8501.
"""

TROUBLESHOOTING_GUIDE_CONTENT = r"""# System Troubleshooting & Diagnostic Guide
## Agentic Predictive Maintenance Framework

This guide provides troubleshooting solutions for common startup errors, port conflicts, dependency mismatches, database locks, and runtime anomalies.

---

## 1. Quick Diagnostic Checklist

If the system fails to start or behaves unexpectedly, execute these diagnostic checks in sequence:

| Diagnostic Step | Command / Inspection | Expected Result |
| :--- | :--- | :--- |
| **Python Version** | `python --version` | Python `3.10.x`, `3.11.x`, or `3.12.x` (64-bit). |
| **Virtual Environment** | `Get-Command python` (Windows) | Points to `.venv\Scripts\python.exe`. |
| **Model Artifacts** | `Test-Path models\baseline\*.joblib` | Returns `True` (32 models present). |
| **Database File** | `Test-Path data\maintenance_history.db` | Returns `True`. |
| **Port Availability** | `netstat -ano \| findstr :8000` | Port 8000 must not be occupied by another process. |
| **Port Availability** | `netstat -ano \| findstr :8501` | Port 8501 must not be occupied by another process. |

---

## 2. Common Errors and Solutions

### Issue 2.1: Port Conflict (Error: [Errno 10048] address already in use)
- **Symptoms**: FastAPI or Streamlit terminates immediately with `Address already in use` on port 8000 or 8501.
- **Cause**: A previous server instance was terminated abnormally or another application occupies the port.
- **Solution (Windows PowerShell)**:
  ```powershell
  # Find PID occupying port 8000
  $conn = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
  if ($conn) { Stop-Process -Id $conn.OwningProcess -Force; Write-Host "Killed PID $($conn.OwningProcess)" }

  # Find PID occupying port 8501
  $conn = Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue
  if ($conn) { Stop-Process -Id $conn.OwningProcess -Force; Write-Host "Killed PID $($conn.OwningProcess)" }
  ```
- **Solution (Linux / macOS)**:
  ```bash
  kill -9 $(lsof -t -i:8000) 2>/dev/null
  kill -9 $(lsof -t -i:8501) 2>/dev/null
  ```

---

### Issue 2.2: SQLite Database Locked (Error: sqlite3.OperationalError: database is locked)
- **Symptoms**: Review updates or recommendations fail with `database is locked`.
- **Cause**: Multiple concurrent threads or processes attempted write access without closing transactions.
- **Solution**:
  1. Terminate running server processes (`CTRL+C`).
  2. The recommendation engine implements an automatic 10-second connection timeout:
     ```python
     sqlite3.connect("data/maintenance_history.db", timeout=10.0)
     ```
  3. If corruption persists, backup and vacuum the database:
     ```powershell
     python -c "import sqlite3; conn = sqlite3.connect('data/maintenance_history.db'); conn.execute('VACUUM'); conn.close(); print('DB Vacuumed')"
     ```

---

### Issue 2.3: PyTorch Model Load Error (Error: RuntimeError: Error(s) in loading state_dict)
- **Symptoms**: Prediction agent throws mismatch in layer tensor shapes when loading `.pt` checkpoints.
- **Cause**: Attempting to load an $L=60$ checkpoint into an $L=30$ model instance.
- **Solution**: Ensure the model factory matches the checkpoint window length:
  ```python
  # For L=30 checkpoint:
  model = LSTMRegressor(input_dim=14, hidden_dim=64, num_layers=2)
  model.load_state_dict(torch.load("models/deep_learning/lstm_rul_w30.pt", map_location="cpu"))

  # For L=60 checkpoint:
  model = BiLSTMRegressor(input_dim=14, hidden_dim=64, num_layers=2)
  model.load_state_dict(torch.load("models/deep_learning/bilstm_rul_w60.pt", map_location="cpu"))
  ```

---

### Issue 2.4: Out-of-Range Sensor Values / NaN Inputs (HTTP 422 Unprocessable Content)
- **Symptoms**: API returns HTTP 422 or UI displays red validation alert.
- **Cause**: Negative temperatures (in Kelvin) or non-numeric sensor readings submitted.
- **Solution**: The Pydantic request schema enforces physical bounds (`Air Temp > 250 K`, `Torque > 0 Nm`). Validate telemetry ranges before sending requests.

---

### Issue 2.5: Missing Package Dependencies (ModuleNotFoundError)
- **Symptoms**: `ModuleNotFoundError: No module named 'fastapi'` or similar.
- **Cause**: Dependencies installed outside the active virtual environment.
- **Solution**: Verify active virtual environment path and re-run installation:
  ```powershell
  .\.venv\Scripts\python.exe -m pip install -r requirements.txt
  ```
"""

def generate_section_03():
    files = {
        "Installation_Guide.md": INSTALL_GUIDE_CONTENT,
        "Configuration_Guide.md": CONFIG_GUIDE_CONTENT,
        "Application_User_Manual.md": USER_MANUAL_CONTENT,
        "Troubleshooting_Guide.md": TROUBLESHOOTING_GUIDE_CONTENT,
    }
    for fname, content in files.items():
        fpath = os.path.join(DEST_DIR, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content.strip() + "\n")
        print(f"Generated: {fpath} ({len(content)} bytes)")

if __name__ == "__main__":
    generate_section_03()
