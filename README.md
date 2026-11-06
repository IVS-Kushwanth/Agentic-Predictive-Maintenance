# Agentic Predictive Maintenance Framework

**An Explainable Multivariate Time-Series Framework for Early Machine Failure Detection, Remaining Useful Life Estimation and Maintenance Decision Support**

[![Python 3.12](https://img.shields.io/badge/python-3.12.3-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-orange.svg)](https://github.com/langchain-ai/langgraph)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests: 40/40 Passing](https://img.shields.io/badge/Tests-40%2F40%20Pass%20(100%25)-brightgreen.svg)]()

---

## 1. Project Overview

The **Agentic Predictive Maintenance Framework** is an enterprise-grade industrial AI decision-support platform designed for high-consequence multivariate machinery. It combines:
1. **Multivariate Time-Series Modeling:** Continuous Remaining Useful Life (RUL) regression and multi-horizon early-warning binary classification on NASA C-MAPSS turbofan engines.
2. **Industrial Failure Diagnostics:** Machine failure classification and failure-mode attribution (Tool Wear, Overstrain, Heat Dissipation, Power Failure) on UCI AI4I 2020 CNC milling systems.
3. **Unsupervised Anomaly Detection:** Multivariate sensor reconstruction error and density scoring using Isolation Forest, LSTM Autoencoders, and Local Outlier Factor (LOF).
4. **Explainable AI (XAI):** Mathematically rigorous local and global sensor attributions via TreeSHAP and Captum DeepLift.
5. **Agentic Orchestration (LangGraph):** Autonomous multi-agent coordination (Supervisor, Prediction, Anomaly, Explanation, Maintenance, Safety Safeguard) with human-in-the-loop review.
6. **Full-Stack Application:** Decoupled FastAPI REST backend, SQLite audit persistence, and an interactive 7-page Streamlit web dashboard.

---

## 2. System Architecture

```mermaid
flowchart TD
    subgraph Data Layer
        NASA[NASA C-MAPSS FD001-FD004]
        AI4I[UCI AI4I 2020 CNC]
        ValData[(Runtime Datasets: 15.2 MB)]
    end

    subgraph Intelligence Core
        Models[(121 Saved Model Artifacts)]
        XGB[XGBoost & Random Forest Baselines]
        DL[PyTorch LSTM / GRU / CNN-LSTM]
        Anom[Isolation Forest & Autoencoders]
        XAI[TreeSHAP & Captum Engine]
    end

    subgraph Agentic Orchestration
        LG[LangGraph StateGraph]
        Supervisor[Supervisor Agent]
        Predictor[Prediction Agent]
        AnomalyAgent[Anomaly Agent]
        ExplainerAgent[Explanation Agent]
        Advisor[Recommendation Engine]
        Safeguard[Safety Safeguard Agent]
    end

    subgraph Full-Stack Interface
        FastAPI[FastAPI REST API :8000]
        Streamlit[Streamlit Dashboard :8501]
        DB[(SQLite Persistent Storage)]
    end

    Data Layer --> Intelligence Core
    Intelligence Core --> Agentic Orchestration
    Agentic Orchestration --> FastAPI
    FastAPI --> Streamlit
    FastAPI --> DB
```

---

## 3. Repository Structure

```
agentic-predictive-maintenance/
├── app/
│   ├── backend/                        # FastAPI REST API Microservice
│   │   ├── main.py                     # ASGI entry point, CORS, and health checks
│   │   ├── routes/                     # REST API routing endpoints (10 endpoints)
│   │   ├── schemas/                    # Pydantic request/response validation schemas
│   │   └── services/                   # Service layer & cached model singletons
│   └── frontend/                       # Streamlit 7-Page Interactive Web Dashboard
│       ├── main.py                     # Streamlit entry point
│       ├── components/                 # Reusable UI headers, sidebar, charts, tables
│       └── pages/                      # Overview, Monitoring, Prediction, Anomaly, XAI, Agentic, History
│
├── config/                             # Centralized YAML configurations
├── data/
│   ├── processed/                      # Preserved runtime validation datasets (15.19 MB)
│   └── maintenance_history.db          # SQLite persistent audit database
│
├── deployment_audit/                   # Complete pre-deployment audits & verification reports
├── documentation/                      # Master technical reports, manuals, and paper
├── models/                             # 121 trained model checkpoints (169.84 MB)
│   ├── baseline/                       # XGBoost, RF, GBDT regressors & classifiers
│   ├── deep_learning/                  # PyTorch LSTM, GRU, CNN-LSTM, Early-Warning, MLP
│   ├── anomaly_detection/              # Isolation Forest, Autoencoder, LOF
│   └── preprocessing/                  # Scalers and ColumnTransformers
│
├── presentation/                       # Conference & Viva slide deck
├── scripts/                            # Verification, test runner, profiling, and audit scripts
├── src/                                # Core domain packages
│   ├── anomaly_detection/              # Unsupervised scoring & threshold calibration
│   ├── database/                       # ACID SQLite connection, migrations, history DB
│   ├── deep_learning/                  # PyTorch neural network architectures & dataloaders
│   ├── explainability/                 # TreeSHAP & Captum integration
│   ├── graph/                          # LangGraph state machine & workflow graphs
│   └── recommendations/                # Priority engine & subsystem mapping rules
│
├── tests/                              # Automated test suites (40/40 tests passing)
├── Dockerfile                          # Multi-stage production container build
├── docker-compose.yml                  # Local container orchestration
├── render.yaml                         # Render Cloud Blueprint specification
└── requirements.txt                    # Pinned production dependencies
```

---

## 4. Quick Start & Local Execution

### 4.1 Prerequisites
- Python 3.12+ (Virtual environment recommended)
- Git

### 4.2 Installation
```bash
# Clone the repository
git clone <repository-url>
cd agentic-predictive-maintenance

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # On Linux/macOS
.\.venv\Scripts\Activate.ps1    # On Windows PowerShell

# Install dependencies (CPU PyTorch optimized)
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 4.3 Running the Application

#### Option A: Unified Launcher (Recommended)
Launches both the FastAPI backend (port 8000) and Streamlit dashboard (port 8501) simultaneously:
```bash
python run_dashboard.py
```
- **Streamlit Web UI:** Open [http://localhost:8501](http://localhost:8501) in your browser.
- **FastAPI Interactive Docs:** Open [http://localhost:8000/docs](http://localhost:8000/docs).

#### Option B: Standalone Services
```bash
# Run FastAPI Backend only
python run_dashboard.py --backend

# Run Streamlit Frontend only
python run_dashboard.py --frontend
```

---

## 5. API Reference Summary

| Method | Endpoint | Description |
| :---: | :--- | :--- |
| `GET` | `/health` | System health check and uptime status |
| `GET` | `/api/models` | Catalog of all 121 available trained model artifacts |
| `POST` | `/api/predict/rul` | Estimates turbofan Remaining Useful Life (RUL) in cycles |
| `POST` | `/api/predict/failure` | Estimates CNC machine breakdown risk and failure probability |
| `POST` | `/api/anomaly/detect` | Evaluates sensor anomalies using Isolation Forest / Autoencoder |
| `POST` | `/api/explain` | Generates TreeSHAP local feature attribution explanations |
| `POST` | `/api/agents/analyze` | Executes complete 6-node LangGraph multi-agent decision workflow |
| `POST` | `/api/recommendations` | Dispatches priority-ranked maintenance actions |
| `GET` | `/api/history` | Fetches historical maintenance recommendations |
| `POST` | `/api/recommendations/review` | Records engineering approval, rejection, or request for analysis |

---

## 6. Testing & Quality Assurance

Run the complete 40-test regression suite:
```bash
python scripts/run_all_tests.py
```
**Results:** `40/40 tests PASSED (100% pass rate)` across:
- `tests/test_phase8_backend.py` (10/10 PASS)
- `tests/test_phase9_e2e_resilience.py` (11/11 PASS)
- `src/test_phase6_orchestration.py` (9/9 PASS)
- `src/test_phase7_recommendations.py` (10/10 PASS)

---

## 7. Cloud Deployment Guide

### Target: Render (Backend) + Streamlit Community Cloud (Frontend)

1. **Backend on Render:**
   - Connect your GitHub repository to Render.
   - Use Blueprint deploy via [`render.yaml`](file:///C:/Users/HP/OneDrive/Desktop/ML%20Model/agentic-predictive-maintenance/render.yaml) or select **Starter Plan** ($7/mo, 1 GB RAM).
   - Attach a 1 GB persistent disk mounted at `/data`.
   - Set environment variable: `DATABASE_PATH=/data/maintenance.db`.

2. **Frontend on Streamlit Community Cloud:**
   - Connect your repository on [Streamlit Community Cloud](https://share.streamlit.io/).
   - Set main file path: `app/frontend/main.py`.
   - Under **App Settings -> Secrets**, set:
     ```toml
     BACKEND_API_URL = "https://<your-render-backend-url>.onrender.com"
     ```

For full step-by-step instructions, see [`deployment_audit/DEPLOYMENT_CHECKLIST.md`](file:///C:/Users/HP/OneDrive/Desktop/ML%20Model/agentic-predictive-maintenance/deployment_audit/DEPLOYMENT_CHECKLIST.md).

---

## 8. License

This project is licensed under the MIT License - see the LICENSE file for details.
