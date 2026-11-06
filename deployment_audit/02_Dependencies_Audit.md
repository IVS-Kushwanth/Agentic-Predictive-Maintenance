# 02. Requirements.txt & Dependency Compatibility Audit
**Project:** Agentic Predictive Maintenance Framework  
**Document ID:** AUDIT-02-DEPENDENCIES  
**Auditor:** Senior DevOps & Python Software Architect  
**Audit Date:** November 2026  
**Status:** ACTION REQUIRED — CRITICAL OMISSIONS DETECTED  

---

## 1. Audit of Current `requirements.txt`

Inspection of the existing `requirements.txt` reveals:

```text
pandas>=2.2.0
numpy>=1.26.0
scipy>=1.12.0
scikit-learn>=1.4.0
matplotlib>=3.8.0
seaborn>=0.13.0
joblib>=1.3.0
pyyaml>=6.0.1
imbalanced-learn>=0.12.0
requests>=2.31.0
ipykernel>=6.29.0
xgboost>=3.0.0
torch>=2.2.0

shap>=0.45.0
captum>=0.7.0
langgraph>=0.2.0
langchain-core>=0.3.0
```

---

## 2. Dependency Discrepancies & Blocking Issues

### 2.1 Critical Missing Production Packages (BLOCKING)
The current `requirements.txt` was formulated during early Phase 1-3 model development. When the project expanded into a production web platform (Phases 6-8), several essential packages were installed into the local virtual environment but were **never added to `requirements.txt`**:

| Missing Package | Installed Local Version | Architectural Necessity | Consequence in Clean Cloud Container |
| :--- | :---: | :--- | :--- |
| **`fastapi`** | `0.162.0` | Core backend REST API framework | **CRASH:** `ModuleNotFoundError: No module named 'fastapi'` |
| **`uvicorn`** | `0.54.0` | Production ASGI web server | **CRASH:** Cannot start backend service |
| **`streamlit`** | `1.66.1` | Interactive web dashboard | **CRASH:** `ModuleNotFoundError: No module named 'streamlit'` |
| **`plotly`** | `7.1.0` | Core interactive charts | **CRASH:** Visualization components fail |
| **`pydantic`** | `2.14.3` | Request/response schema validation | **CRASH:** Backend routes fail to load |
| **`python-multipart`** | `0.0.33` | Form & file upload parsing | Backend file uploads fail |
| **`httpx`** | `0.28.1` | Async HTTP test client | Test suites fail |
| **`python-docx`** | `1.2.0` | Formal report generation | Report generator fails |

### 2.2 Version Constraint Conflicts (POTENTIAL BLOCKING)
- **`xgboost>=3.0.0`**: While XGBoost 3.4.1 is installed in this Python 3.12 environment, certain Linux cloud container base images (e.g., Python 3.10 / Debian Bullseye) only provide precompiled wheels up to XGBoost 2.1.x. Pinning `xgboost>=2.1.0` ensures cross-platform compatibility without breaking installed models.
- **`torch>=2.2.0`**: PyTorch CPU wheels installed from PyPI default to multi-gigabyte CUDA packages unless explicitly specified (`--index-url https://download.pytorch.org/whl/cpu`). On lightweight cloud tiers (e.g., Render, Railway, Streamlit Community Cloud with 512MB-1GB RAM limits), downloading full CUDA PyTorch causes disk exhaustion and out-of-memory (OOM) build aborts.

---

## 3. Recommended Production `requirements.txt`

To ensure deterministic, clean cloud deployment, `requirements.txt` must be updated with pinned, compatible versions:

```text
# Core Scientific & Data Processing
numpy>=1.26.0,<3.0.0
pandas>=2.2.0,<4.0.0
scipy>=1.12.0
scikit-learn>=1.4.0
imbalanced-learn>=0.12.0
joblib>=1.3.0
pyyaml>=6.0.1

# Machine Learning & Deep Learning (CPU Optimized)
torch>=2.2.0
xgboost>=2.1.0

# Anomaly Detection & Explainable AI
shap>=0.45.0
captum>=0.7.0

# Multi-Agent Orchestration
langgraph>=0.2.0
langchain-core>=0.3.0

# Backend Web Microservice
fastapi>=0.115.0
uvicorn>=0.30.0
pydantic>=2.8.0
python-multipart>=0.0.9
requests>=2.31.0
httpx>=0.27.0

# Frontend Web Dashboard & Charting
streamlit>=1.38.0
plotly>=5.22.0
matplotlib>=3.8.0
seaborn>=0.13.0

# Documentation & Utilities
python-docx>=1.1.0
```

---

## 4. Verdict
**Status:** `FAIL (Prior to update)` $\to$ **Remediated by generating production `requirements.txt`.**
