# 01. Complete Project Structure Audit
**Project:** Agentic Predictive Maintenance Framework  
**Document ID:** AUDIT-01-PROJECT-STRUCTURE  
**Auditor:** Senior DevOps & Software Architecture Specialist  
**Audit Date:** November 2026  
**Status:** AUDITED — ACTIONABLE ITEMS IDENTIFIED  

---

## 1. Directory Tree & Module Hierarchy Audit

The project directory was audited across all subdirectories to verify file presence, organizational modularity, and operational separation of concerns:

```text
agentic-predictive-maintenance/
|-- .vscode/                           # IDE configurations
|-- app/                               # Full-stack production web application
|   |-- backend/                       # FastAPI REST microservice
|   |   |-- routes/api_routes.py       # 10 REST endpoints
|   |   |-- schemas/api_schemas.py     # Pydantic v2 request/response schemas
|   |   |-- services/                  # Singleton model & orchestration services
|   |   `-- main.py                    # FastAPI ASGI application entrypoint
|   |-- database/database.py           # SQLite connection & session manager
|   `-- frontend/                      # Streamlit interactive web dashboard
|       |-- components/                # Modular UI widgets, loaders & Plotly charts
|       |-- pages/                     # 7 specialized operational view pages
|       `-- main.py                    # Streamlit application entrypoint
|-- config/                            # Global pipeline configuration
|-- data/                              # Benchmark datasets & runtime caches
|   |-- raw/ (CMAPSS, ai4i)            # Raw downloaded text/CSV archives
|   |-- processed/ (nasa, ai4i)        # Processed CSVs and sequence .npz tensors
|   `-- maintenance_history.db         # SQLite persistent recommendation database
|-- documentation/                     # Master documentation package (41 files)
|-- models/                            # 121 serialized model artifacts
|   |-- baseline/                      # Scikit-learn & XGBoost .joblib models
|   |-- deep_learning/                 # PyTorch .pt sequence checkpoints
|   |-- anomaly_detection/             # Autoencoders, Isolation Forests, OCSVM, LOF
|   `-- scalers/                       # Fitted MinMaxScaler & StandardScaler objects
|-- notebooks/                         # 14 exploratory Jupyter research notebooks
|-- presentation/                      # Academic presentation slide deck
|-- reports/                           # Phase-wise analytical reports & figures
|-- scripts/                           # Documentation and audit generator scripts
|-- src/                               # Foundational algorithmic & agent modules
|   |-- agents/                        # 5 specialized LangGraph agent classes
|   |-- graph/                         # Compiled StateGraph & routing predicates
|   |-- recommendations/               # ISO 13381-1 priority engine & rule catalog
|   |-- services/                      # Model & Explanation singleton services
|   |-- training/                      # ML/DL training execution pipelines
|   `-- [loaders & processors]         # Low-level ingestion & sequence builders
|-- tests/                             # Automated test suites (backend & resilience)
|-- README.md                          # Project documentation overview
|-- requirements.txt                   # Dependency specification file
`-- run_dashboard.py                   # Dual-process launcher script
```

---

## 2. Structural Verification Findings

### 2.1 Python Source Code Completeness: PASS
- **Frontend Pages**: All 7 operational pages exist in `app/frontend/pages/` (`page_overview.py`, `page_monitoring.py`, `page_predictive.py`, `page_explainability.py`, `page_agentic.py`, `page_recommendations.py`, `page_history.py`).
- **Backend Routes**: All 10 REST endpoints are implemented in `app/backend/routes/api_routes.py` with corresponding Pydantic schemas in `app/backend/schemas/api_schemas.py`.
- **LangGraph Agents**: All 5 specialized agent classes exist in `src/agents/` (`supervisor.py`, `prediction_agent.py`, `anomaly_agent.py`, `explanation_agent.py`, `maintenance_agent.py`).
- **Database Module**: Centralized SQLite connector and table manager exist in `src/database/` and are re-exported in `app/database/database.py`.

### 2.2 Broken & Circular Imports Audit: PASS
- An earlier circular import between `src.agents` and `src.graph` was resolved via a lazy worker initialization pattern (`get_supervisor()`, `get_predictor()`).
- All 27 Python source files across `app/backend/`, `app/frontend/`, and `src/` were compiled via `py_compile.compile(doraise=True)` with **zero syntax or import errors**.

### 2.3 Redundant Entrypoints & Architectural Duality: PARTIAL (NON-CRITICAL)
- **Finding**: Two distinct application layers exist:
  1. `app/backend/main.py` & `app/frontend/main.py`: The production full-stack implementation used by `run_dashboard.py`.
  2. `src/api/app.py` & `src/dashboard/app.py`: An earlier development prototype implementation.
- **Impact**: While both function, having duplicate entry points can confuse cloud deployment maintainers.
- **Recommendation**: Standardize deployment configuration exclusively around `app.backend.main:app` and `app/frontend/main.py`.

### 2.4 Temporary & Build Artifact Cleanliness: ACTION REQUIRED
- `__pycache__` directories and `.pyc` files are present across modules.
- `.venv/` is located directly in the workspace root.
- **Blocking Issue**: No `.gitignore` file exists in the repository! Committing to Git without `.gitignore` will push gigabytes of compiled bytecode and virtual environment packages.

---

## 3. Structural Action Items
1. Generate a robust `.gitignore` excluding `.venv/`, `__pycache__/`, `*.pyc`, `*.db`, and ephemeral logs.
2. Confirm `run_dashboard.py` and cloud startup scripts point exclusively to `app.backend.main:app` and `app/frontend/main.py`.
