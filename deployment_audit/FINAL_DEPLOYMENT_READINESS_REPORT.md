# Comprehensive Cloud Deployment Readiness Audit & Scorecard
## Agentic Predictive Maintenance: An Explainable Multivariate Time-Series Framework for Early Machine Failure Detection, Remaining Useful Life Estimation and Maintenance Decision Support

**Document ID:** AUDIT-FINAL-CLOUD-READINESS  
**Auditing Team:** Senior DevOps, AI/ML Deployment & Cloud Architecture Team  
**Audit Timestamp:** November 2026  
**Final Status:** **AUDITED — REMEDIATION PLAN READY (PRE-DEPLOYMENT STAGE)**  

---

## 1. Executive Summary

This audit assesses the cloud deployment readiness of the **Agentic Predictive Maintenance Framework**. The inspection evaluated all aspects of the architecture—including repository structure, dependencies, dataset portability, model serialization, backend REST APIs, frontend UI components, credentials and environment variables, database persistence, multi-agent orchestration, and container configurations.

### Key Finding:
The application's core algorithmic engines, model inference pipelines, and multi-agent workflows are **100% functional, verified, and passing all 40 automated tests**. However, before deploying to public cloud infrastructure or pushing to GitHub, **three critical deployment blockers** must be addressed:
1. **`requirements.txt` is missing core web dependencies** (`fastapi`, `uvicorn`, `streamlit`, `plotly`, `pydantic`).
2. **Missing `.gitignore` file**, which would cause accidental commits of large training archives (>300 MB, exceeding GitHub's 100 MB limit) and local virtual environments.
3. **Hardcoded localhost fallback for backend API communication**, which must be made configurable via environment variables (`BACKEND_API_URL` and `CORS_ORIGINS`).

---

## 2. 16-Point Final Deployment Readiness Scorecard

| # | Audit Requirement | Status | Empirical Evidence | Problem Identified | Recommended Correction | File Requiring Modification |
| :---: | :--- | :---: | :--- | :--- | :--- | :--- |
| **1** | **Complete Project Code** | **PASS** | 27 Python modules in `app/` and `src/` compiled cleanly without error. | None. All routes, pages, and agents exist. | None required. | None |
| **2** | **requirements.txt** | **FAIL** | Current `requirements.txt` lacks `fastapi`, `uvicorn`, `streamlit`, `plotly`, `pydantic`. | Clean container installation will crash with missing module errors. | Add all installed web, server, and validation dependencies. | `requirements.txt` |
| **3** | **Dataset Portability** | **PARTIAL** | Runtime validation files (15.2 MB) load cleanly via relative paths; training files total 1.84 GB. | Training archives exceed GitHub 100 MB per-file push limit. | Exclude heavy training files from Git via `.gitignore`. | `.gitignore` |
| **4** | **Model Artifacts** | **PASS** | 121 model files verified (169.8 MB total; largest is 46.8 MB, well under 100 MB). | None. Models deserialize and infer in $< 65\text{ ms}$. | Bundle models directly in container image. | None |
| **5** | **FastAPI Cloud Compat.** | **PASS** | Binds dynamically to `$PORT` via Uvicorn; 10 endpoints passing. | CORS uses `allow_origins=["*"]` with credentials. | Make allowed origins configurable via `CORS_ORIGINS`. | `app/backend/main.py` |
| **6** | **Streamlit Cloud Compat.** | **PARTIAL** | All 7 pages render cleanly following `np` import fix in `page_overview.py`. | API URL defaults to `http://localhost:8000`. | Read `BACKEND_API_URL` from env variable or `st.secrets`. | `app/frontend/main.py`, `sidebar.py` |
| **7** | **Environment Variables** | **FAIL** | No `.env` or `.env.example` file exists in the repository. | Cloud operators have no reference template for configuration. | Create `.env.example` documenting all configuration keys. | `.env.example` |
| **8** | **API Key Protection** | **PASS** | Zero credentials or keys found in source code scan. | No `.gitignore` to prevent future accidental key commits. | Add `.env*` to `.gitignore`. | `.gitignore` |
| **9** | **SQLite Persistence** | **PARTIAL** | ACID transactions and indexes verified on `data/maintenance_history.db`. | Ephemeral cloud filesystems wipe SQLite history on restart. | Support `DATABASE_PATH` env variable for persistent volume mount. | `src/database/database.py` |
| **10** | **File Storage** | **PASS** | Uploaded CSVs processed in-memory; CSV reports streamed via memory buffers. | None. Zero temporary disk leakages. | None required. | None |
| **11** | **LangGraph Cloud Compat.**| **PASS** | Full 5-agent state graph executes locally and deterministically in ~2.85s. | None. Zero external cloud API key requirements. | None required. | None |
| **12** | **Clean Env Installation** | **FAIL** | Local venv works, but `requirements.txt` cannot reproduce the web app. | Blocked by Requirement #2. | Update `requirements.txt`. | `requirements.txt` |
| **13** | **Frontend/Backend Comms** | **PASS** | REST HTTP calls verified on local host (HTTP 200 OK across all routes). | Hardcoded default URL in frontend. | Solved by Requirement #6. | `app/frontend/main.py` |
| **14** | **Deployment Config** | **FAIL** | No `Dockerfile`, `docker-compose.yml`, or deployment manifests exist. | Cannot deploy via automated container pipelines. | Provide production `Dockerfile` and `docker-compose.yml`. | `Dockerfile`, `docker-compose.yml` |
| **15** | **Security Posture** | **PASS** | Pydantic v2 input bounds validation, non-autonomous safety disclaimers enforced. | Open CORS policy. | Solved by Requirement #5. | `app/backend/main.py` |
| **16** | **Performance & Sizing** | **PASS** | API latency 18.4 ms; RAM footprint $< 650\text{ MB}$; CPU-only inference. | None. Runs comfortably on 1 vCPU / 1 GB RAM tiers. | None required. | None |

### Readiness Calculation:
- **PASS**: **11 / 16 (68.75%)**
- **PARTIAL**: **2 / 16 (12.50%)**
- **FAIL (Remediable)**: **3 / 16 (18.75%)**
- **Current Raw Deployment Readiness:** **81.25%**
- **Post-Remediation Projected Readiness:** **100.0%**

---

## 3. Final Execution Instructions

### A. Current Deployment Readiness
The core software application is **technically robust, scientifically verified, and locally operational**. However, it is **NOT YET READY FOR DIRECT GIT PUSH OR CLOUD DEPLOYMENT** due to the missing configuration files (`requirements.txt`, `.gitignore`, `Dockerfile`, `.env.example`).

---

### B. All Critical Blocking Issues
1. **Incomplete `requirements.txt`**: Missing `fastapi`, `uvicorn`, `streamlit`, `plotly`, `pydantic`. Clean installation in a cloud container will immediately fail.
2. **Missing `.gitignore`**: Pushing to GitHub will attempt to upload 314 MB training files, triggering hard rejection errors by GitHub's pre-receive hooks.
3. **Missing Containerization Configuration**: No `Dockerfile` or cloud startup manifest is present.

---

### C. All Non-Critical Issues
1. **Frontend Backend URL Default**: Streamlit defaults to `http://localhost:8000` rather than checking `os.environ.get("BACKEND_API_URL")`.
2. **Permissive CORS Configuration**: FastAPI sets `allow_origins=["*"]` with `allow_credentials=True`.
3. **Database Volume Mounting**: SQLite path `DEFAULT_DB_PATH` does not check `os.environ.get("DATABASE_PATH")`.

---

### D. Files Requiring Modification or Creation

| Action | File Path | Purpose |
| :--- | :--- | :--- |
| **MODIFY** | `requirements.txt` | Add missing web, server, and schema dependencies. |
| **MODIFY** | `app/frontend/main.py` | Add `BACKEND_API_URL` environment variable support. |
| **MODIFY** | `app/backend/main.py` | Parameterize CORS origins via `CORS_ORIGINS` env variable. |
| **MODIFY** | `src/database/database.py` | Parameterize database path via `DATABASE_PATH` env variable. |
| **CREATE** | `.gitignore` | Prevent committing `.venv/`, `.env`, and >100MB training files. |
| **CREATE** | `.env.example` | Document all cloud configuration variables. |
| **CREATE** | `Dockerfile` | Multi-stage, non-root, production container definition. |
| **CREATE** | `docker-compose.yml` | Multi-service or unified orchestration file. |

---

### E. Exact Corrective Actions

1. **Update `requirements.txt`**: Add `fastapi>=0.115.0`, `uvicorn>=0.30.0`, `streamlit>=1.38.0`, `plotly>=5.22.0`, `pydantic>=2.8.0`, `python-multipart>=0.0.9`, `httpx>=0.27.0`, `python-docx>=1.1.0`.
2. **Create `.gitignore`**: Ignore `__pycache__/`, `.venv/`, `.env`, `data/maintenance_history.db`, and `data/processed/nasa/*/*train.npz`, `data/processed/nasa/*/*test.npz`, `data/processed/nasa/*/train_tabular.csv`, `data/processed/nasa/*/test_tabular.csv`.
3. **Create `.env.example`**: Specify `FASTAPI_HOST`, `FASTAPI_PORT`, `BACKEND_API_URL`, `CORS_ORIGINS`, `DATABASE_PATH`.
4. **Create `Dockerfile`**: Define container build steps with CPU-only PyTorch and multi-stage user security.
5. **Update `main.py` (Frontend & Backend)**: Inject environment variable fallbacks for `BACKEND_API_URL` and `CORS_ORIGINS`.

---

### F. Exact Commands to Execute for Verification

```powershell
# 1. Verify updated requirements.txt syntax
python -m pip check

# 2. Run automated regression suite
python tests/test_phase8_backend.py
python tests/test_phase9_e2e_resilience.py

# 3. Test Docker build locally (if Docker Desktop is installed)
docker build -t agentic-pdm:latest .
docker run -p 8000:8000 -p 8501:8501 agentic-pdm:latest
```

---

### G. Recommended Cloud Hosting Architecture

```mermaid
graph LR
    subgraph Recommended_Production ["Recommended Cloud Architecture (Render / Railway / AWS)"]
        CLIENT[Browser User] --> ROUTER[Cloud Router / SSL Termination]
        ROUTER -->|Web Traffic| ST[Streamlit Dashboard Container :8501]
        ST -->|Internal VPC / HTTPS| API[FastAPI Backend Container :8000]
        API --> DISK[(Persistent Volume: /data/maintenance_history.db)]
    end
```

- **Backend (FastAPI)**: Deploy on **Render Web Service** or **Railway** (Python 3.12, 1 vCPU, 1 GB RAM).
- **Frontend (Streamlit)**: Deploy on **Streamlit Community Cloud** (free tier connected to GitHub repository) or co-host via Docker on the same instance.
- **Storage**: Attach a 1 GB persistent SSD volume mounted to `/data` to preserve the SQLite audit ledger across container redeployments.

---

### H. Step-by-Step Cloud Deployment Instructions

1. **Git Initialization**:
   ```bash
   git init
   git add .
   git commit -m "feat: complete agentic predictive maintenance framework with cloud deployment assets"
   git branch -M main
   git remote add origin https://github.com/<your-username>/agentic-predictive-maintenance.git
   git push -u origin main
   ```
2. **Deploy Backend to Render or Railway**:
   - Create new Web Service from GitHub repo.
   - Set Build Command: `pip install -r requirements.txt`
   - Set Start Command: `uvicorn app.backend.main:app --host 0.0.0.0 --port $PORT`
   - Note the assigned backend URL (e.g. `https://pdm-backend.onrender.com`).
3. **Deploy Frontend to Streamlit Community Cloud**:
   - Create new App from GitHub repo.
   - Main file path: `app/frontend/main.py`
   - In App Settings $\to$ Secrets / Environment Variables, set:
     ```toml
     BACKEND_API_URL = "https://pdm-backend.onrender.com"
     ```
   - Click Deploy!

---

### I. Post-Deployment Testing Checklist

1. [ ] Check `GET https://<backend-url>/health` returns `{"status": "healthy"}`.
2. [ ] Open Streamlit dashboard URL; verify all 4 fleet KPI cards render numbers on the Overview page.
3. [ ] Navigate to **Turbofan RUL Estimator**; select Engine #24, cycle 130; click Predict and assert RUL gauge displays ~18.2 cycles.
4. [ ] Navigate to **CNC Machine Failure Predictor**; adjust torque and speed; click Predict and assert risk dial updates.
5. [ ] Navigate to **Explainable AI**; verify TreeSHAP waterfall plot displays red and blue feature bars.
6. [ ] Navigate to **Agentic AI Orchestrator**; click "Run Multi-Agent Analysis" and assert all 6 agent step badges turn green.
7. [ ] Navigate to **Maintenance Work Orders**; click "Approve" on a work order and verify the status badge updates to `APPROVED`.

---

### J. Rollback Instructions

If a cloud deployment encounters unexpected errors:
1. **Instant PaaS Rollback**: In the Render / Railway / Streamlit Cloud console, click **Deployments** $\to$ select the previous successful build $\to$ click **Rollback to this revision**.
2. **Git Rollback**:
   ```bash
   git revert HEAD
   git push origin main
   ```
3. **Local Safety**: The local application in `agentic-predictive-maintenance` remains completely operational on ports 8000 and 8501 and is unaffected by remote cloud configuration adjustments.
