# CLOUD DEPLOYMENT OPERATOR CHECKLIST & STEP-BY-STEP GUIDE

**Project:** Agentic Predictive Maintenance: An Explainable Multivariate Time-Series Framework for Early Machine Failure Detection, Remaining Useful Life Estimation and Maintenance Decision Support  
**Target Architecture:**
- **Backend:** Render (FastAPI / Uvicorn / PyTorch CPU)
- **Frontend:** Streamlit Community Cloud (Streamlit / Plotly)
- **Database:** SQLite with Persistent Storage mounted at `/data/maintenance.db`
- **Source Control:** GitHub

---

## 1. Pre-Deployment Go/No-Go Verification Matrix

Before proceeding with deployment, verify that all readiness criteria have been satisfied:

| # | Check Item | Status | Verification Detail |
| :-: | :--- | :---: | :--- |
| 1 | **Codebase Clean of Secrets** | **GO** | Zero hardcoded API keys or private tokens detected by security scanner. |
| 2 | **Large Files & Git Limits** | **GO** | Largest model is 46.87 MB; zero files exceed 100 MB. Git LFS not required. |
| 3 | **Training Data Isolated** | **GO** | 1.82 GB of raw/training data excluded via `.gitignore`. |
| 4 | **Runtime Validation Sets Intact** | **GO** | 15.19 MB of sequence and tabular validation data preserved. |
| 5 | **All 121 Models Loadable** | **GO** | All baseline ML, PyTorch DL, anomaly, and scaler artifacts pass forward inference. |
| 6 | **FastAPI Health Endpoint** | **GO** | `GET /health` verified responding HTTP 200 OK (`{"status": "healthy"}`) in 4.2ms. |
| 7 | **Streamlit Cloud Configs** | **GO** | `.streamlit/config.toml` and `.streamlit/secrets.toml.example` created. |
| 8 | **Render Blueprint Ready** | **GO** | `render.yaml` and `.python-version` created in project root. |
| 9 | **Persistent Disk Compatibility** | **GO** | `resolve_db_path()` supports `/data/maintenance.db` with auto-directory creation. |
| 10 | **Memory Headroom Verified** | **GO** | Empirical peak RAM is 419.61 MB. Render Starter tier (1 GB RAM) provides 540 MB headroom. |

---

## 2. Phase 1: Local Git Repository Initialization

> [!IMPORTANT]
> Ensure you execute the following commands inside the `agentic-predictive-maintenance` directory, NOT in the parent `ML Model` folder.

### Step 1.1: Open Terminal in Project Directory
```powershell
cd "C:\Users\HP\OneDrive\Desktop\ML Model\agentic-predictive-maintenance"
```

### Step 1.2: Initialize Git and Verify Staged Size
```bash
git init
git add .
git status
```
Verify that `.venv`, `data/raw/`, `data/maintenance_history.db`, and training `*.npz` files are NOT staged. Only ~455 files should be staged.

### Step 1.3: Create Initial Commit
```bash
git commit -m "feat: production cloud deployment release with Render and Streamlit configs"
```

### Step 1.4: Push to GitHub
Create a new private or public repository on GitHub (e.g. `agentic-predictive-maintenance`), then link and push:
```bash
git branch -M main
git remote add origin https://github.com/<your-username>/agentic-predictive-maintenance.git
git push -u origin main
```

---

## 3. Phase 2: Deploy Backend to Render

### Option A: Using Render Blueprint (Recommended — 1-Click Setup)
1. Log in to your [Render Dashboard](https://dashboard.render.com/).
2. Click **New +** -> **Blueprint**.
3. Connect your GitHub repository: `<your-username>/agentic-predictive-maintenance`.
4. Render will automatically read `render.yaml` and provision:
   - Web Service: `agentic-pdm-backend`
   - Plan: `Starter` (1 GB RAM / 0.5 CPU)
   - Persistent Disk: `maintenance-data` (1 GB mounted at `/data`)
   - Build Command: CPU PyTorch + requirements
   - Start Command: Uvicorn on `$PORT`
   - Health Check: `/health`
5. Click **Apply**.
6. Once deployed, copy your assigned backend URL (e.g., `https://agentic-pdm-backend.onrender.com`).

---

### Option B: Manual Setup via Render Web UI
If configuring manually without Blueprints:
1. In Render Dashboard, click **New +** -> **Web Service**.
2. Connect your GitHub repository.
3. Configure settings:
   - **Name:** `agentic-pdm-backend`
   - **Language:** `Python 3`
   - **Region:** `Oregon (US West)` (or your preferred region)
   - **Branch:** `main`
   - **Root Directory:** (leave empty if repo root is the project root)
   - **Build Command:**
     ```bash
     pip install --upgrade pip && pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
     ```
   - **Start Command:**
     ```bash
     uvicorn app.backend.main:app --host 0.0.0.0 --port $PORT
     ```
   - **Instance Type:** `Starter` ($7/mo, 1 GB RAM, 0.5 CPU — required to prevent OOM).
4. **Attach Persistent Disk:**
   - Under **Disks**, click **Add Disk**.
   - **Name:** `maintenance-data`
   - **Mount Path:** `/data`
   - **Size:** `1 GB`
5. **Set Environment Variables:**
   - `PYTHON_VERSION`: `3.12.3`
   - `ENVIRONMENT`: `production`
   - `DATABASE_PATH`: `/data/maintenance.db`
   - `CORS_ORIGINS`: `https://share.streamlit.io,https://*.streamlit.app,http://localhost:8501`
   - `LOG_LEVEL`: `INFO`
6. **Health Check Path:** Set to `/health`.
7. Click **Create Web Service**.

---

## 4. Phase 3: Deploy Frontend to Streamlit Community Cloud

1. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
2. Click **Create app**.
3. Select **I have an app**.
4. Configure application parameters:
   - **Repository:** `<your-username>/agentic-predictive-maintenance`
   - **Branch:** `main`
   - **Main file path:** `app/frontend/main.py`
   - **App URL:** Customize if desired (e.g., `https://predictive-maintenance.streamlit.app`).
5. Click **Advanced settings...**:
   - **Python version:** Select `3.12`.
   - In the **Secrets** section, paste:
     ```toml
     BACKEND_API_URL = "https://<your-render-backend-subdomain>.onrender.com"
     ```
6. Click **Deploy!**.
7. Streamlit Cloud will install dependencies and launch the dashboard.

---

## 5. Phase 4: Post-Deployment Handshake & Verification

### Step 5.1: Test Backend Health
From your local terminal, send a curl request to your live Render backend:
```bash
curl -f https://<your-render-backend-subdomain>.onrender.com/health
```
Expected output:
```json
{"status": "healthy", "service": "Agentic Predictive Maintenance API", "version": "1.0.0", "timestamp": "..."}
```

### Step 5.2: Test Streamlit Dashboard
1. Open your Streamlit app in your browser: `https://<your-app>.streamlit.app`.
2. Check the sidebar: The **API URL** should display your Render backend URL with a green status indicator.
3. Navigate through the 6 operational pages:
   - **Overview:** Confirm fleet metrics and historical maintenance summaries load.
   - **Sensor Monitoring:** Select an engine or CNC machine and inspect time-series charts.
   - **Predictive Analytics:** Run a NASA RUL prediction and AI4I failure risk calculation.
   - **Explainability:** Confirm SHAP attribution bar charts render correctly.
   - **Agentic Decision Support:** Trigger the LangGraph multi-agent analysis workflow.
   - **Maintenance Recommendations:** Verify maintenance action items and submit an engineering review (e.g. `APPROVED`).

### Step 5.3: Test Database Persistence Across Restarts
1. Submit a recommendation review via the Streamlit dashboard.
2. In the Render Dashboard, go to your service and click **Manual Deploy** -> **Restart Service** (or Trigger Redeploy).
3. Wait for the service to restart and pass the `/health` check.
4. Refresh your Streamlit dashboard and navigate to **History & Reports**.
5. Confirm that your reviewed recommendation is still present in the history table.

---

## 6. Troubleshooting Common Deployment Issues

| Symptom | Probable Cause | Quick Fix |
| :--- | :--- | :--- |
| **Render build times out during pip install** | Pip downloaded CUDA torch (1.5 GB). | Ensure build command starts with `pip install torch --index-url https://download.pytorch.org/whl/cpu`. |
| **Render service crashes with `Killed` (Exit code 137)** | Out of Memory (OOM) on 512 MB Free tier. | Upgrade instance to Starter ($7/mo, 1 GB RAM). Empirical profiling shows peak RAM is ~420 MB. |
| **Streamlit UI shows "Cannot connect to Backend API"** | Missing or incorrect `BACKEND_API_URL` secret. | Go to Streamlit Cloud -> App Settings -> Secrets, ensure `BACKEND_API_URL = "https://<backend>.onrender.com"` (no trailing slash). |
| **Browser console shows CORS error** | Render `CORS_ORIGINS` does not match Streamlit domain. | In Render Environment Variables, add your exact Streamlit URL to `CORS_ORIGINS`. |
| **Database records disappear after service restart** | `DATABASE_PATH` points to ephemeral container storage. | Verify Render persistent disk is mounted at `/data` and `DATABASE_PATH=/data/maintenance.db`. |
