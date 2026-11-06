# 06. Streamlit Frontend Deployment Audit
**Project:** Agentic Predictive Maintenance Framework  
**Document ID:** AUDIT-06-STREAMLIT-FRONTEND  
**Auditor:** Senior Frontend & Cloud Deployment Specialist  
**Audit Date:** November 2026  
**Status:** AUDITED — MINOR CONFIGURATION TWEAK REQUIRED  

---

## 1. Frontend Architecture & Entrypoint

The web dashboard is implemented in `app/frontend/main.py` using Streamlit:

- **Module Path:** `app/frontend/main.py`
- **Application Framework:** Streamlit 1.66.1
- **Standalone Cloud Startup Command:**
  ```bash
  streamlit run app/frontend/main.py --server.port 8501 --server.headless true --server.address 0.0.0.0
  ```

---

## 2. Page Navigation & Component Verification

All 7 operational view pages were verified:
1. **Overview Dashboard (`page_overview.py`)**: Metrics render cleanly without `NameError` (verified following `import numpy as np` patch).
2. **Machine Telemetry & Degradation Monitoring (`page_monitoring.py`)**: Plotly interactive charts display sensor time-series trends without cross-engine distortion.
3. **Predictive Analytics (`page_predictive.py`)**: Semi-circular gauges, RUL values, and calibrated failure probabilities render correctly.
4. **Explainable AI Diagnostics (`page_explainability.py`)**: TreeSHAP waterfall charts and feature importance bars load within 200 ms.
5. **Agentic AI Orchestration (`page_agentic.py`)**: Multi-agent state machine visualizer displays dynamic execution steps.
6. **Prescriptive Recommendations (`page_recommendations.py`)**: Action cards, safety disclaimers, and human review sign-off buttons execute properly.
7. **History & Reports (`page_history.py`)**: SQLite audit queries load with color-coded status badges and CSV export functionality.

---

## 3. Cloud Backend Communication Audit

### 3.1 Backend URL Discovery: ACTION REQUIRED
- **Current Implementation in `app/frontend/main.py` and `sidebar.py`**:
  ```python
  api_url = st.session_state.get("api_url", "http://localhost:8000")
  ```
- **Issue**: In cloud deployments where the Streamlit frontend runs on Streamlit Community Cloud and the FastAPI backend runs on a separate URL (e.g., `https://predictive-maintenance-api.onrender.com`), the frontend defaults to `localhost:8000`, causing network timeouts unless the user manually types the remote URL in the sidebar text input on every session.
- **Remediation**: Update `app/frontend/main.py` to check `os.environ.get("BACKEND_API_URL")` or `st.secrets.get("BACKEND_API_URL")` before falling back to `http://localhost:8000`:
  ```python
  default_api_url = os.environ.get("BACKEND_API_URL") or "http://localhost:8000"
  api_url = st.session_state.get("api_url", default_api_url)
  ```

---

## 4. File Upload Security
- The CSV file uploader in `sidebar.py` validates file types (`type=["csv"]`).
- `validate_and_parse_csv` checks column names against expected sensor schemas and rejects malformed inputs gracefully.
