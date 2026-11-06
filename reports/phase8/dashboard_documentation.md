# Phase 8: Web Dashboard Documentation
## Agentic Predictive Maintenance: An Explainable Multivariate Time-Series Framework for Early Machine Failure Detection, Remaining Useful Life Estimation and Maintenance Decision Support

---

### 1. Overview and Design Philosophy

The Phase 8 Web Dashboard provides an enterprise-grade industrial decision-support interface built with **Streamlit** and backed by a high-performance **FastAPI** service. The application adheres to strict industrial human-factors design principles:

- **Clean Modern Industrial Aesthetics:** High-contrast slate dark theme (`#0F172A`, `#1E293B`), corporate blue primary accents (`#3B82F6`), emerald nominal indicators (`#10B981`), amber warning flags (`#F59E0B`), and rose critical indicators (`#EF4444`).
- **Information Hierarchy:** High-level fleet KPI metrics at a glance, progressing into interactive multivariate degradation curves, model attributions, and prescriptive maintenance action cards.
- **Dataset-Aware Logic:** Strict physical separation between continuous degradation modeling for the NASA C-MAPSS turbofan fleet and discrete stress condition modeling for the UCI AI4I 2020 CNC machining center.
- **Explainability First:** Quantitative predictions are never presented in isolation; every inference is paired with TreeSHAP feature attributions, anomaly reconstruction error gauges, and plain-text engineering briefs.
- **Human-in-the-Loop Governance:** Consequential maintenance interventions strictly enforce authorized human engineering review (Approve, Reject, Request Further Analysis) with SQLite database commit.

---

### 2. Application Architecture

```
+--------------------------------------------------------------------------------------------------+
|                                    STREAMLIT WEB DASHBOARD                                       |
|                                                                                                  |
|   +------------------------------------------------------------------------------------------+   |
|   | Global Sidebar: Dataset Selector (NASA / AI4I) | Unit Selector | Telemetry Input Mode     |   |
|   +------------------------------------------------------------------------------------------+   |
|                                                |                                                 |
|   +--------------------------------------------v---------------------------------------------+   |
|   | Header: Breadcrumbs | Platform Status Badge | System Architecture Indicators             |   |
|   +------------------------------------------------------------------------------------------+   |
|                                                |                                                 |
|   +--------------------------------------------v---------------------------------------------+   |
|   | 7 Specialized Operational Views:                                                         |   |
|   |  1. Overview: Fleet health KPIs, critical units, recent decision history table           |   |
|   |  2. Machine Monitoring: Interactive Plotly sensor degradation curves, stress radar       |   |
|   |  3. Predictive Analysis: Predicted RUL, calibrated failure probability, anomaly gauge   |   |
|   |  4. Explainable AI: TreeSHAP waterfall charts, feature rankings, diagnostic brief       |   |
|   |  5. Agentic AI Analysis: Interactive LangGraph multi-agent execution console             |   |
|   |  6. Maintenance Recommendations: Action cards, protocol references, review sign-off      |   |
|   |  7. History and Reports: SQLite audit queries, model benchmark matrix, report downloads  |   |
|   +------------------------------------------------------------------------------------------+   |
+------------------------------------------------|-------------------------------------------------+
                                                 | REST HTTP
                                                 v
+--------------------------------------------------------------------------------------------------+
|                                      FASTAPI BACKEND SERVICE                                     |
|                                                                                                  |
|  Endpoints: /health | /api/models | /api/predict/* | /api/anomaly/detect | /api/explain          |
|             /api/agents/analyze | /api/recommendations | /api/history | /api/recommendations/review|
|                                                                                                  |
|  Services: ModelService (Cached Checkpoints) | ExplainabilityManager | LangGraph StateGraph      |
|  Storage:  SQLite Database (`data/maintenance_history.db`)                                       |
+--------------------------------------------------------------------------------------------------+
```

---

### 3. Detailed Page Walkthrough

#### Page 1: Overview
- **Fleet Metrics:** Displays total monitored engines/machines, fleet average Remaining Useful Life (RUL), count of high-priority units ($RUL \le 30$ cycles or failure probability $\ge 75\%$), and anomaly counts.
- **Data Provenance Disclosure:** Explicit visual card identifying whether metrics derive from the official validation cohort or live API inference.
- **Recent Decision History:** Tabular feed of recent maintenance decisions queried directly from `data/maintenance_history.db` with status-colored tags (`APPROVED`, `PENDING`, `REJECTED`, `FURTHER_ANALYSIS_REQUESTED`).

#### Page 2: Machine Monitoring
- **NASA Turbofans:** Interactive multi-sensor Plotly chart plotting 30-cycle temporal sliding-window degradation profiles. Users can select and overlay individual channels (e.g., `sensor_11` HPC outlet temp, `sensor_4` LPT exhaust temp, `sensor_12` fan bypass). Terminal cycle sensor readings are tabulated with physical subsystem mapping.
- **UCI AI4I CNC Machining:** Operational stress radar chart and parameter table. Reflects discrete machine operating parameters (air temp, process temp, rotational speed, torque, tool wear, heat dissipation index) without assuming false continuous time-series trajectories.

#### Page 3: Predictive Analysis
- **Supervised Outputs:**
  - NASA C-MAPSS: Predicted Remaining Useful Life in operational flight cycles, delta against ground-truth, and model confidence bounds.
  - UCI AI4I 2020: Breakdown probability percentage, predicted operational label (`Machine Failure` vs `Normal Operation`), and explicit calibration disclosure confirming isotonic calibration against decision threshold $0.7969$.
- **Early-Warning Distribution:** Proximity probability bar chart across 10, 20, and 30 flight cycles.
- **Unsupervised Anomaly Gauge:** Plotly radial gauge displaying multivariate autoencoder reconstruction error against nominal threshold ($0.0370$).

#### Page 4: Explainable AI
- **TreeSHAP Attributions:** Horizontal waterfall bar chart visualizing feature contributions. Red bars indicate features accelerating degradation or elevating failure risk; green bars indicate protective nominal factors.
- **Ranked Contributions:** Tabular display of features sorted by absolute Shapley weight.
- **Physical Domain Grounding:** Explanatory notes linking sensor attributions to thermodynamic/electromechanical degradation mechanisms (e.g., compressor blade erosion, spindle bearing friction).

#### Page 5: Agentic AI Analysis
- **Multi-Agent Orchestration:** Real-time console executing the compiled LangGraph StateGraph:
  1. *Supervisor Agent:* Validates tensor dimensions and checks sensor boundary conditions.
  2. *Prediction Agent:* Runs trained XGBoost/LSTM models.
  3. *Anomaly Detection Agent:* Evaluates unsupervised autoencoder reconstruction error.
  4. *Explanation Agent:* Computes TreeSHAP feature attributions.
  5. *Maintenance Agent:* Resolves priority conflict matrix and formulates structured actions.
  6. *Result Validation Node:* Compiles structured JSON output and verifies schema integrity.
- **Interactive Step Trace:** Visual progress bar and live status badges displaying agent transitions.
- **Deep-Dive Tabs:** Dedicated inspection tabs for each agent's internal state and tensor outputs.

#### Page 6: Maintenance Recommendations
- **Prescriptive Protocols:** Detailed action cards identifying specific physical tasks (e.g., borescope inspection of HPC outlet, cutting insert replacement), target assembly, urgency horizon, and official reference standards (EASA/FAA Part 145, ISO 10791, ISO 3685).
- **Supporting Evidence List:** Quantitative linkage to supervised predictions, anomaly scores, and attribution drivers.
- **Certified Engineer Authorization Panel:** Interactive sign-off form allowing certified reliability engineers to record their badge ID, review notes, and execute authorization decisions:
  - `✅ Approve Recommendation`
  - `❌ Reject Recommendation`
  - `🔍 Request Further Analysis`
- Immediate write commit to the SQLite database with instant UI status refresh.

#### Page 7: History and Reports
- **Database Audit Trail:** Interactive table of all historical recommendations stored in `data/maintenance_history.db` with filtering by review status and machine ID.
- **Model Performance Matrix:** Comparative benchmark tables detailing RMSE, MAE, R², Accuracy, F1-Score, and ROC-AUC across all Phase 2 and Phase 3 models.
- **Report Download Center:** One-click markdown downloads for all Phase 2 through Phase 7 technical reports.

---

### 4. Input Flexibility and CSV Upload Verification

The dashboard supports two distinct data ingestion paths:
1. **Pre-loaded Validation Sample:** Zero-configuration instant access to official NASA FD001 validation engines and AI4I machine tool records.
2. **Custom CSV Upload:** Users can upload custom telemetry files. The backend enforces strict schema validation:
   - Verifies exact feature column names against the training pipeline.
   - Rejects missing, mismatched, or extra columns.
   - Never silently rearranges or fabricates input features.

---

### 5. Startup Commands

#### Method 1: Single-Command Full Stack Launcher
```bash
# Run both FastAPI backend and Streamlit frontend together:
.\.venv\Scripts\python.exe run_dashboard.py
```

#### Method 2: Separate Services Launcher
```bash
# Terminal 1 - Start FastAPI Backend (Port 8000):
.\.venv\Scripts\python.exe -m uvicorn app.backend.main:app --host 0.0.0.0 --port 8000

# Terminal 2 - Start Streamlit Dashboard (Port 8501):
.\.venv\Scripts\python.exe -m streamlit run app/frontend/main.py --server.port 8501
```

- **Web Dashboard URL:** `http://localhost:8501`
- **FastAPI Interactive Swagger Docs:** `http://localhost:8000/docs`
