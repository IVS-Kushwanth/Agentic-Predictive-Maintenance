# Phase 9: Live Demonstration Script & Operational Walkthrough
## Agentic Predictive Maintenance: System Demonstration Protocol for Evaluation Committee

---

### 1. Demonstration Purpose and Setup

This document provides a structured, step-by-step demonstration script for presenting the **Agentic Predictive Maintenance** platform to academic evaluation committees, industrial stakeholders, and technical examiners.

#### Prerequisites & Environment Verification:
- Operating System: Windows / Linux / macOS
- Python Environment: Python 3.12 Virtual Environment (`.venv`)
- Network: Localhost (no external cloud dependencies required)
- Service Ports: FastAPI (`http://localhost:8000`), Streamlit (`http://localhost:8501`)

---

### 2. Live Walkthrough Protocol

#### Step 1: System Startup
1. Open PowerShell or Terminal in the repository root:
   ```powershell
   cd "agentic-predictive-maintenance"
   .\.venv\Scripts\python.exe run_dashboard.py
   ```
2. **Examiner Talking Point:**
   > *"We launch both the high-performance FastAPI backend service and the Streamlit web dashboard via a single unified runner. The backend pre-loads model checkpoints into an in-memory singleton cache to eliminate reload latency on concurrent requests."*
3. Open browser to `http://localhost:8501`. Confirm platform status badge displays: **SYSTEM OPERATIONAL**.

---

#### Step 2: Fleet Overview & Data Provenance (Page 1)
1. Select **1. Overview** on the sidebar.
2. In the sidebar dropdown, select **NASA C-MAPSS (Turbofans)**.
3. Observe Fleet Health KPIs:
   - Total Monitored Turbofans: `100 Units`
   - Fleet Average RUL: `~108.5 Cycles`
   - High-Priority Engines: `18 Units`
4. **Examiner Talking Point:**
   > *"Notice the prominent data source disclosure badge: all figures derive from the official NASA C-MAPSS FD001 validation cohort and real model predictions. We never present synthetic or fabricated metrics."*
5. Toggle to **UCI AI4I 2020 (CNC Machines)**. Observe discrete manufacturing breakdown KPIs (2,000 test shifts, 68 breakdowns, 3.4% prevalence).

---

#### Step 3: Machine Telemetry & Degradation Monitoring (Page 2)
1. Switch sidebar to **2. Machine Monitoring**.
2. Select **NASA C-MAPSS** and choose unit **`Engine-FD001-099`** (a heavily degraded engine near failure).
3. Observe the 30-cycle sliding window multi-sensor degradation curves.
4. Check sensor channels `sensor_11` (HPC outlet static temp T30) and `sensor_4` (LPT exhaust temp).
5. **Examiner Talking Point:**
   > *"Notice the clear non-linear thermal climb over the final 30 operational cycles. The High-Pressure Compressor outlet temperature climbs steadily, indicating physical blade tip erosion and gas leakage. For NASA, we display genuine temporal degradation curves; for AI4I, we present operating stress radar charts without making false continuous time-series assumptions."*

---

#### Step 4: Predictive Analytics & Condition Assessment (Page 3)
1. Navigate to **3. Predictive Analysis**.
2. Inspect the **Predicted RUL**:
   - Model outputs approximately `14.2 Cycles` (Ground Truth: `15.0 Cycles`, Error: `-0.8 Cycles`).
3. Inspect the **Early-Warning Horizon Breakdown**:
   - Probability of failure within 10 cycles: `> 80%`.
4. Inspect the **Multivariate Anomaly Score**:
   - LSTM Autoencoder Reconstruction Error: `0.2105` (well above nominal threshold `0.0370`). Severity badge: **High anomaly**.
5. **Examiner Talking Point:**
   > *"Our system provides both supervised RUL estimates and unsupervised anomaly detection. Notice the calibration disclosure on AI4I: probabilities are strictly calibrated against validation loss curves, avoiding the hazardous false certainty common in raw neural network outputs."*

---

#### Step 5: Explainable AI & Diagnostic Attributions (Page 4)
1. Navigate to **4. Explainable AI**.
2. Review the **Diagnostic Decision Brief**:
   > *"Model predicts RUL of 14.2 cycles. Primary driver is 'sensor_11' with a contribution of +4.12 cycles reducing machine operational lifespan."*
3. Inspect the **TreeSHAP Horizontal Waterfall Chart**:
   - `sensor_11` appears in bright red at the top, followed by `sensor_4`.
4. **Examiner Talking Point:**
   > *"This satisfies the critical Explainable AI requirement. We don't just tell the maintenance engineer that the engine is failing; we show them game-theoretically proven Shapley values pointing directly to the High-Pressure Compressor as the physical root cause."*

---

#### Step 6: LangGraph Multi-Agent Orchestration (Page 5)
1. Navigate to **5. Agentic AI Analysis**.
2. Click **🚀 Run Multi-Agent Analysis**.
3. Watch the real-time execution trace progress across the 6 state graph nodes:
   - `Supervisor Agent` $\to$ `Prediction Agent` $\to$ `Anomaly Agent` $\to$ `Explanation Agent` $\to$ `Maintenance Agent` $\to$ `Result Validation Node`.
4. Observe the compiled **Executive Decision Brief**:
   - Priority: **CRITICAL REVIEW REQUIRED**
   - Status: **IMMINENT_EXHAUSTION**
5. **Examiner Talking Point:**
   > *"Here, the system acts agentically. The Supervisor Agent validates tensor bounds; specialized agents compute inference and Shapley attributions; and the Maintenance Agent resolves priority conflict matrices. Notice that LLMs do NOT invent numerical predictions; all metrics derive from deterministic, trained models."*

---

#### Step 7: Prescriptive Maintenance & Certified Human Sign-Off (Page 6)
1. Navigate to **6. Maintenance Recommendations**.
2. Review the structured action cards:
   - Action 1: **Urgent Borescope Inspection of HPC Outlet** (Reference: *EASA / FAA Part 145 Overhaul Standards*).
   - Action 2: **Standby LRU Turbofan Pre-Allocation** (Urgency: *Immediate*).
3. Scroll down to the **Certified Engineer Authorization Panel**.
4. Enter Engineer Badge: `Chief Inspector Dave (A&P License #384910)`.
5. Enter Review Notes: `Inspected borescope camera logs. Thermal erosion confirmed. Work order authorized.`
6. Click **✅ Approve Recommendation**.
7. Observe instant UI update: Review Status changes to **APPROVED** with timestamp.
8. **Examiner Talking Point:**
   > *"This enforces our industrial non-autonomous safety guardrail. The AI provides prescriptive decision support, but physical intervention requires certified human sign-off, recorded immutably in SQLite."*

---

#### Step 8: Historical Audit Trail (Page 7)
1. Navigate to **7. History and Reports**.
2. Under the **Recommendation History** tab, observe the freshly approved record:
   - Recommendation ID: `REC-...`
   - Machine ID: `Engine-FD001-099`
   - Priority: `Critical review required`
   - Review Status: `APPROVED`
   - Reviewed By: `Chief Inspector Dave`
3. Click the **Model Performance Benchmarks** tab to show the committee the full comparative evaluation matrix.
4. Click the **Downloadable Technical Reports** tab to demonstrate one-click markdown exports.

---

### 3. Distinction: Simulated Telemetry vs Real-Time Industrial Deployment

| Operating Dimension | Simulated Academic Demonstration | Real-Time Industrial Deployment |
| :--- | :--- | :--- |
| **Data Ingestion** | Ingested from verified NASA C-MAPSS and AI4I validation cohorts | Ingested via streaming industrial IoT protocols (MQTT, OPC-UA, Kafka) |
| **Compute Hardware** | Local CPU / Workstation with in-memory caching | High-availability Kubernetes cluster or on-premise industrial edge server |
| **Inference Frequency**| On-demand batch execution or UI click | Continuous sub-second streaming inference on incoming telemetry frames |
| **Action Authority** | Human sign-off logged to local SQLite | Signed work orders dispatched automatically to enterprise ERP/EAM (SAP PM, Maximo) |
