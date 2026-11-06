# End-User Application Operation Manual
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
