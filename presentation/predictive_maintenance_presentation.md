# Agentic Predictive Maintenance: Final Academic Presentation Slide Deck

---

## Slide 1: Title Slide
### **Agentic Predictive Maintenance**
#### *An Explainable Multivariate Time-Series Framework for Early Machine Failure Detection, Remaining Useful Life Estimation and Maintenance Decision Support*

- **Candidate / Presenter:** Advanced Machine Learning Engineering Team
- **Department:** Computer Science & Industrial AI Engineering
- **Advisors / Committee:** Capstone Evaluation Panel
- **Date:** November 2026
- **Project Status:** Complete, Fully Integrated, Production-Tested (40/40 Tests Passed)

*Speaker Notes:*
Good morning, respected members of the evaluation committee. Today, I am proud to present our final capstone research project: "Agentic Predictive Maintenance". This work bridges the critical divide between academic time-series modeling and actionable industrial decision-support through an explainable, multi-agent framework.

---

## Slide 2: Industrial Context & Background
### **The Evolution of Industrial Maintenance**

1. **Run-to-Failure (Corrective):**
   - High machine destruction, unplanned downtime, astronomical costs.
2. **Scheduled (Preventive / Time-Based):**
   - Conservative calendar or cycle replacements; discards up to 40% of remaining asset life.
3. **Predictive Maintenance (Condition-Based / PdM):**
   - Ingests real-time multivariate sensor telemetry to predict failure and estimate Remaining Useful Life (RUL).

*Economic Impact:*
- Aircraft on Ground (AOG) costs: Up to \$500,000 per unscheduled engine removal event.
- CNC Spindle Tool Rupture: Scraps high-precision aerospace workpieces and damages machine spindles.

---

## Slide 3: Problem Statement
### **Why do existing predictive maintenance systems fail in practice?**

1. **Predictive Target Leakage:**
   - Overlapping time-series sliding windows cross train/test splits, causing models to memorize stationary unit baselines rather than learning degradation dynamics.
2. **The "Black-Box" Interpretability Deficit:**
   - Deep models output a number (e.g. $\text{RUL} = 14.2$) without transparent physical justification. Plant engineers cannot justify grounding aircraft without evidence.
3. **Severe Class Imbalance & Miscalibration:**
   - Real breakdowns represent $< 4\%$ of operational data. Models optimize for accuracy by predicting "Normal" 99% of the time.
4. **Lack of Prescriptive Actionability:**
   - Systems predict failure, but do not provide prioritized, evidence-linked action plans mapped to established engineering standards.

---

## Slide 4: Identified Research Gaps
### **Critical Limitations in Contemporary Literature**

| Contemporary PdM Literature | This Project's Solution |
| :--- | :--- |
| **Random train/test splits** cause severe data leakage across time windows | **Engine-Level Stratification:** Zero cross-engine window contamination |
| **Equal penalty on early and late RUL errors** via standard MSE | **Deep Recurrent Architectures** optimizing the exponential NASA Asymmetric Loss |
| **Uncalibrated probabilities** leading to false operational confidence | **Isotonic Calibration** establishing reliable decision thresholds ($\tau^* = 0.7969$) |
| **Static dashboards** returning disconnected predictions | **LangGraph Multi-Agent Orchestration** with SQLite human review governance |

---

## Slide 5: Project Objectives
### **Core Engineering and Research Goals**

1. **Leakage-Free Foundations:** Build isolated preprocessing pipelines for NASA C-MAPSS and UCI AI4I 2020.
2. **Multivariate RUL & Breakdown Modeling:** Benchmark 32 ML baselines and 18 deep learning architectures.
3. **Unsupervised Anomaly Detection:** Train Autoencoders and Isolation Forests strictly on nominal operational states.
4. **Transparent Explainable AI (XAI):** Deploy TreeSHAP and Integrated Gradients for local and global sensor attributions.
5. **Agentic Orchestration:** Develop a stateful multi-agent StateGraph via LangGraph.
6. **Prescriptive Recommendations:** Implement a 5-tier priority engine with conflict resolution and SQLite audit trails.
7. **Full-Stack Delivery:** Deploy a FastAPI REST backend and a 7-page Streamlit web dashboard.

---

## Slide 6: Dataset Strategy
### **Two Independent Industrial Benchmarks**

#### **1. NASA C-MAPSS (Primary Dataset):**
- **System:** Commercial high-bypass turbofan engines (90,000 lb thrust class).
- **Subsets:** FD001 (1 flight regime, HPC degradation), FD002–FD004.
- **Data Volume:** 100 test engines ($13,096$ cycles); 80 train ($16,340$ cycles); 20 val ($4,291$ cycles).
- **Task:** Continuous Remaining Useful Life (RUL) regression & Early-Warning ($H \in \{10, 20, 30\}$ cycles).

#### **2. UCI AI4I 2020 (Secondary Dataset):**
- **System:** Synthetic digital twin of a CNC milling machine center.
- **Data Volume:** 10,000 records ($8,000$ train, $2,000$ test, 3.39% failure prevalence).
- **Task:** Machine breakdown classification & 5 distinct physical failure modes (OSF, PWF, HDF, TWF, RNF).

*Strict Rule:* The two datasets originate from fundamentally different physical systems and are strictly separated.

---

## Slide 7: Leakage-Free Preprocessing Methodology
### **Mathematical Isolation of Temporal Features**

- **Constant Sensor Pruning:** 7 sensors with zero variance dropped (`sensor_1`, `5`, `6`, `10`, `16`, `18`, `19`).
- **Backward-Looking Rolling Features Only:**
  $$\mu_t^{(w)} = \frac{1}{\min(t, w)} \sum_{k=0}^{\min(t, w)-1} x_{t-k}, \quad w \in \{5, 10, 20\}, \quad \text{min\_periods}=1$$
- **153 Active Engineered Features:** Expanding 14 raw sensors into rolling means, standard deviations, and delta values.
- **Joblib Scalers:** Scalers are fitted **strictly on training engines** and serialized to disk.
- **3D Sequence Generation:** Sliding windows of length $L \in \{30, 50, 60\}$ are constructed strictly within individual engine boundaries.

---

## Slide 8: Machine Learning Baseline Models (Phase 2)
### **Comprehensive Evaluation Across 32 Baseline Models**

- **NASA RUL Regression:**
  - Evaluated: Linear Regression, Random Forest, Gradient Boosting, XGBoost.
  - **Best Model:** XGBoost Regressor achieved test terminal **RMSE of 22.50 cycles** and **MAE of 15.35 cycles**, training in 1.55 seconds (43x faster than GBDT).
- **NASA Early-Warning Multi-Horizon Classification:**
  - Horizon $H=10$ (Emergency): XGBoost PR-AUC = **0.8220**, Recall = **82.35%**.
  - Horizon $H=30$ (Strategic): Random Forest PR-AUC = **0.8958**, F1 = **0.7780**.
- **UCI AI4I Machine Failure Classification:**
  - **Best Model:** Gradient Boosting Classifier with threshold tuning ($\tau^* = 0.485$).
  - **Test Performance:** Precision = **98.15%**, Recall = **77.94%**, F1 = **0.8689**, PR-AUC = **0.8679**.

---

## Slide 9: Time-Series Deep Learning Architectures (Phase 3)
### **Why Deep Learning Wins the NASA Asymmetric Score**

| Model Architecture | Input Window ($L$) | Test Terminal RMSE | Test Terminal MAE | **NASA Asymmetric Score ($S$)** |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost Regressor** | Tabular ($L=1$) | 22.50 cycles | 15.35 cycles | 9,525.5 |
| **Gradient Boosting** | Tabular ($L=1$) | 22.50 cycles | 16.09 cycles | 3,202.4 |
| **LSTM Regressor** | Sequence ($L=30$) | 21.98 cycles | 16.15 cycles | **1,464.3** (-84.6%) |
| **LSTM Regressor** | Sequence ($L=60$) | **20.02 cycles** | **14.83 cycles** | **889.4** (-90.7%) |

*The Secret of the NASA Asymmetric Penalty:*
$$S = \sum s_i, \quad s_{\text{late}} = \exp(d/10)-1 \quad \text{vs} \quad s_{\text{early}} = \exp(-d/13)-1$$
- Decision trees evaluate static cycles; transient noise can trigger a sudden 30-cycle jump in predicted RUL, incurring huge late penalties.
- Recurrent LSTM cells accumulate internal hidden state memory ($h_t$) monotonically, preventing sharp overestimates.

---

## Slide 10: Unsupervised Anomaly Detection (Phase 4)
### **Establishing Nominal Boundaries Without Failure Label Contamination**

1. **Isolation Forest (Tree Isolation):**
   - Outperformed random chance ($0.0340$) by **$9.1\times$** on AI4I with test PR-AUC = **0.3079** and ROC-AUC = **0.8824**.
2. **Multivariate LSTM Autoencoder (PyTorch):**
   - Stacked 2-layer encoder-decoder trained on early healthy flight cycles.
   - Calibrated nominal reconstruction threshold $\tau = \mathbf{0.0370}$.
   - Max nominal score in training = $0.0362$. Any test cycle exceeding $0.0370$ flags sensor anomaly.
3. **Categorical Severity Scale:**
   - `Normal` $\to$ `Low anomaly` $\to$ `Medium anomaly` $\to$ `High anomaly`.

---

## Slide 11: Explainable AI & TreeSHAP Attributions (Phase 5)
### **Unmasking Black-Box Models with Game-Theoretic Attributions**

- **TreeSHAP Implementation:**
  - Exact Shapley value calculation in polynomial time:
    $$f(x) = \phi_0 + \sum_{i=1}^M \phi_i(x)$$
- **Physical Domain Findings (NASA C-MAPSS):**
  - **Primary Driver:** `sensor_11` (High-Pressure Compressor outlet static temperature T30) — positive attribution indicates thermodynamic erosion.
  - **Secondary Driver:** `sensor_4` (Low-Pressure Turbine exhaust temperature).
- **Physical Domain Findings (UCI AI4I):**
  - Primary failure triggers: Spikes in `Mechanical power` and excessive `Tool wear` ($> 200\text{ min}$).

---

## Slide 12: LangGraph Multi-Agent Orchestration (Phase 6)
### **Stateful Directed Acyclic Graph Coordination**

```mermaid
flowchart LR
    S[Supervisor] --> P[Prediction] --> A[Anomaly] --> E[Explanation] --> M[Maintenance] --> V[Validation]
```

- **Supervisor Agent:** Validates tensor dimensions, guards against NaNs, manages fail-fast routing.
- **Prediction Agent:** Executes cached ML/DL models; returns RUL or failure probability.
- **Anomaly Agent:** Evaluates autoencoder reconstruction residual.
- **Explanation Agent:** Computes local TreeSHAP attributions.
- **Maintenance Agent:** Formulates evidence-linked maintenance actions.
- **Result Validation Node:** Enforces schema integrity before returning typed JSON.

---

## Slide 13: Prescriptive Recommendation Engine (Phase 7)
### **From Predictions to Actionable Physical Protocols**

- **5-Tier Operational Priority Framework:**
  `Informational` $\to$ `Low priority` $\to$ `Medium priority` $\to$ `High priority` $\to$ `Critical review required`
- **Conflict Resolution Matrix:**
  - *Scenario:* High RUL ($> 100$ cycles) + High Anomaly Severity.
  - *Resolution:* Engine escalates priority to **High priority**, assigns status `conflicting_evidence`, and flags `human_approval_required = True`.
- **Subsystem Mapping:**
  - Maps `sensor_11` $\to$ HPC Borescope Overhaul (FAA Part 145).
  - Maps `Tool wear` $\to$ CNC Spindle & Insert Replacement (ISO 10791 / ISO 3685).
- **SQLite Audit Trail (`data/maintenance_history.db`):**
  - Tracks review lifecycle: `PENDING` $\to$ `APPROVED` | `REJECTED` | `FURTHER_ANALYSIS_REQUESTED`.

---

## Slide 14: Full-Stack Web Dashboard & API (Phase 8)
### **Production-Grade Delivery Architecture**

- **FastAPI REST Backend (Port 8000):**
  - 10 endpoints: `/health`, `/api/models`, `/api/predict/*`, `/api/anomaly/*`, `/api/explain`, `/api/agents/analyze`, `/api/recommendations`, `/api/history`, `/api/recommendations/review`.
  - In-memory singleton cache eliminating model reload latency.
- **Streamlit Web Dashboard (Port 8501):**
  - **Page 1: Overview** (Fleet KPIs, active alerts, recent audits)
  - **Page 2: Machine Monitoring** (Plotly degradation curves, stress radar)
  - **Page 3: Predictive Analysis** (RUL, calibrated failure risk, anomaly gauge)
  - **Page 4: Explainable AI** (TreeSHAP waterfall charts, sensor rankings)
  - **Page 5: Agentic AI Analysis** (Interactive LangGraph execution console)
  - **Page 6: Maintenance Recommendations** (Prescriptive cards, sign-off panel)
  - **Page 7: History & Reports** (SQLite queries, performance tables, report downloads)

---

## Slide 15: NASA C-MAPSS Experimental Results
### **Terminal Benchmark Performance on 100 Test Engines**

```
Model                     RMSE (Cycles)    MAE (Cycles)    NASA Score
---------------------------------------------------------------------
Linear Regression             26.54           19.57         11,181.9
Random Forest                 24.31           17.65         22,754.7
Gradient Boosting             22.50           16.09          3,202.4
XGBoost (Best ML)             22.50           15.35          9,525.5
GRU (w=30)                    22.21           16.43          1,769.1
LSTM (w=30)                   21.98           16.15          1,464.3
LSTM (w=60) (Best DL)         20.02           14.83            889.4
```
- **Key Takeaway:** LSTM ($L=60$) achieves lowest RMSE (20.02) and slashes exponential late penalties by over **90%** compared to XGBoost.

---

## Slide 16: UCI AI4I 2020 Experimental Results
### **Breakdown Classification Under 3.39% Failure Prevalence**

- **Test Confusion Matrix ($N=2,000$ Machine Records):**
  - True Negatives: **1,931** (Specificity: 99.95%)
  - False Positives: **1**
  - False Negatives: **15**
  - True Positives: **53** (Recall: 77.94%)
- **Precision:** $\mathbf{98.15\%}$ | **F1-Score:** $\mathbf{0.8689}$ | **ROC-AUC:** $\mathbf{0.9712}$
- **Failure Mode Recall:**
  - Overstrain Failure (OSF): **100.0%** (20/20)
  - Power Failure (PWF): **100.0%** (18/18)
  - Heat Dissipation Failure (HDF): **95.5%** (21/22)

---

## Slide 17: Complete System Verification & Test Audit
### **100% Automated Test Pass Rate Across 40 Test Cases**

| Test Suite | Focus Area | Tests Executed | Passed | Status |
| :--- | :--- | :---: | :---: | :---: |
| `tests/test_phase6_orchestration.py` | Multi-Agent LangGraph StateGraph | 9 | 9 | **PASS** |
| `src/test_phase7_recommendations.py` | Recommendation Engine & SQLite Review | 10 | 10 | **PASS** |
| `tests/test_phase8_backend.py` | FastAPI REST Service Endpoints | 10 | 10 | **PASS** |
| `tests/test_phase9_e2e_resilience.py` | End-to-End Resilience & Fault Tolerance| 11 | 11 | **PASS** |
| **Total Automated Regression Suite**| **Full Platform Pipeline** | **40** | **40** | **100% PASS** |

---

## Slide 18: Live Demonstration Walkthrough
### **Demonstrating End-to-End Workflow**

1. **Fleet Ingestion:** Select `NASA C-MAPSS` or `UCI AI4I 2020`.
2. **Unit Selection:** Select degraded test unit (e.g. `Engine-FD001-099`).
3. **Telemetry Inspection:** Review 30-cycle multivariate degradation curves on Page 2.
4. **Predictive Inference:** Inspect predicted RUL ($14.2$ cycles) and anomaly score ($0.2105$) on Page 3.
5. **XAI Diagnostic:** Examine TreeSHAP waterfall chart identifying `sensor_11` as the primary driver on Page 4.
6. **Agentic Execution:** Trigger LangGraph multi-agent analysis on Page 5; trace state progression.
7. **Prescriptive Work Order:** Review EASA Part 145 borescope overhaul recommendation on Page 6.
8. **Human Authorization:** Certified engineer enters badge ID and executes `✅ Approve Recommendation`, committing to SQLite.

---

## Slide 19: Non-Autonomous Industrial Safety Guardrails
### **Ensuring Safe Human-Machine Pairing**

- **Advisory Authority Boundary:**
  - The framework provides decision-support advisories; it **never** issues direct mechanical commands.
- **Mandatory Human Sign-Off:**
  - For `Critical review required` and `High priority`, `human_approval_required = True`.
- **Probabilistic Calibration:**
  - Raw uncalibrated probabilities are never displayed as absolute truths.
- **Audit Logging:**
  - Complete immutable record of reviewer ID, timestamp, and notes stored in SQLite.

---

## Slide 20: Academic Contributions & Achievements
### **Key Research Deliverables**

1. **Leakage-Free Preprocessing Standard:** Proved that engine-level partitioning prevents cross-window memorization.
2. **Empirical Asymmetric Loss Advantage:** Demonstrated why recurrent architectures minimize catastrophic late RUL errors.
3. **Calibrated Cost-Sensitive Machine Failure Modeling:** Achieved 98.15% precision and 100% OSF/PWF recall.
4. **Agentic AI Orchestration:** Replaced rigid heuristics with a dynamic LangGraph multi-agent decision support system.
5. **Production Full-Stack Application:** Delivered tested FastAPI and Streamlit applications.

---

## Slide 21: Limitations & Future Research
### **Opportunities for Extension**

- **Current Limitations:**
  - NASA data originates from simulated thermodynamic physics.
  - AI4I records are discrete shifts rather than continuous trajectories.
- **Future Directions:**
  - Temporal Transformers (PatchTST, Informer) for multi-condition fleets (FD002/FD004).
  - Microcontroller quantization (ONNX Runtime, INT8) for edge-spindle deployment.
  - Industrial IoT integration with live MQTT and OPC-UA protocols.

---

## Slide 22: Conclusion & Viva Q&A
### **Summary & Open Floor**

- **Framework Status:** Complete, reproducible, fully documented, and academically defensible.
- **Total Models Trained:** 32 ML Baselines, 18 Deep Learning Architectures, 5 Anomaly Detectors.
- **Test Pass Rate:** 40 / 40 Automated Tests Passed (100%).

*Thank you to the examination committee. I now invite your questions.*
