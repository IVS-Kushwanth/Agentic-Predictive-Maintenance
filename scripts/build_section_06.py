"""
Script to generate:
documentation/06_Architecture_Diagrams/All_Architecture_Diagrams.md
Containing 14 comprehensive Mermaid diagrams reflecting the actual source code.
"""

import os

DOC_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "documentation", "06_Architecture_Diagrams")
os.makedirs(DOC_ROOT, exist_ok=True)

DIAGRAMS_CONTENT = r"""# Comprehensive System Architecture & Workflow Diagrams
## Agentic Predictive Maintenance Framework

This document consolidates 14 technical architecture and workflow diagrams spanning the entire end-to-end framework. All diagrams are formulated using GitHub Flavored Markdown Mermaid notation and reflect the verified production implementation in `src/`.

---

## Diagram 1: Complete Project Architecture (7-Tier CPS)
Describes the macro-level separation of concerns from industrial sensor telemetry to web interface.

```mermaid
graph TB
    subgraph Layer1 ["Tier 1: Telemetry Ingestion & Transformation"]
        D_NASA["NASA C-MAPSS Telemetry (FD001-FD004)"]
        D_AI4I["UCI AI4I 2020 CNC Records"]
        P_NASA["NASA Pipeline: 7 Constant Sensors Dropped, MinMax Scaling, 136 Rolling Features"]
        P_AI4I["AI4I Pipeline: Physical Power Features, Standard Scaling, One-Hot Encoding"]
        D_NASA --> P_NASA
        D_AI4I --> P_AI4I
    end

    subgraph Layer2 ["Tier 2: Model Analytics & Prediction Engine"]
        T_SEQ["Sliding Window Tensors (L=30, L=60)"]
        T_TAB["Engineered Tabular Matrices"]
        P_NASA --> T_SEQ
        P_NASA --> T_TAB
        P_AI4I --> T_TAB

        M_XGB_RUL["XGBoost Regressor (RUL)"]
        M_LSTM_RUL["Bi-LSTM / GRU Regressor (RUL)"]
        M_EW["Early Warning Classifiers (H=10, 20, 30)"]
        M_AI4I_GB["Gradient Boosting Classifier (Failure tau=0.485)"]

        T_TAB --> M_XGB_RUL
        T_TAB --> M_EW
        T_TAB --> M_AI4I_GB
        T_SEQ --> M_LSTM_RUL
    end

    subgraph Layer3 ["Tier 3: Anomaly Detection & Explainability"]
        AD_AE["LSTM Autoencoder (Sequence Reconstruction MSE)"]
        AD_IF["Isolation Forest (Contamination=0.034)"]
        XAI_SHAP["TreeSHAP Attribution Engine"]
        XAI_TEMP["Deep Temporal Saliency Engine"]

        T_SEQ --> AD_AE
        T_TAB --> AD_IF
        M_XGB_RUL & M_AI4I_GB --> XAI_SHAP
        M_LSTM_RUL --> XAI_TEMP
    end

    subgraph Layer4 ["Tier 4: LangGraph Multi-Agent Orchestrator"]
        AG_SUP["Supervisor Agent (State Evaluator & Router)"]
        AG_PRED["Prediction Agent"]
        AG_ANOM["Anomaly Agent"]
        AG_EXPL["Explanation Agent"]
        AG_MAINT["Maintenance Recommendation Agent"]

        AG_SUP --> AG_PRED
        AG_SUP --> AG_ANOM
        AG_SUP --> AG_EXPL
        AG_SUP --> AG_MAINT
    end

    subgraph Layer5 ["Tier 5: Maintenance Recommendation & Priority Engine"]
        REC_ENG["ISO 13381-1 Decision Rules Engine"]
        PRIOR["Priority Tiers: CRITICAL / HIGH / MEDIUM / LOW / INFORMATIONAL"]
        ACT_MAP["Subsystem Work-Order Dispatch (HPC, HPT, Bearings, Tooling)"]
        REC_ENG --> PRIOR --> ACT_MAP
    end

    subgraph Layer6 ["Tier 6: Persistence & Governance"]
        SQL_DB[("SQLite Database: maintenance_history.db")]
        AUDIT["Human Review Audit Trail (PENDING / APPROVED / REJECTED)"]
        ACT_MAP --> SQL_DB --> AUDIT
    end

    subgraph Layer7 ["Tier 7: Application & Presentation Layer"]
        API["FastAPI REST Backend (:8000)"]
        DASH["Streamlit Interactive Dashboard (:8501)"]
        API <--> DASH
        API <--> AG_SUP
        SQL_DB <--> API
    end

    Layer1 --> Layer2
    Layer2 --> Layer3
    Layer3 --> Layer4
    Layer4 --> Layer5
    Layer5 --> Layer6
    Layer6 --> Layer7
```

---

## Diagram 2: Data Acquisition Pipeline
Illustrates the automated raw data retrieval, directory structure initialization, and schema validation.

```mermaid
flowchart TD
    Start([Execute data_download.py]) --> CheckDir{Directory Structure Exists?}
    CheckDir -- No --> CreateDirs[Create data/raw/CMAPSS, data/raw/ai4i, data/processed, models/]
    CheckDir -- Yes --> FetchNASA[Download NASA C-MAPSS Zip Archive]
    CreateDirs --> FetchNASA
    FetchNASA --> UnzipNASA[Extract train_FD001.txt - train_FD004.txt, test_FD001.txt - test_FD004.txt, RUL_FD001.txt - RUL_FD004.txt]
    UnzipNASA --> FetchAI4I[Download UCI AI4I 2020 Predictive Maintenance CSV]
    FetchAI4I --> ValidateSchema[Run data_validation.py: Verify Column Headers, Row Counts, Missing Values]
    ValidateSchema --> DataReady([Raw Data Verified and Staged in data/raw/])
```

---

## Diagram 3: NASA Preprocessing & Feature Engineering Workflow
Shows the elimination of invariant sensors, rolling feature synthesis, and target clipping.

```mermaid
flowchart LR
    RawNASA[Raw C-MAPSS 26 Columns: unit, cycle, os1-3, s1-21] --> DropCols[Drop Constant Sensors: s1, s5, s6, s10, s16, s18, s19]
    DropCols --> CleanSensors[14 Active Sensor Channels + 3 Operational Settings]
    CleanSensors --> RULCalc[Calculate RUL: MaxCycle - CurrentCycle]
    RULCalc --> PiecewiseClip[Piecewise Linear Clip: min RUL, 125 cycles]
    PiecewiseClip --> EWCalc[Synthesize Early Warning Labels: H10, H20, H30 <= H]
    
    CleanSensors --> RollingStats[Compute Rolling Window Features: w=5, 15, 30]
    RollingStats --> RollMean[Rolling Mean: 14 x 3 = 42 features]
    RollingStats --> RollStd[Rolling Std Dev: 14 x 3 = 42 features]
    RollingStats --> LagDiff[Lag Differences: 14 x 3 = 42 features]
    RollingStats --> BaseDev[Baseline Deviation: 14 x 1 = 14 features]
    
    RollMean & RollStd & LagDiff & BaseDev --> FeatureConcat[Concatenate: 17 base + 136 engineered = 153 features]
    FeatureConcat --> TrainSplit[Engine-Level GroupKFold Split: 80 Engines Train, 20 Validation]
    TrainSplit --> Scale[Fit MinMaxScaler on Train Split ONLY]
    Scale --> TabularExport([Saved: data/processed/val_tabular.csv])
```

---

## Diagram 4: NASA Sequence Tensor Generation (Sliding Window)
Illustrates how sliding sequence windows of lengths $L=30$ and $L=60$ are constructed for deep recurrent networks without cross-engine boundary contamination.

```mermaid
flowchart TD
    EngData["Sorted Engine Trajectory (Engine i, t = 1 ... T_i)"] --> FilterEng{"Engine Length T_i >= Window Length L?"}
    FilterEng -- No --> PadOrSkip[Skip Engine or Pad Sequence]
    FilterEng -- Yes --> InitWindow["Initialize Window: [t - L + 1 : t]"]
    InitWindow --> StepWindow["Slide Window Forward by 1 Time Step"]
    StepWindow --> ExtractMatrix["Extract Sensor Sub-matrix (L x 14)"]
    ExtractMatrix --> AlignTarget["Align Target RUL at terminal step t"]
    AlignTarget --> EndCheck{"t == T_i (Final Cycle)?"}
    EndCheck -- No --> StepWindow
    EndCheck -- Yes --> NextEngine{"More Engines in Split?"}
    NextEngine -- Yes --> EngData
    NextEngine -- No --> PackNPZ["Pack into 3D Tensor: (N_samples, L, 14) and 1D Target (N_samples,)"]
    PackNPZ --> SaveNPZ(["Saved: data/processed/sequences_w30_train.npz, sequences_w60_val.npz"])
```

---

## Diagram 5: UCI AI4I 2020 Processing & Failure Classification Workflow
Demonstrates physics-informed feature mechanics engineering, strict target isolation, and threshold-tuned gradient boosting.

```mermaid
flowchart TD
    RawAI4I["AI4I Raw CSV (10,000 instances, 14 columns)"] --> DropMeta["Drop Identifiers: UDI, Product ID"]
    DropMeta --> CatEncode["One-Hot / Target Encode 'Type' (L, M, H)"]
    
    CatEncode --> PhysEng["Physics-Informed Feature Engineering"]
    PhysEng --> P1["Power (W) = Torque (Nm) x Rotational Speed (rad/s)"]
    PhysEng --> P2["Temp Diff (K) = Process Temp - Air Temp"]
    PhysEng --> P3["Tool Wear Strain = Tool Wear (min) x Torque (Nm)"]
    
    P1 & P2 & P3 --> FormMatrix["Engineered Feature Matrix (14 Features)"]
    
    RawAI4I --> IsolateTargets["Target Isolation Engine"]
    IsolateTargets --> Y_Fail["Primary Target: 'Machine failure' (3.4% positive)"]
    IsolateTargets --> Y_Modes["Sequester Failure Modes: TWF, HDF, PWF, OSF, RNF (Analysis Only)"]
    
    FormMatrix --> StratSplit["Stratified 80/20 Train/Test Split (8,000 train, 2,000 test)"]
    StratSplit --> Scale["StandardScaler fit on Train ONLY"]
    Scale --> TrainGB["Train Gradient Boosting / XGBoost Classifier"]
    TrainGB --> ThreshTune["Threshold Optimization on PR-Curve (Optimal tau* = 0.485)"]
    ThreshTune --> EvalResults["Precision: 98.15%, Recall: 77.94%, F1: 0.8689, PR-AUC: 0.8679"]
    EvalResults --> PostHocDiag["Correlate False Negatives with Sequestered Failure Modes"]
```

---

## Diagram 6: Traditional ML Model Architecture & Baseline Comparison
Visualizes the parallel training and cross-validation pipeline for linear, tree-ensemble, and boosting models.

```mermaid
flowchart TD
    InputFeatures["Processed Tabular Features (153 NASA / 14 AI4I)"] --> CV["5-Fold Cross Validation / GroupKFold"]
    
    CV --> M1["Linear / Ridge / Logistic Regression"]
    CV --> M2["Random Forest (n_estimators=200, max_depth=12)"]
    CV --> M3["Gradient Boosting (learning_rate=0.05, n_estimators=150)"]
    CV --> M4["XGBoost Regressor / Classifier (colsample_bytree=0.8, subsample=0.8)"]
    
    M1 --> Res1["NASA RMSE: 24.81 | AI4I F1: 0.6210"]
    M2 --> Res2["NASA RMSE: 22.95 | AI4I F1: 0.8420"]
    M3 --> Res3["NASA RMSE: 22.70 | AI4I F1: 0.8689 (Best AI4I)"]
    M4 --> Res4["NASA RMSE: 22.50 (Best ML) | AI4I F1: 0.8610"]
    
    Res1 & Res2 & Res3 & Res4 --> ModelSelection["Model Registry Persistence (models/baseline/)"]
```

---

## Diagram 7: Deep Temporal Architecture (Bidirectional LSTM & GRU)
Details the deep neural network layer configurations, dropout regularization, and dense regression head.

```mermaid
graph TB
    subgraph InputTensor ["Input Sequence Tensor"]
        IN["Input: (Batch Size, Sequence Length L=60, 14 Sensor Features)"]
    end

    subgraph RecurrentBackbone ["Recurrent Sequence Backbone (Bi-LSTM / GRU)"]
        L1["Bi-LSTM Layer 1: 64 hidden units per direction (Total = 128)"]
        DO1["Spatial Dropout (p = 0.20)"]
        L2["Bi-LSTM Layer 2: 32 hidden units per direction (Total = 64)"]
        DO2["Spatial Dropout (p = 0.20)"]
        
        IN --> L1 --> DO1 --> L2 --> DO2
    end

    subgraph TemporalPooling ["Temporal Representation Pooling"]
        LAST_H["Extract Final Time-Step Hidden State h_T (64-dim)"]
        DO2 --> LAST_H
    end

    subgraph DenseHead ["Fully Connected Prediction Head"]
        FC1["Dense Layer: 64 -> 32 units (ReLU Activation)"]
        DO3["Dense Dropout (p = 0.10)"]
        FC2["Dense Layer: 32 -> 16 units (ReLU Activation)"]
        OUT["Linear Output Unit: Scalar Predicted RUL (cycles)"]
        
        LAST_H --> FC1 --> DO3 --> FC2 --> OUT
    end

    subgraph Optimization ["Training & Loss"]
        LOSS["Huber Loss / Smooth L1 Loss"]
        OPT["AdamW Optimizer (lr = 1e-3, weight_decay = 1e-4)"]
        SCHED["ReduceLROnPlateau Scheduler (factor = 0.5, patience = 5)"]
        OUT --> LOSS
        OPT -.-> RecurrentBackbone & DenseHead
    end
```

---

## Diagram 8: Unsupervised Anomaly Detection Workflow
Contrasts the continuous reconstruction error mechanism of the LSTM Autoencoder with the partition depth scoring of the Isolation Forest.

```mermaid
flowchart TD
    subgraph NASA_Stream ["NASA Sequence Telemetry"]
        SEQ["Nominal Early-Life Sequences (Cycles 1 to 30)"] --> FIT_AE["Fit LSTM Autoencoder on Nominal States"]
        TEST_SEQ["Unseen Sequence Window (t - 29 : t)"] --> AE_INF["Autoencoder Encode-Decode Pass"]
        AE_INF --> RECON_ERR["Reconstruction MSE = 1/L sum(x_t - x^_t)^2"]
        RECON_ERR --> AE_THRESH{"MSE > Threshold tau=0.0370?"}
        AE_THRESH -- No --> ANO_NORM["Tier: NOMINAL (Score < 0.037)"]
        AE_THRESH -- Yes --> ANO_HIGH["Tier: HIGH / CRITICAL Degraded State"]
    end

    subgraph AI4I_Stream ["AI4I Tabular Telemetry"]
        TAB["Nominal Operating Points"] --> FIT_IF["Fit Isolation Forest (n_estimators=150)"]
        TEST_TAB["Test Telemetry Vector x"] --> IF_INF["Calculate Average Path Length h(x) across Trees"]
        IF_INF --> IF_SCORE["Anomaly Score s(x, n) = 2^(-E[h(x)] / c(n))"]
        IF_SCORE --> IF_TIER{"Severity Tier Mapping"}
        IF_TIER --> T1["s < 0.50 : NOMINAL"]
        IF_TIER --> T2["0.50 <= s < 0.65 : LOW"]
        IF_TIER --> T3["0.65 <= s < 0.80 : MEDIUM"]
        IF_TIER --> T4["s >= 0.80 : HIGH"]
    end
```

---

## Diagram 9: Explainable AI & Feature Attribution Workflow
Details the TreeSHAP game-theoretic decomposition and deep temporal sequence gradients.

```mermaid
flowchart TD
    PredictiveOutput["Model Output: Predicted RUL or P(Failure)"] --> SelectXAI{"Model Architecture Type"}
    
    SelectXAI -- Tree Ensemble (XGBoost/RF) --> TreeSHAP["TreeSHAP Explainer (Conditional Expectation)"]
    TreeSHAP --> ComputePhi["Compute Exact Shapley Attributions phi_i for Each Feature"]
    ComputePhi --> EffCheck{"Additive Efficiency: sum(phi_i) == f(x) - E[f(x)]"}
    EffCheck -- True --> GenWaterfall["Render SHAP Waterfall Plot (Top Contributors)"]
    
    SelectXAI -- Deep Recurrent (LSTM) --> TemporalGrad["Deep Temporal Gradient / Integrated Gradients"]
    TemporalGrad --> SaliencyMatrix["Compute Saliency Matrix S in R^(L x F)"]
    SaliencyMatrix --> TimeDecay["Evaluate Time-Step Degradation Sensitivity Profile"]
    
    GenWaterfall & TimeDecay --> SubsystemMap["Map Top Attributed Sensors to Physical Components (HPC, HPT, LPT, Bearing)"]
    SubsystemMap --> XAI_JSON(["Structured Explanation Output to LangGraph State"])
```

---

## Diagram 10: LangGraph Multi-Agent Orchestration State Graph
Illustrates the cyclical state graph, conditional edges, error trapping, and state payload persistence.

```mermaid
stateDiagram-v2
    [*] --> Supervisor : Ingest Telemetry Payload
    
    state Supervisor {
        [*] --> ValidatePayload
        ValidatePayload --> DetermineRoute : Dataset Aware (NASA vs AI4I)
    }

    Supervisor --> PredictionAgent : Dispatch to Ingestion Node
    
    state PredictionAgent {
        [*] --> ExecuteInference
        ExecuteInference --> FormatPredictions : RUL, Failure Prob, Risk Tier
    }
    
    PredictionAgent --> AnomalyAgent : Update State
    
    state AnomalyAgent {
        [*] --> ScoreTelemetry
        ScoreTelemetry --> ClassifySeverity : NOMINAL / LOW / MEDIUM / HIGH
    }
    
    AnomalyAgent --> ExplanationAgent : Update State
    
    state ExplanationAgent {
        [*] --> RunTreeSHAP
        RunTreeSHAP --> ExtractTopSensors : Rank Attributions
    }
    
    ExplanationAgent --> MaintenanceAgent : Update State
    
    state MaintenanceAgent {
        [*] --> EvaluateISO13381
        EvaluateISO13381 --> SynthesizeWorkOrder : Assign Action & Priority
        SynthesizeWorkOrder --> AppendDisclaimer : Enforce Safety Protocol
    }
    
    MaintenanceAgent --> ResultValidation : Validate Complete State
    
    state ResultValidation {
        [*] --> CheckCompleteness
        CheckCompleteness --> PersistSQLite : Write to maintenance_history.db
    }
    
    ResultValidation --> [*] : Return Final Diagnostic Response
```

---

## Diagram 11: Maintenance Decision-Support & Action Engine Workflow
Flowchart of the priority assignment rules based on RUL, anomaly score, and failure probability.

```mermaid
flowchart TD
    InState["Inputs: RUL Estimate, Failure Probability, Anomaly Tier, Top Sensors"] --> CheckP1{"RUL <= 20 cycles OR Failure Prob >= 0.75 OR Anomaly Tier == HIGH?"}
    
    CheckP1 -- Yes --> SetCritical["Priority: CRITICAL"]
    SetCritical --> ActCrit["Action: Immediate Asset Stand-down & Borescope / Spindle Overhaul"]
    SetCritical --> SignCrit["Mandatory Sign-off: Chief Reliability Engineer"]
    
    CheckP1 -- No --> CheckP2{"RUL <= 50 cycles OR Failure Prob >= 0.40 OR Anomaly Tier == MEDIUM?"}
    CheckP2 -- Yes --> SetHigh["Priority: HIGH"]
    ActCrit & SetHigh --> ActHigh["Action: Schedule Maintenance within 48 Hours"]
    SetHigh --> SignHigh["Mandatory Sign-off: Shift Maintenance Supervisor"]
    
    CheckP2 -- No --> CheckP3{"RUL <= 80 cycles OR Failure Prob >= 0.20 OR Anomaly Tier == LOW?"}
    CheckP3 -- Yes --> SetMed["Priority: MEDIUM"]
    SetMed --> ActMed["Action: Inspect Sensor Calibration & Clean Fluid Filters at Next Shift"]
    
    CheckP3 -- No --> SetLow["Priority: LOW / INFORMATIONAL"]
    SetLow --> ActLow["Action: Normal Operation; Continue Standard Telemetry Logging"]
    
    ActHigh & ActMed & ActLow --> FormatSummary["Format ISO 13381-1 Maintenance Advisory Card"]
    FormatSummary --> SQLiteInsert["Persist to SQLite with Review Status = 'PENDING'"]
```

---

## Diagram 12: Backend FastAPI Architecture
Illustrates the router layout, middleware, dependency injection, and Pydantic validation boundaries.

```mermaid
graph TB
    subgraph ClientRequests ["HTTP Client Requests"]
        REQ1["GET /health"]
        REQ2["GET /api/models"]
        REQ3["POST /api/predict/rul"]
        REQ4["POST /api/predict/failure"]
        REQ5["POST /api/anomaly/detect"]
        REQ6["POST /api/explain"]
        REQ7["POST /api/agents/analyze"]
        REQ8["POST /api/recommendations/review"]
    end

    subgraph FastAPICore ["FastAPI Application (app.py)"]
        CORS["CORSMiddleware (Allow All / Specified Origins)"]
        EXCEPTION["Global Exception Handler (HTTP 422, 400, 500)"]
        ROUTERS["API Routers: Predict, Anomaly, Explain, Agents, Review"]
        
        CORS --> EXCEPTION --> ROUTERS
    end

    subgraph ServiceLayer ["Business Logic & Services"]
        SRV_MOD["ModelService (models/registry.py)"]
        SRV_ANO["AnomalyService (anomaly/)"]
        SRV_EXP["ExplanationService (explainability/)"]
        SRV_WF["WorkflowOrchestrator (graph/workflow.py)"]
        SRV_DB["DatabaseManager (data/maintenance_history.db)"]
    end

    ClientRequests --> FastAPICore
    ROUTERS --> ServiceLayer
```

---

## Diagram 13: Frontend Streamlit Dashboard Layout
Shows the navigational structure and page-by-page functional capabilities of the web dashboard.

```mermaid
graph LR
    NAV["Sidebar Navigation Menu"] --> P1["Page 1: Fleet Monitoring Overview"]
    NAV --> P2["Page 2: NASA Turbofan RUL Estimator"]
    NAV --> P3["Page 3: CNC Machine Failure Predictor"]
    NAV --> P4["Page 4: Multivariate Anomaly Detection"]
    NAV --> P5["Page 5: Explainable AI Diagnostics (SHAP)"]
    NAV --> P6["Page 6: Multi-Agent Orchestrator Studio"]
    NAV --> P7["Page 7: Maintenance Work Orders & History"]

    P1 --> V1["Global KPIs, Active Alerts, Asset Status Gauges"]
    P2 --> V2["Interactive Trajectory Plot, RUL Dial, Multi-Horizon Alert"]
    P3 --> V3["Operating Parameter Sliders, Failure Risk Gauge, Breakdown Modes"]
    P4 --> V4["Reconstruction Error Chart, Severity Radar, Anomaly Trends"]
    P5 --> V5["Interactive SHAP Waterfall, Sensor Importance Ranking, Time Profiles"]
    P6 --> V6["LangGraph Execution Visualizer, Agent State Inspection"]
    P7 --> V7["SQLite History Table, Approve/Reject Actions, CSV Export"]
```

---

## Diagram 14: Complete System Execution Workflow
End-to-end lifecycle from raw sensor streaming to human sign-off in the physical plant.

```mermaid
flowchart TD
    Sensor["Machine Sensor Telemetry (Turbofan / CNC Mill)"] --> Ingest["Ingestion & Cleaning Pipeline"]
    Ingest --> Scalers["Apply Scalers & Rolling Windows"]
    Scalers --> APICall["POST /api/agents/analyze"]
    
    APICall --> SupervisorAgent["LangGraph Supervisor Orchestration"]
    SupervisorAgent --> ParallelAnalysis["Run Analytics: Predictions + Anomalies + SHAP Attributions"]
    ParallelAnalysis --> SynthesizeRec["Maintenance Agent: ISO 13381-1 Decision Engine"]
    
    SynthesizeRec --> StoreDB["Store Work Order in SQLite (Status: PENDING)"]
    StoreDB --> SendDashboard["Render Diagnostic Visuals on Streamlit Dashboard"]
    
    SendDashboard --> HumanReview{"Plant Reliability Engineer Action"}
    HumanReview -- Reject --> RecReject["POST /api/recommendations/review (status='REJECTED')"]
    HumanReview -- Request Analysis --> RecReq["POST /api/recommendations/review (status='FURTHER_ANALYSIS')"]
    HumanReview -- Approve --> RecApprove["POST /api/recommendations/review (status='APPROVED')"]
    
    RecApprove --> DispatchWorkOrder["Dispatch Physical Work Order to ERP / SCADA Maintenance Crew"]
    RecReject & RecReq & DispatchWorkOrder --> AuditClose(["Closed-Loop Audit Trail Logged"])
```
"""

def generate_section_06():
    fpath = os.path.join(DOC_ROOT, "All_Architecture_Diagrams.md")
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(DIAGRAMS_CONTENT.strip() + "\n")
    print(f"Generated: {fpath} ({len(DIAGRAMS_CONTENT)} bytes)")

if __name__ == "__main__":
    generate_section_06()
