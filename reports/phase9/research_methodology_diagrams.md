# Phase 9: Research Methodology & Architecture Diagrams
## Comprehensive Mermaid Visualizations for Project Thesis and Technical Presentations

---

### Diagram 1: Overall End-to-End System Architecture

```mermaid
flowchart TD
    subgraph DataLayer ["Data Ingestion & Preprocessing Layer"]
        D1["NASA C-MAPSS Telemetry (FD001–FD004)"] --> P1["Engine-Level Partitioning & Rolling Feat. Eng."]
        D2["UCI AI4I 2020 CNC Telemetry"] --> P2["Physical Stress Engineering & Normalization"]
        P1 --> A1["3D Sequence NPZ Archives (N, L, 153)"]
        P2 --> A2["Standardized Tabular Splits (N, 14)"]
    end

    subgraph ModelLayer ["Predictive & Anomaly Modeling Layer"]
        A1 --> M1["XGBoost / Random Forest RUL Regressors"]
        A1 --> M2["PyTorch Bidirectional LSTM / GRU Networks"]
        A1 --> M3["LSTM Autoencoder Unsupervised Detector"]
        A2 --> M4["Gradient Boosting / XGBoost Failure Classifiers"]
        A2 --> M5["Isolation Forest Operating Outlier Detector"]
    end

    subgraph ExplainLayer ["Explainable AI (XAI) Layer"]
        M1 & M4 --> E1["TreeSHAP Shapley Attribution Engine"]
        M2 --> E2["Captum Integrated Gradients Engine"]
        E1 & E2 --> E3["Ranked Sensor Drivers & Plain-Text Summaries"]
    end

    subgraph AgentLayer ["LangGraph Multi-Agent Orchestration Layer"]
        E3 --> AG1["Supervisor Agent (State & Validation)"]
        AG1 --> AG2["Prediction Agent"]
        AG2 --> AG3["Anomaly Detection Agent"]
        AG3 --> AG4["Explanation Agent"]
        AG4 --> AG5["Maintenance Recommendation Agent"]
    end

    subgraph PrescriptiveLayer ["Prescriptive Recommendation & Storage"]
        AG5 --> R1["5-Tier Priority & Conflict Resolution Engine"]
        R1 --> R2["Subsystem Protocol Mapper (FAA / ISO Standards)"]
        R2 --> DB[("SQLite Database: data/maintenance_history.db")]
    end

    subgraph UILayer ["Delivery & Presentation Layer"]
        DB --> API["FastAPI REST Service (Port 8000)"]
        API --> UI["Streamlit Interactive Web Dashboard (Port 8501)"]
    end
```

---

### Diagram 2: Data Preprocessing & Leakage-Free Feature Engineering

```mermaid
flowchart LR
    subgraph RawData ["1. Ingestion"]
        R1["Raw NASA C-MAPSS Text Files"]
        R2["Raw AI4I CSV File"]
    end

    subgraph Cleaning ["2. Cleaning & Dropping"]
        R1 --> C1["Drop 7 Constant Sensors (Zero Variance)"]
        R2 --> C2["Extract Physical Features; Exclude Target Failure Modes"]
    end

    subgraph Partitioning ["3. Engine-Level Partitioning"]
        C1 --> S1["Partition by Engine ID: Train (80) / Val (20) / Test (100)"]
        C2 --> S2["Stratified Split (80:20) Preserving Failure Prevalence (3.39%)"]
    end

    subgraph RollingEng ["4. Backward Rolling Engineering"]
        S1 --> FE1["Compute Rolling Mean, Std, Delta over w in {5, 10, 20} (min_periods=1)"]
        S2 --> FE2["Formulate Power, Torque-Speed Ratio, Heat Dissipation Index"]
    end

    subgraph Scaling ["5. Isolated Scaler Fitting"]
        FE1 --> SC1["Fit StandardScaler strictly on Training Engines; Save via Joblib"]
        FE2 --> SC2["Fit Preprocessor strictly on Training Split; Save via Joblib"]
    end
```

---

### Diagram 3: NASA Temporal 3D Sequence Construction (No Engine Boundary Overlap)

```mermaid
flowchart TD
    subgraph Engine1 ["Engine Unit 1 Trajectory (Cycles 1 to 192)"]
        E1_W1["Window 1: Cycles 1 to 30 -> Target RUL_30"]
        E1_W2["Window 2: Cycles 2 to 31 -> Target RUL_31"]
        E1_WN["Window K: Cycles 163 to 192 -> Target RUL_192"]
    end

    subgraph Boundary ["STRICT BOUNDARY WALL: Sliding Windows Never Cross Engines"]
        BW["NO OVERLAPPING SLICES PERMITTED"]
    end

    subgraph Engine2 ["Engine Unit 2 Trajectory (Cycles 1 to 287)"]
        E2_W1["Window 1: Cycles 1 to 30 -> Target RUL_30"]
        E2_W2["Window 2: Cycles 2 to 31 -> Target RUL_31"]
    end

    Engine1 -.-> Boundary
    Boundary -.-> Engine2
    Engine1 & Engine2 --> T3D["Concatenated 3D Array: (Total Windows, 30, 153)"]
```

---

### Diagram 4: Deep Learning Time-Series Model Architectures

```mermaid
flowchart LR
    subgraph LSTM_Arch ["Bidirectional LSTM Regressor (L=60)"]
        IN1["Input Window (60, 153)"] --> L1["Bidirectional LSTM Layer 1 (Hidden=64)"]
        L1 --> D1["Dropout (p=0.20)"]
        D1 --> L2["Bidirectional LSTM Layer 2 (Hidden=32)"]
        L2 --> POOL1["Terminal Hidden State h_T"]
        POOL1 --> FC1["Dense Layer (64 -> 32 -> ReLU)"]
        FC1 --> OUT1["Predicted RUL (Scalar)"]
    end

    subgraph CNN_LSTM_Arch ["Hybrid CNN-LSTM Regressor"]
        IN2["Input Window (30, 153)"] --> CONV1["1D Conv (Filters=32, Kernel=3) + ReLU"]
        CONV1 --> MAXP["MaxPool1d (Pool=2)"]
        MAXP --> LSTMC["LSTM Layer (Hidden=64)"]
        LSTMC --> FC2["Dense Linear Head"]
        FC2 --> OUT2["Predicted RUL (Scalar)"]
    end
```

---

### Diagram 5: Unsupervised Anomaly Detection Pipeline

```mermaid
flowchart TD
    IN["Nominal Training Telemetry (Early Life Cycles / Healthy Shifts)"]
    
    subgraph AutoencoderPath ["Path A: Deep Latent Reconstruction"]
        IN --> ENC["LSTM / Dense Encoder (Bottleneck Compression)"]
        ENC --> LAT["Latent Representation z"]
        LAT --> DEC["LSTM / Dense Decoder (Reconstruction)"]
        DEC --> REC["Reconstructed Telemetry x_hat"]
        REC --> LOSS["Reconstruction Residual: e = ||x - x_hat||^2"]
        LOSS --> CALIB1["Threshold Calibration: tau = 97.5th Percentile = 0.0370"]
    end

    subgraph IsolationPath ["Path B: Tree Isolation"]
        IN --> IF["Isolation Forest (100 Decision Trees)"]
        IF --> PATH["Average Path Length E(h(x))"]
        PATH --> SCORE["Anomaly Score s(x, n) in [-1, 1]"]
        SCORE --> CALIB2["Contamination Calibration: tau = -0.050"]
    end

    CALIB1 & CALIB2 --> SEV["Categorical Severity: Normal | Low | Medium | High Anomaly"]
```

---

### Diagram 6: Explainable AI (TreeSHAP Attribution) Workflow

```mermaid
flowchart TD
    SAMPLE["Target Machine Feature Vector x"] --> EXP["TreeSHAP Explainer (Model + Background)"]
    EXP --> SHAP_VALS["Compute Exact Shapley Attributions phi_i(x)"]
    
    subgraph Decomp ["Additive Feature Attribution"]
        SHAP_VALS --> SUM["f(x) = phi_0 + sum(phi_i)"]
        SUM --> POS["phi_i > 0: Increases Failure Probability / Shortens RUL"]
        SUM --> NEG["phi_i < 0: Nominal Protective Factor / Extends RUL"]
    end

    POS & NEG --> RANK["Rank Features by Absolute Magnitude |phi_i|"]
    RANK --> SUB["Subsystem Correlator (Map sensor_11 -> HPC, Tool wear -> Spindle)"]
    SUB --> VIZ["Generate Waterfall Chart & Plain-Text Engineering Brief"]
```

---

### Diagram 7: LangGraph Multi-Agent StateGraph Execution

```mermaid
stateDiagram-v2
    [*] --> Supervisor
    Supervisor --> PredictionAgent: Input Validated (No NaNs, Correct Dims)
    Supervisor --> ResultValidation: Input Invalid (Fail Fast)
    
    PredictionAgent --> AnomalyAgent: Predictions Computed (RUL / Prob)
    AnomalyAgent --> ExplanationAgent: Anomaly Severity Evaluated
    ExplanationAgent --> MaintenanceAgent: TreeSHAP Attributions Generated
    
    MaintenanceAgent --> ResultValidation: Actions Formulated & Saved to SQLite
    ResultValidation --> [*]: Return Typed Response JSON
```

---

### Diagram 8: Prescriptive Maintenance Priority & Conflict Resolution

```mermaid
flowchart TD
    RUL["Supervised Predicted RUL (NASA)"] 
    PROB["Breakdown Probability (AI4I)"]
    ANOM["Anomaly Severity (Autoencoder)"]

    RUL & PROB & ANOM --> DECIDE{"Priority Decision Engine"}

    DECIDE -->|RUL <= 20 or Risk >= 75%| CRIT["Critical Review Required (Immediate Action)"]
    DECIDE -->|RUL > 100 AND High Anomaly| CONF["High Priority: Conflicting Evidence Flagged"]
    DECIDE -->|RUL 20-50 or Risk 40-75%| HIGH["High Priority: Accelerated Degradation"]
    DECIDE -->|RUL 50-100 or Medium Anomaly| MED["Medium Priority: Planned Line Check"]
    DECIDE -->|RUL > 100 AND Normal Anomaly| INFO["Informational: Nominal Schedule"]

    CRIT & CONF & HIGH & MED --> HIL{"Human-in-the-Loop Sign-Off"}
    HIL -->|Approve| APP["Status: APPROVED (Commit to DB)"]
    HIL -->|Reject| REJ["Status: REJECTED (Commit to DB)"]
    HIL -->|Further Analysis| FA["Status: FURTHER_ANALYSIS_REQUESTED"]
```

---

### Diagram 9: Complete Full-Stack Web Application Architecture

```mermaid
flowchart TD
    subgraph Client ["Client Browser Layer"]
        U1["Overview Page"]
        U2["Machine Monitoring Page"]
        U3["Predictive Analysis Page"]
        U4["Explainable AI Page"]
        U5["Agentic AI Analysis Page"]
        U6["Maintenance Recommendations Page"]
        U7["History & Reports Page"]
    end

    subgraph Frontend ["Streamlit Presentation Layer (Port 8501)"]
        APP["app/frontend/main.py"]
        COMP["Components: Header, Sidebar, Plotly Charts, Data Loader"]
        U1 & U2 & U3 & U4 & U5 & U6 & U7 --> APP
        APP --> COMP
    end

    subgraph Backend ["FastAPI REST Service Layer (Port 8000)"]
        ROUTER["API Router: /api/predict, /api/anomaly, /api/explain, /api/agents"]
        SERVICES["Services: ModelService, AnomalyService, ExplainService, AgentService"]
        CACHE["Singleton Model Cache (Joblib & PyTorch Models in Memory)"]
        ROUTER --> SERVICES --> CACHE
    end

    subgraph Storage ["Persistence Layer"]
        DB[("SQLite Database: data/maintenance_history.db")]
        LOGS["Experiment Logs & Model Checkpoints"]
    end

    COMP -->|HTTP REST Requests| ROUTER
    SERVICES --> DB
    SERVICES --> LOGS
```
