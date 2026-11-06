# Overall System Architecture & Workflow
## Agentic Predictive Maintenance Framework

---

## 1. High-Level Multi-Tier Industrial CPS Architecture

The framework is architected according to a 7-tier Cyber-Physical System (CPS) pattern. This modular separation guarantees that low-level sensor ingest, mathematical feature transformations, machine learning inference, agentic orchestration, and user interaction remain decoupled and independently maintainable.

```mermaid
graph TD
    subgraph Tier1 ["Tier 1: Telemetry Ingestion & Preprocessing"]
        RAW_NASA[NASA C-MAPSS Raw Text] --> PRE_N[NASA Ingestion & 7-Sensor Drop]
        RAW_AI4I[UCI AI4I 2020 Raw CSV] --> PRE_A[AI4I Feature Engineering & Power Mechanics]
        PRE_N --> FE_N[Rolling Window Feature Synthesizer 136 Features]
        PRE_N --> SEQ_N[Sliding Sequence Tensor Builder L=30, 60]
    end

    subgraph Tier2 ["Tier 2: Model Analytics & Prediction Engine"]
        FE_N --> ML_RUL[XGBoost / RF RUL Regressor]
        SEQ_N --> DL_RUL[Bidirectional LSTM / GRU Sequence Regressor]
        FE_N --> ML_EW[Multi-Horizon Early Warning Classifiers H=10, 20, 30]
        PRE_A --> ML_FAIL[Gradient Boosting / XGBoost Failure Classifier tau=0.485]
        PRE_A --> ML_MODES[Post-Hoc Failure Mode Diagnostics TWF, HDF, PWF, OSF]
    end

    subgraph Tier3 ["Tier 3: Anomaly Detection & Explainability"]
        SEQ_N --> AE_DET[LSTM Sequence Autoencoder Reconstruction Error]
        PRE_A --> IF_DET[Isolation Forest Tabular Anomaly Scorer]
        ML_FAIL & ML_RUL --> SHAP_EXP[TreeSHAP Feature Attribution Engine]
        DL_RUL --> GRAD_EXP[Deep Temporal Attribution Engine]
    end

    subgraph Tier4 ["Tier 4: LangGraph Multi-Agent Orchestrator"]
        SUP[Supervisor Agent] --> PRE_AG[Prediction Agent]
        SUP --> ANO_AG[Anomaly Agent]
        SUP --> EXP_AG[Explanation Agent]
        SUP --> MNT_AG[Maintenance Recommendation Agent]
        PRE_AG --> ML_RUL & DL_RUL & ML_FAIL
        ANO_AG --> AE_DET & IF_DET
        EXP_AG --> SHAP_EXP & GRAD_EXP
        MNT_AG --> DEC_ENG[ISO 13381-1 Priority & Action Engine]
    end

    subgraph Tier5 ["Tier 5: Persistence & Governance"]
        DEC_ENG --> DB[(SQLite: maintenance_history.db)]
        DB --> REV_LOG[Human Review State: PENDING / APPROVED / REJECTED]
    end

    subgraph Tier6 ["Tier 6: Application & Presentation Layer"]
        FAP[FastAPI Backend Port 8000] <--> SUP
        FAP <--> DB
        ST_UI[Streamlit Web Dashboard Port 8501] <--> FAP
        OP_ENG[Plant Reliability Engineer] <--> ST_UI
    end

    style Tier1 fill:#f8f9fa,stroke:#adb5bd,stroke-width:1px;
    style Tier2 fill:#e3f2fd,stroke:#2196f3,stroke-width:2px;
    style Tier3 fill:#fff3e0,stroke:#ff9800,stroke-width:2px;
    style Tier4 fill:#e8f5e9,stroke:#4caf50,stroke-width:2px;
    style Tier5 fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px;
    style Tier6 fill:#fbe9e7,stroke:#ff5722,stroke-width:2px;
```

---

## 2. End-to-End Diagnostic Sequence Workflow

When a predictive maintenance query is executed (via the dashboard or REST API), the system follows a deterministic execution sequence:

```mermaid
sequenceDiagram
    autonumber
    actor Engineer as Maintenance Engineer
    participant UI as Streamlit Dashboard (:8501)
    participant API as FastAPI Backend (:8000)
    participant Supervisor as LangGraph Supervisor
    participant Predictor as Prediction Agent
    participant Anomaly as Anomaly Agent
    participant Explainer as Explanation Agent
    participant RecAgent as Maintenance Agent
    participant DB as SQLite History DB

    Engineer->>UI: Select Asset & Submit Telemetry (e.g. Engine #24, Cycle 130)
    UI->>API: POST /api/agents/analyze (payload: sensor data, dataset type)
    API->>Supervisor: Initialize MaintenanceState & Run Graph
    
    Supervisor->>Predictor: Dispatch Telemetry for Ingestion
    Predictor->>Predictor: Execute ML/DL Pipeline (RUL = 18.2 cycles, Risk = HIGH)
    Predictor-->>Supervisor: Return Predictions & Confidence
    
    Supervisor->>Anomaly: Dispatch Telemetry for Anomaly Scoring
    Anomaly->>Anomaly: Compute Reconstruction Error / Isolation Score (Severity = HIGH)
    Anomaly-->>Supervisor: Return Anomaly Metrics
    
    Supervisor->>Explainer: Dispatch Active Model & Features
    Explainer->>Explainer: Calculate TreeSHAP Attributions (Top: s11 HPC Temp, s4 LPT Temp)
    Explainer-->>Supervisor: Return Feature Attributions
    
    Supervisor->>RecAgent: Synthesize All Evidence into Action Plan
    RecAgent->>RecAgent: Apply ISO 13381-1 Rules -> Priority: CRITICAL
    RecAgent->>DB: Persist Recommendation Record (Status: PENDING)
    RecAgent-->>Supervisor: Finalize MaintenanceState
    
    Supervisor-->>API: Return Unified Diagnostic JSON
    API-->>UI: Deliver Response Payload
    UI-->>Engineer: Render RUL Gauge, Anomaly Radar, SHAP Waterfall & Action Card
    
    Engineer->>UI: Click "Approve Maintenance Work Order"
    UI->>API: POST /api/recommendations/review (id, status="APPROVED", notes="Approved for hangar inspection")
    API->>DB: Update Review State to APPROVED with Timestamp
    API-->>UI: Confirm Status Updated
    UI-->>Engineer: Display Verified Audit Badge
```

---

## 3. Resilience and Fallback Mechanisms
1. **Model Fallback Protocols**: If sequence data ($L=60$) is unavailable (e.g., in early engine cycles $t < 60$), the system automatically routes inference to the tabular XGBoost regressor, preventing sequence truncation crashes.
2. **Missing Feature Graceful Imputation**: Incomplete sensor dictionaries trigger forward-fill or median baseline replacement accompanied by an explicit data quality warning in the state output.
3. **Safety Disclaimers**: In compliance with industrial safety guidelines, all recommendations carry an explicit disclaimer stating that outputs represent non-autonomous decision support and mandate qualified personnel verification.
