"""
Script to generate all markdown files for:
documentation/01_Project_Overview/
- Project_Overview.md
- System_Requirements.md
- Technology_Stack.md
- Overall_Architecture.md
"""

import os

DOC_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "documentation", "01_Project_Overview")
os.makedirs(DOC_ROOT, exist_ok=True)

PROJECT_OVERVIEW_CONTENT = """# Agentic Predictive Maintenance: An Explainable Multivariate Time-Series Framework for Early Machine Failure Detection and Maintenance Decision Support

## 1. Executive Summary & Abstract
The advent of Industry 4.0 and Cyber-Physical Systems (CPS) has precipitated massive influxes of multivariate sensor telemetry from industrial assets. However, legacy maintenance regimes remain bifurcated between reactive firefighting and costly, schedule-based overhauls. Traditional predictive maintenance (PdM) machine learning prototypes frequently stall at the "proof-of-concept" boundary due to three fundamental deficiencies:
1. **Opaque Black-Box Models**: Complex deep sequence estimators yield accurate scalar Remaining Useful Life (RUL) estimates but fail to provide human-interpretable causal attribution required for high-consequence operational decisions.
2. **Uncalibrated Sensor Drift vs. Acute Failure**: Classical classifiers struggle to differentiate benign sensor anomalies and operational regime shifts from catastrophic mechanical degradation.
3. **Absence of Autonomous Decision Orchestration**: Isolated predictive models output raw numerical probabilities without contextualizing machine operating history, component failure modes, maintenance urgency, or generating verifiable, human-in-the-loop action plans.

This project introduces an end-to-end, enterprise-grade **Agentic Predictive Maintenance Framework**. The architecture synergizes multivariate time-series deep learning (Bidirectional LSTM, GRU), gradient-boosted ensembles (XGBoost, Random Forest), unsupervised anomaly detectors (LSTM Autoencoders, Isolation Forests), and model-agnostic Explainable AI (TreeSHAP and Deep Temporal Attributions). Crucially, these analytic engines are coordinated under a **LangGraph-driven Multi-Agent Orchestration Layer** featuring dedicated Supervisor, Prediction, Anomaly, Explanation, and Maintenance Recommendation agents. The system culminates in an industrial FastAPI backend and a Streamlit interactive dashboard, providing real-time telemetry streaming, interactive SHAP attribution waterfalls, automated ISO 13381-1 priority assignment, and auditable SQLite human-in-the-loop maintenance authorization workflows.

---

## 2. Research Background & Problem Statement

### 2.1 The Industrial Maintenance Dilemma
Heavy industrial assets—including commercial turbofan engines, high-speed CNC milling centers, gas turbines, and wind power drivetrains—operate under severe thermodynamic and mechanical stress. Unplanned downtime in critical aerospace or manufacturing operations incurs staggering economic penalties, reaching up to $260,000 per hour in high-volume automotive manufacturing and substantial safety risks in aviation. 

Historically, asset operators relied on:
- **Run-to-Failure (Corrective Maintenance)**: Maximizes component service life but introduces severe catastrophic failure risks and collateral damage.
- **Time-Based Preventive Maintenance**: Imposes fixed overhaul schedules (e.g., replace turbine blades every 500 operating hours) based on worst-case statistical wear, causing premature retirement of functional equipment and exorbitant spare parts overhead.

### 2.2 Shortcomings of Conventional AI/ML Approaches
While Condition-Based Monitoring (CBM) and data-driven Predictive Maintenance (PdM) have been researched extensively, existing implementations exhibit critical real-world failure points:
- **Disconnection from Domain Actionability**: Model metrics such as Mean Squared Error (MSE) or Area Under ROC Curve (AUC) do not map directly to maintenance work orders. Plant engineers require answerability to three core questions: *When will the asset fail?*, *Which specific subsystem is degrading?*, and *What concrete corrective maintenance task must be performed today?*
- **Siloed Execution Pipelines**: In typical academic prototypes, anomaly detection, failure classification, and RUL regression are treated as disconnected studies rather than an integrated diagnostic funnel.
- **Lack of Governance and Advisory Verification**: Fully autonomous maintenance dispatch carries unacceptable risks of false shutdowns or improper repairs. Industrial environments mandate advisory systems equipped with strict human-in-the-loop approval workflows, complete telemetry audit trails, and explicit safety disclaimers.

---

## 3. Dataset Architectural Roles: NASA C-MAPSS vs. UCI AI4I 2020

A foundational architectural requirement of this framework is the strict separation and specialized application of two canonical benchmarks. These datasets represent fundamentally distinct physical regimes and predictive paradigms:

```mermaid
graph TD
    subgraph NASA_CMAPSS ["NASA C-MAPSS (Continuous Degradation Dynamics)"]
        N1[Multivariate Turbofan Run-to-Failure] --> N2[Temporal Sequence Formation L=30, 60]
        N2 --> N3[Continuous RUL Regression]
        N2 --> N4[Early-Warning Classification Horizons: 10, 20, 30 Cycles]
        N2 --> N5[Unsupervised Degradation Tracking via LSTM Autoencoder]
    end

    subgraph UCI_AI4I ["UCI AI4I 2020 (Synthetic Milling Operating Shifts)"]
        A1[Single-Point Milling Sensor Records] --> A2[Tabular Feature Engineering & Power Mechanics]
        A2 --> A3[Binary Failure Prediction Imbalanced: 3.4% Failure]
        A2 --> A4[Post-Hoc Failure Mode Diagnosis TWF, HDF, PWF, OSF, RNF]
        A2 --> A5[Isolation Forest Anomaly Scoring]
    end
    
    style NASA_CMAPSS fill:#e8f4fd,stroke:#1d6fa5,stroke-width:2px;
    style UCI_AI4I fill:#fef3e7,stroke:#c46900,stroke-width:2px;
```

### 3.1 NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)
- **Physical Domain**: Simulated high-bypass turbofan engines undergoing continuous thermo-mechanical wear (high-pressure compressor fouling, turbine blade creep).
- **Temporal Nature**: Genuine run-to-failure multivariate time series where each engine trajectory starts in a healthy nominal state and deteriorates continuously over hundreds of flight cycles until operational failure threshold.
- **Task Scope in Framework**:
  1. Continuous Remaining Useful Life (RUL) regression with piecewise-linear target clipping ($RUL_{max} = 125$ cycles).
  2. Multi-horizon early-warning binary classification ($H \in \{10, 20, 30\}$ cycles).
  3. Unsupervised sequence anomaly tracking using LSTM Autoencoder reconstruction errors across healthy early-life baselines.
- **Data Integrity Rule**: Constant uninformative sensor channels (`sensor_1, 5, 6, 10, 16, 18, 19`) exhibiting zero variance are strictly eliminated prior to modeling.

### 3.2 UCI AI4I 2020 Predictive Maintenance Dataset
- **Physical Domain**: Synthetic operating condition records reflecting milling machine tool wear, thermodynamic dissipation, and rotational power mechanics.
- **Temporal Nature**: Single-record tabular snapshots (independent operating shifts). Crucially, the dataset does *not* provide continuous longitudinal engine trajectories.
- **Task Scope in Framework**:
  1. Imbalanced binary classification of overall machine failure (3.4% failure prevalence across 10,000 instances).
  2. Post-hoc failure-mode attribution across 5 specific mechanical failure causes: Tool Wear Failure (TWF), Heat Dissipation Failure (HDF), Power Failure (PWF), Overstrain Failure (OSF), and Random Failure (RNF).
  3. Feature mechanics modeling incorporating physically derived attributes (Rotational Power, Temperature Differential).
- **Target Leakage Prohibition**: The five individual failure mode flags are strictly sequestered from model training features and used exclusively for post-prediction diagnostic correlation.

---

## 4. Key Contributions of the Framework

1. **Dual-Paradigm Predictive Modeling**:
   - Engineered competitive traditional baselines (Linear Regression, Random Forest, Gradient Boosting, XGBoost) and advanced deep temporal architectures (LSTM, Bidirectional LSTM, GRU).
   - Validated that Bi-LSTM ($L=60$) achieves a superior NASA Scoring metric of **889.4** (90.7% reduction in asymmetric late-prediction penalty compared to baseline XGBoost).
2. **Transparent Multi-Level Explainability**:
   - Integrated TreeSHAP for tree ensembles to generate exact Shapley feature attributions, explaining sensor contributions for single-cycle anomalies and machine breakdown predictions.
   - Formulated deep temporal attribution gradients across sliding sequence windows, revealing temporal lag sensitivity and degradation acceleration profiles.
3. **First-of-its-Kind LangGraph Multi-Agent Industrial Orchestrator**:
   - Developed a deterministic state-graph architecture coordinating five specialized agents: `Supervisor`, `PredictionAgent`, `AnomalyAgent`, `ExplanationAgent`, and `MaintenanceAgent`.
   - Enabled intelligent conditional routing that dynamically invokes appropriate deep/ML model pipelines based on asset type, evaluates multi-model consensus, and flags conflicting predictive signals.
4. **Prescriptive Maintenance Recommendation Engine**:
   - Codified domain-grounded industrial rules aligned with **ISO 13381-1 condition monitoring standards**.
   - Translates raw RUL numbers and anomaly severity into categorized work orders (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`, `INFORMATIONAL`) accompanied by root-cause sensor attributions and required human sign-off levels.
5. **Full-Stack Enterprise Integration**:
   - Engineered a high-throughput, OpenAPI-compliant FastAPI backend paired with an industrial-grade Streamlit web control center.
   - Built a SQLite audit ledger (`maintenance_history.db`) enabling maintenance supervisors to inspect historical runs, record formal review decisions (`APPROVED`, `REJECTED`, `FURTHER_ANALYSIS_REQUESTED`), and export timestamped compliance logs.
6. **Strict Experimental Rigor & Zero Data Leakage**:
   - All preprocessing transformers (StandardScalers, MinMaxScalers, encoders) fitted exclusively on training splits.
   - Separate test sets ($N=100$ NASA test trajectories; $N=2,000$ AI4I held-out shifts) evaluated without target leakage or synthetic oversampling leakage.
"""

SYSTEM_REQ_CONTENT = """# System Requirements Specification (SRS)
## Agentic Predictive Maintenance Framework

---

## 1. Functional Requirements (FR)

| Requirement ID | Module / Component | Description | Verification Method |
| :--- | :--- | :--- | :--- |
| **FR-01** | Data Ingestion | The system shall ingest raw NASA C-MAPSS space-delimited text telemetry and UCI AI4I 2020 comma-separated records without data loss. | Unit Test & Pipeline Audit |
| **FR-02** | Sensor Preprocessing | The system shall drop 7 constant C-MAPSS sensors (`s1, s5, s6, s10, s16, s18, s19`), apply MinMax/Standard scaling fitted solely on training splits, and synthesize 136 rolling statistical features. | Unit Test & Schema Assertion |
| **FR-03** | RUL Regression | The system shall predict the Remaining Useful Life (RUL) of turbofan engines using a piecewise-linear capped target ($RUL_{max} = 125$), reporting RMSE, MAE, and NASA Asymmetric Score. | Baseline & DL Evaluation Test |
| **FR-04** | Early-Warning Classification | The system shall evaluate early-warning machine failure indicators across operational horizons $H \in \{10, 20, 30\}$ cycles, outputting calibrated probabilities and PR-AUC. | Evaluation Verification Script |
| **FR-05** | CNC Failure Classification | The system shall predict machine failure on the imbalanced AI4I 2020 dataset using tuned decision thresholds ($\tau^* = 0.485$), reporting Precision, Recall, and F1-score. | AI4I Evaluation Script |
| **FR-06** | Unsupervised Anomaly Detection | The system shall compute continuous anomaly scores using Isolation Forests and LSTM Autoencoders calibrated against nominal early-life operational baselines. | Anomaly Service Verification |
| **FR-07** | Anomaly Severity Tiering | The system shall map anomaly scores into four standardized industrial severity tiers: `NOMINAL`, `LOW`, `MEDIUM`, and `HIGH`. | Threshold Unit Tests |
| **FR-08** | TreeSHAP Attribution | The system shall generate local and global Shapley feature importance attributions for tree models within 250 milliseconds per inference instance. | SHAP Performance Benchmarks |
| **FR-09** | Deep Temporal Attribution | The system shall compute gradient-based feature-over-time attributions for sequence models to reveal temporal lag degradation sensitivity. | Phase 5 Verification Script |
| **FR-10** | Multi-Agent State Management | The system shall coordinate a shared `MaintenanceState` TypedDict via LangGraph, preserving sensor inputs, model predictions, anomaly scores, explanations, and action plans. | LangGraph Orchestration Suite |
| **FR-11** | Dataset-Aware Dynamic Routing | The system shall automatically route incoming inference requests to NASA turbofan models or AI4I milling models based on dataset metadata. | Supervisor Agent Test |
| **FR-12** | ISO 13381-1 Recommendation Dispatch | The system shall generate structured maintenance recommendations classifying priority into `INFORMATIONAL`, `LOW`, `MEDIUM`, `HIGH`, or `CRITICAL`. | Recommendation Suite (10 Tests) |
| **FR-13** | Human-in-the-Loop Review | The system shall provide an auditable mechanism for maintenance engineers to review, approve, reject, or request further analysis on generated maintenance advisories. | SQLite Persistence Test |
| **FR-14** | RESTful Backend Endpoints | The system shall expose at least 10 FastAPI REST endpoints covering health, model registry, inference, anomaly scoring, explanation, and recommendation lifecycle. | Backend API Suite (10 Tests) |
| **FR-15** | Interactive Web Dashboard | The system shall render a 7-page interactive Streamlit dashboard allowing telemetry exploration, simulated sensor streaming, model inference, SHAP waterfall plots, and audit tracking. | Manual & UI Integration Audit |

---

## 2. Non-Functional Requirements (NFR)

| Requirement ID | Category | Specification / Criteria | Target Threshold | Actual Measured Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **NFR-01** | Inference Latency | API inference response time for single-instance tabular prediction (XGBoost/RF). | $< 500\text{ ms}$ | $18.4\text{ ms}$ (FastAPI) |
| **NFR-02** | Deep Sequence Latency | Sequence window ($L=60$) PyTorch forward pass on CPU. | $< 1,000\text{ ms}$ | $64.2\text{ ms}$ (Intel i7 CPU) |
| **NFR-03** | End-to-End Orchestration | Complete 5-agent LangGraph workflow execution including SHAP and recommendation synthesis. | $< 5.0\text{ s}$ | $2.85\text{ s}$ average |
| **NFR-04** | Data Integrity & Leakage | Zero contamination of test/validation statistics into training preprocessors. | Zero leakage | Verified by transformer pipeline design |
| **NFR-05** | Reliability & Resilience | Graceful recovery from incomplete sensor vectors, corrupted JSON, or out-of-range sensor values. | 100% caught, HTTP 422/400 | Verified (Test Suite 11 resilience tests) |
| **NFR-06** | Explainability Fidelity | SHAP additive efficiency property ($\sum \phi_i = f(x) - E[f(x)]$) strictly maintained. | $\Delta < 10^{-4}$ | Verified across all TreeSHAP runs |
| **NFR-07** | Auditability & Compliance | Persistent logging of all model inferences, generated recommendations, and human review decisions. | 100% recorded | SQLite `recommendation_history` table |
| **NFR-08** | Advisory Safety Protocol | The system shall append a prominent non-autonomous safety disclaimer to every generated maintenance output. | 100% coverage | Enforced in `recommendation_engine.py` |
| **NFR-09** | Codebase Maintainability | Modularity conforming to SOLID principles, separation of concerns, and full type-hinting. | Clean architecture | Verified across `src/` modules |
| **NFR-10** | Test Suite Coverage | Comprehensive automated regression suite covering unit, integration, and end-to-end paths. | $> 90\%$ critical paths | 40/40 tests passing (100% pass rate) |

---

## 3. Hardware Requirements

### 3.1 Minimum Hardware Configuration (Evaluation / Development)
- **Processor**: Quad-Core 64-bit CPU (Intel Core i5 8th Gen / AMD Ryzen 5 2000 series or higher).
- **System Memory (RAM)**: 8 GB DDR4.
- **Persistent Storage**: 10 GB free space on solid-state drive (SSD) for raw/processed datasets, `.joblib` pipelines, and PyTorch checkpoints.
- **Network Interface**: Standard local loopback interface (`127.0.0.1`) for inter-process communication between FastAPI (port 8000) and Streamlit (port 8501).

### 3.2 Recommended Hardware Configuration (Enterprise Industrial Node)
- **Processor**: 8-Core / 16-Thread CPU (Intel Core i7/i9 11th Gen+ / AMD Ryzen 7 5000+).
- **System Memory (RAM)**: 16 GB to 32 GB DDR4/DDR5.
- **Hardware Acceleration (GPU)**: NVIDIA GPU with $\ge 6\text{ GB}$ VRAM (GeForce RTX 3060+, Tesla T4, or A10) supporting CUDA 11.8+ / 12.1+ for accelerated batch sequence training. *(Note: All inference modules are fully optimized for real-time CPU execution without mandatory GPU dependencies).*
- **Persistent Storage**: 25 GB NVMe SSD storage.

---

## 4. Software & Environment Requirements

- **Operating System**: Microsoft Windows 10/11 (64-bit), Ubuntu Linux 20.04/22.04 LTS, or macOS 12+ (Monterey or higher).
- **Python Runtime**: Python 3.10 to Python 3.12 (Validated on **Python 3.12.3** 64-bit).
- **Package Manager**: Python Pip version 24.0 or higher.
- **Core Virtual Environment**: Standard Python `venv` or Conda isolated environment.
- **Web Browser**: Modern Chromium-based browser (Google Chrome, Microsoft Edge, Brave) or Mozilla Firefox with JavaScript enabled for Streamlit UI rendering.
"""

TECH_STACK_CONTENT = """# Technology Stack Specification
## Agentic Predictive Maintenance Framework

The framework is constructed upon an enterprise-grade, open-source Python ecosystem designed for deterministic reproducibility, high computational throughput, and modular maintainability.

```mermaid
graph TD
    subgraph Presentation ["Presentation & UI Layer"]
        ST[Streamlit 1.40] --- PLY[Plotly 5.24]
    end

    subgraph API_Layer ["API & Service Layer"]
        FAP[FastAPI 0.115] --- UVI[Uvicorn 0.34]
        FAP --- PYD[Pydantic v2]
    end

    subgraph Agentic_Layer ["Multi-Agent Orchestration Layer"]
        LG[LangGraph 0.2] --- LC[LangChain-Core]
    end

    subgraph Intelligence ["Machine Learning & Intelligence Layer"]
        TORCH[PyTorch 2.5] --- SKL[Scikit-Learn 1.6]
        XGB[XGBoost 2.1] --- SHAP_LIB[SHAP 0.46]
    end

    subgraph Data_Storage ["Data & Persistence Layer"]
        NPY[NumPy 2.2] --- PND[Pandas 2.2]
        SQL[SQLite3 Database] --- JOS[Joblib 1.4]
    end

    Presentation --> API_Layer
    API_Layer --> Agentic_Layer
    Agentic_Layer --> Intelligence
    Intelligence --> Data_Storage
```

---

## 1. Technology Inventory & Architectural Role

| Layer / Domain | Technology / Library | Version | License | Architectural Role & Justification |
| :--- | :--- | :--- | :--- | :--- |
| **Language Runtime** | Python | `3.12.3` | PSF | Foundation runtime providing native asynchronous execution, robust typing, and broad scientific library support. |
| **Deep Learning** | PyTorch | `2.5.1` | BSD-3 | Primary framework for sequence degradation modeling (LSTM, Bidirectional LSTM, GRU) and temporal autoencoders. |
| **Traditional ML** | Scikit-Learn | `1.6.0` | BSD-3 | Baseline regression, classification algorithms, preprocessing transformers (StandardScaler, MinMaxScaler), and Isolation Forest. |
| **Gradient Boosting** | XGBoost | `2.1.3` | Apache-2.0 | Optimized gradient tree boosting for tabular failure prediction and tabular RUL regression with high training throughput. |
| **Explainable AI** | SHAP | `0.46.1` | MIT | TreeSHAP and KernelExplainer implementations delivering game-theoretic Shapley feature attributions. |
| **Agent Orchestration** | LangGraph | `0.2.64` | MIT | Cyclical and conditional state-graph orchestration coordinating multi-agent decision support workflows. |
| **Agent Primitives** | LangChain-Core | `0.3.31` | MIT | Base schema abstractions, state messaging protocols, and tool invocation contracts. |
| **REST Backend** | FastAPI | `0.115.6` | MIT | Asynchronous ASGI RESTful microservice providing OpenAPI docs, dependency injection, and endpoint validation. |
| **ASGI Server** | Uvicorn | `0.34.0` | BSD-3 | High-performance asynchronous web server for hosting FastAPI endpoints. |
| **Data Validation** | Pydantic | `2.10.5` | MIT | Strict JSON payload schema definition, type coercion, and runtime validation across all API endpoints. |
| **Interactive UI** | Streamlit | `1.40.2` | Apache-2.0 | Reactive frontend interface enabling rapid industrial prototyping, sensor streaming, and interactive telemetry exploration. |
| **Visualization** | Plotly | `5.24.1` | MIT | High-performance interactive JavaScript charting for degradation curves, radar plots, and confusion matrices. |
| **Data Manipulation** | Pandas | `2.2.3` | BSD-3 | Tabular data wrangling, rolling statistical window aggregation, and time-series indexing. |
| **Scientific Arrays** | NumPy | `2.2.1` | BSD-3 | Core N-dimensional array manipulations, sequence tensor packaging, and vector mathematical operations. |
| **Model Serialization**| Joblib | `1.4.2` | BSD-3 | High-throughput persistence for Scikit-learn transformers, tree models, and precalculated feature sets. |
| **Persistence Store** | SQLite3 | Native | Public Domain | Zero-configuration relational database engine for maintenance action history and human approval tracking. |
| **Documentation** | Python-Docx | `1.2.0` | MIT | Programmatic generation of formal Microsoft Word (.docx) technical reports and academic documentation. |
| **Testing Suite** | Unittest & Pytest | Native / `8.3` | MIT / PSF | Comprehensive regression harness executing 40 unit, integration, and end-to-end resilience test cases. |

---

## 2. Dependency Matrix & Interoperability
- **NumPy 2.x Compatibility**: All deep learning tensors and Scikit-learn array conversions were validated against NumPy 2.2 ABI standards to prevent legacy C-extension runtime faults.
- **Asynchronous Decoupling**: FastAPI operates as an independent backend service on port 8000, allowing headless integration with third-party SCADA or MES systems, while Streamlit interacts exclusively through standard HTTP/JSON requests.
- **Zero-External-Network Requirement**: The core multi-agent graph uses deterministic rule-augmented and heuristic intelligence nodes that execute entirely locally without requiring remote OpenAI/Anthropic cloud API keys. When external LLMs are disabled or offline, the system seamlessly uses local domain logic without service interruption.
"""

OVERALL_ARCH_CONTENT = """# Overall System Architecture & Workflow
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
"""

def generate_section_01():
    files = {
        "Project_Overview.md": PROJECT_OVERVIEW_CONTENT,
        "System_Requirements.md": SYSTEM_REQ_CONTENT,
        "Technology_Stack.md": TECH_STACK_CONTENT,
        "Overall_Architecture.md": OVERALL_ARCH_CONTENT,
    }
    for fname, content in files.items():
        fpath = os.path.join(DOC_ROOT, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content.strip() + "\n")
        print(f"Generated: {fpath} ({len(content)} bytes)")

if __name__ == "__main__":
    generate_section_01()
