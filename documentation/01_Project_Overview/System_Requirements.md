# System Requirements Specification (SRS)
## Agentic Predictive Maintenance Framework

---

## 1. Functional Requirements (FR)

| Requirement ID | Module / Component | Description | Verification Method |
| :--- | :--- | :--- | :--- |
| **FR-01** | Data Ingestion | The system shall ingest raw NASA C-MAPSS space-delimited text telemetry and UCI AI4I 2020 comma-separated records without data loss. | Unit Test & Pipeline Audit |
| **FR-02** | Sensor Preprocessing | The system shall drop 7 constant C-MAPSS sensors (`s1, s5, s6, s10, s16, s18, s19`), apply MinMax/Standard scaling fitted solely on training splits, and synthesize 136 rolling statistical features. | Unit Test & Schema Assertion |
| **FR-03** | RUL Regression | The system shall predict the Remaining Useful Life (RUL) of turbofan engines using a piecewise-linear capped target ($RUL_{max} = 125$), reporting RMSE, MAE, and NASA Asymmetric Score. | Baseline & DL Evaluation Test |
| **FR-04** | Early-Warning Classification | The system shall evaluate early-warning machine failure indicators across operational horizons $H \in \{10, 20, 30\}$ cycles, outputting calibrated probabilities and PR-AUC. | Evaluation Verification Script |
| **FR-05** | CNC Failure Classification | The system shall predict machine failure on the imbalanced AI4I 2020 dataset using tuned decision thresholds ($	au^* = 0.485$), reporting Precision, Recall, and F1-score. | AI4I Evaluation Script |
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
| **NFR-01** | Inference Latency | API inference response time for single-instance tabular prediction (XGBoost/RF). | $< 500	ext{ ms}$ | $18.4	ext{ ms}$ (FastAPI) |
| **NFR-02** | Deep Sequence Latency | Sequence window ($L=60$) PyTorch forward pass on CPU. | $< 1,000	ext{ ms}$ | $64.2	ext{ ms}$ (Intel i7 CPU) |
| **NFR-03** | End-to-End Orchestration | Complete 5-agent LangGraph workflow execution including SHAP and recommendation synthesis. | $< 5.0	ext{ s}$ | $2.85	ext{ s}$ average |
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
- **Hardware Acceleration (GPU)**: NVIDIA GPU with $\ge 6	ext{ GB}$ VRAM (GeForce RTX 3060+, Tesla T4, or A10) supporting CUDA 11.8+ / 12.1+ for accelerated batch sequence training. *(Note: All inference modules are fully optimized for real-time CPU execution without mandatory GPU dependencies).*
- **Persistent Storage**: 25 GB NVMe SSD storage.

---

## 4. Software & Environment Requirements

- **Operating System**: Microsoft Windows 10/11 (64-bit), Ubuntu Linux 20.04/22.04 LTS, or macOS 12+ (Monterey or higher).
- **Python Runtime**: Python 3.10 to Python 3.12 (Validated on **Python 3.12.3** 64-bit).
- **Package Manager**: Python Pip version 24.0 or higher.
- **Core Virtual Environment**: Standard Python `venv` or Conda isolated environment.
- **Web Browser**: Modern Chromium-based browser (Google Chrome, Microsoft Edge, Brave) or Mozilla Firefox with JavaScript enabled for Streamlit UI rendering.
