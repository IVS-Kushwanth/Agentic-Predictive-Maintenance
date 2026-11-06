# Source Code & Module Inventory Reference
## Agentic Predictive Maintenance Framework

This document catalogs every source file, module, class, function signature, and dependency within the `src/` and `tests/` directories.

---

## 1. Directory Structure Inventory

```
agentic-predictive-maintenance/
|-- src/
|   |-- data_download.py               # Automated raw benchmark downloader
|   |-- data_validation.py             # Schema, null, and duplicate integrity auditor
|   |-- nasa_loader.py                 # NASA C-MAPSS text stream parsing
|   |-- ai4i_loader.py                 # AI4I CSV loader and target isolator
|   |-- feature_engineering.py         # Rolling window statistical feature synthesizer
|   |-- sequence_generation.py         # Sliding sequence tensor builder
|   |-- rul_generation.py              # Piecewise linear RUL target generator
|   |-- run_phase2.py                  # Phase 2 ML execution pipeline
|   |-- run_phase3.py                  # Phase 3 Deep Learning execution pipeline
|   |-- run_phase4.py                  # Phase 4 Anomaly Detection execution pipeline
|   |-- run_phase5.py                  # Phase 5 XAI execution pipeline
|   |-- generate_phase3_artifacts.py   # PyTorch DL architectures and training loops
|   |-- final_validation_check.py      # Cross-phase deliverable verification
|   |
|   |-- agents/                        # Multi-agent implementations
|   |   |-- supervisor.py              # Supervisor Agent (state router)
|   |   |-- prediction_agent.py        # Supervised Prediction Agent
|   |   |-- anomaly_agent.py           # Anomaly Detection Agent
|   |   |-- explanation_agent.py       # Explainable AI Agent
|   |   `-- maintenance_agent.py       # Prescriptive Maintenance Agent
|   |
|   |-- graph/                         # LangGraph state graph definitions
|   |   |-- state.py                   # MaintenanceState TypedDict definition
|   |   |-- workflow.py                # StateGraph assembly, nodes, and compilation
|   |   `-- routing.py                 # Conditional edge routing predicates
|   |
|   |-- recommendations/               # Decision-support engine
|   |   |-- recommendation_schema.py   # Pydantic input/output schemas
|   |   |-- priority_engine.py         # 5-tier ISO 13381-1 priority rules
|   |   |-- recommendation_rules.py    # Subsystem action mapping rules
|   |   `-- recommendation_engine.py   # SQLite persistence manager
|   |
|   |-- services/                      # Decoupled business logic services
|   |   |-- model_service.py           # Supervised model inference service
|   |   `-- explanation_service.py     # TreeSHAP & Integrated Gradients service
|   |
|   |-- models/
|   |   `-- registry.py                # Central model path registry and loaders
|   |
|   |-- api/
|   |   `-- app.py                     # FastAPI REST microservice (10 endpoints)
|   |
|   `-- dashboard/
|       `-- app.py                     # Streamlit 7-page interactive dashboard
|
|-- tests/
|   |-- test_phase8_backend.py         # FastAPI REST integration tests (10 tests)
|   `-- test_phase9_e2e_resilience.py  # End-to-end resilience & lifecycle tests (11 tests)
|
`-- run_dashboard.py                   # Dual-process backend/frontend startup script
```

---

## 2. Core Module & Function Signatures

### Module: `src/data_validation.py`
- `validate_raw_data() -> dict`: Scans `data/raw/` archives; verifies non-null counts, row counts, and column schemas; returns status dictionary.

### Module: `src/feature_engineering.py`
- `generate_rolling_features(df: pd.DataFrame, sensor_cols: list, windows=[5, 15, 30]) -> pd.DataFrame`: Synthesizes rolling mean, rolling standard deviation, lag differences, and baseline deviations for specified sensors per engine.

### Module: `src/sequence_generation.py`
- `create_sliding_sequences(df: pd.DataFrame, window_size: int, feature_cols: list, target_col: str) -> tuple[np.ndarray, np.ndarray]`: Generates 3D NumPy array of shape $(N, L, F)$ and 1D target array of shape $(N,)$.

### Module: `src/graph/workflow.py`
- `create_maintenance_graph() -> CompiledGraph`: Constructs, binds conditional edges, and compiles the multi-agent `StateGraph(MaintenanceState)`.
- `execute_workflow(input_data: dict) -> dict`: Public invocation entrypoint taking input sensor payload and running the LangGraph state machine.

### Module: `src/recommendations/recommendation_engine.py`
- `class RecommendationEngine`:
  - `__init__(db_path="data/maintenance_history.db")`: Initializes SQLite connection and tables.
  - `generate_recommendation(input_data: RecommendationInput) -> RecommendationOutput`: Evaluates priority, maps subsystem, generates work order, and persists to SQLite with status `PENDING`.
  - `update_review_status(recommendation_id: str, status: str, notes: str) -> bool`: Updates review status in SQLite to `APPROVED`, `REJECTED`, or `FURTHER_ANALYSIS_REQUESTED`.
  - `get_history(limit=50) -> list[dict]`: Retrieves chronological recommendation records.

### Module: `src/api/app.py`
- `create_api_app() -> FastAPI`: Instantiates the ASGI FastAPI application with CORS middleware, exception handlers, and 10 REST endpoints.
