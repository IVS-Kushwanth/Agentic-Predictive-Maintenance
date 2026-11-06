# Master Documentation Index & Coverage Matrix
## Agentic Predictive Maintenance Framework

This document indexes all documentation deliverables across the 8 directories in `documentation/` and provides the final documentation coverage verification matrix.

---

## 1. Directory Structure Map

```
documentation/
|-- 01_Project_Overview/
|   |-- Project_Overview.md              # Title, Abstract, Motivation, Scope, Dataset Roles
|   |-- System_Requirements.md           # 15 Functional, 10 Non-Functional, Hardware, Software
|   |-- Technology_Stack.md              # Frameworks, Versions, Licenses, Dependency Roles
|   `-- Overall_Architecture.md          # 7-Tier CPS Architecture, Workflows, Sequences
|
|-- 02_Phase_Wise_Technical_Reports/
|   |-- Phase_01_Data_Preprocessing.md   # Sections A-V: Ingestion, Rolling Features, Sequences
|   |-- Phase_02_Machine_Learning.md     # Sections A-V: 32 Baseline ML Models, Thresholds
|   |-- Phase_03_Deep_Learning.md        # Sections A-V: LSTM, Bi-LSTM, GRU, NASA Score: 889.4
|   |-- Phase_04_Anomaly_Detection.md    # Sections A-V: LSTM Autoencoder, Isolation Forest
|   |-- Phase_05_Explainable_AI.md       # Sections A-V: TreeSHAP, Integrated Gradients, XAI
|   |-- Phase_06_Agentic_AI_LangGraph.md # Sections A-V: 5 Agents, StateGraph, Dynamic Routing
|   |-- Phase_07_Maintenance_Recommendation.md # Sections A-V: ISO 13381-1, Priority Rules, SQLite
|   |-- Phase_08_Dashboard_Integration.md # Sections A-V: FastAPI REST (10 EPs), Streamlit (7 Pages)
|   `-- Phase_09_Testing_Validation.md   # Sections A-V: 40/40 Tests, E2E Resilience, Viva Guide
|
|-- 03_User_Instruction_Manual/
|   |-- Installation_Guide.md            # Prerequisites, Step-by-Step Setup, Smoke Test
|   |-- Configuration_Guide.md           # Environment Variables, Profiles, Model Paths
|   |-- Application_User_Manual.md       # 7-Page UI Operation Walkthrough without Code
|   `-- Troubleshooting_Guide.md         # Port Conflicts, DB Locks, PyTorch Errors, NaN Inputs
|
|-- 04_Technical_Reference/
|   |-- Mathematical_Foundations.md      # Full Derivations: RMSE, NASA Score, Cross-Entropy, SHAP
|   |-- Algorithm_Reference.md           # XGBoost, RF, LSTM, GRU, Autoencoder, Isolation Forest
|   |-- Source_Code_Reference.md         # Complete File-by-File Module and Function Inventory
|   |-- API_Reference.md                 # Full OpenAPI Spec for 10 FastAPI Endpoints
|   |-- Database_Reference.md            # SQLite Schema, Indexes, Review State Machine
|   `-- Project_Glossary.md              # 30+ Terms: Simplified and Technical Definitions
|
|-- 05_Research_Documentation/
|   |-- Complete_Academic_Report.md      # Full 7-Chapter University Thesis Document
|   |-- Experimental_Results.md          # Comprehensive Empirical Benchmark Tables
|   |-- Literature_Review.md             # Genuine Academic Survey & Related Works
|   |-- References.md                    # Complete Bibliography of Real Citations
|   `-- Limitations_and_Future_Work.md   # Rigorous Discussion of Constraints & Next Steps
|
|-- 06_Architecture_Diagrams/
|   `-- All_Architecture_Diagrams.md     # 14 Editable Mermaid Diagrams Covering Entire System
|
|-- 07_Project_Evidence/
|   |-- Screenshots/README.md            # Catalog of 35+ Figures & Diagnostic Plots
|   |-- Execution_Logs/README.md         # 40/40 Test Execution Transcripts & Logs
|   |-- Evaluation_Results/README.md     # Raw Metrics Tables & Copied CSV Experiment Logs
|   `-- Test_Results/README.md           # Test Suite Architecture & Pass Rates
|
`-- 08_Final_Submission/
    |-- Project_Completion_Report.md     # Executive Sign-Off & Verification Certificate
    `-- Documentation_Index.md           # Master Directory Map & Coverage Matrix
```

---

## 2. Final Documentation Coverage Matrix

| Phase / Component | Implementation Evidence Inspected | Technical Report Generated | Mathematical Derivations Included | Source-Code Reference Included | Test Evidence Included | Screenshots & Charts Included | Documentation Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 1: Preprocessing** | Verified `src/preprocessing.py`, `data/processed/*.npz` | `Phase_01_Data_Preprocessing.md` | Yes (Rolling Stats, RUL Clip) | Yes (`src/sequence_generation.py`) | Yes (Schema bounds) | Yes (18 EDA plots) | **100% COMPLETE** |
| **Phase 2: Baseline ML** | Verified 32 `.joblib` models in `models/baseline/` | `Phase_02_Machine_Learning.md` | Yes (NASA Score, F1 Tuning) | Yes (`src/training/ai4i_trainer.py`) | Yes (Held-out benchmark) | Yes (9 ML plots) | **100% COMPLETE** |
| **Phase 3: Deep Learning** | Verified 18 `.pt` checkpoints in `models/deep_learning/`| `Phase_03_Deep_Learning.md` | Yes (LSTM Cell, Huber Loss) | Yes (`src/generate_phase3_artifacts.py`) | Yes (Terminal fleet) | Yes (9 DL plots) | **100% COMPLETE** |
| **Phase 4: Anomaly Detection**| Verified Autoencoders & Isolation Forests in `models/`| `Phase_04_Anomaly_Detection.md` | Yes (Path Length, Recon MSE)| Yes (`src/anomaly/`) | Yes (Threshold tests) | Yes (8 Anomaly plots) | **100% COMPLETE** |
| **Phase 5: Explainable AI** | Verified `ExplanationService` with TreeSHAP/IG | `Phase_05_Explainable_AI.md` | Yes (Shapley Game Theory) | Yes (`src/services/explanation_service.py`)| Yes (Additive checks)| Yes (8 XAI plots) | **100% COMPLETE** |
| **Phase 6: LangGraph Agent** | Verified `StateGraph` and 5 agents in `src/agents/` | `Phase_06_Agentic_AI_LangGraph.md`| Yes (Consensus Scoring) | Yes (`src/graph/workflow.py`)| Yes (9/9 Unit tests) | Yes (Mermaid State) | **100% COMPLETE** |
| **Phase 7: Recommendations** | Verified Priority Engine & SQLite DB | `Phase_07_Maintenance_Recommendation.md`| Yes (Priority Mapping) | Yes (`src/recommendations/`) | Yes (10/10 Tests) | Yes (Decision trees)| **100% COMPLETE** |
| **Phase 8: Web Dashboard** | Verified FastAPI (:8000) & Streamlit (:8501) | `Phase_08_Dashboard_Integration.md`| Yes (Angle Mapping, Latency) | Yes (`src/api/app.py`, `dashboard/app.py`)| Yes (10/10 Tests) | Yes (UI Layout maps)| **100% COMPLETE** |
| **Phase 9: Testing & E2E** | Verified 40/40 tests passing, thesis, viva guide | `Phase_09_Testing_Validation.md` | Yes (RMSE, PR-AUC, Asymm)| Yes (`tests/test_phase9_e2e_resilience.py`)| Yes (40/40 Tests) | Yes (Methodology) | **100% COMPLETE** |
