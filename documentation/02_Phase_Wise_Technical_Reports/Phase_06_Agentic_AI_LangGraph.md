# Phase 6: Agentic AI Orchestration Using LangGraph Technical Report

## A. Phase Introduction
Phase 6 implements the agentic decision-orchestration layer using LangGraph. In enterprise industrial applications, predictive models, anomaly detectors, and explainability engines must not exist as disjointed analytical silos. Phase 6 synthesizes these capabilities into a collaborative, multi-agent cybernetic system capable of dynamic routing, consensus evaluation, state management, and verifiable decision synthesis.

---

## B. Objective of the Phase
1. Formulate a multi-agent industrial orchestration architecture using LangGraph `StateGraph`.
2. Implement 5 specialized agents: `SupervisorAgent`, `PredictionAgent`, `AnomalyAgent`, `ExplanationAgent`, and `MaintenanceAgent`.
3. Establish a shared, type-safe `MaintenanceState` TypedDict maintaining complete telemetry and diagnostic context.
4. Implement dataset-aware conditional routing dynamically directing requests to NASA turbofan or AI4I milling pipelines.
5. Engineer fault-tolerant error recovery, input validation, and safety disclaimer enforcement.
6. Verify the multi-agent graph across nominal, degraded, and abnormal edge-case operational scenarios.

---

## C. Why This Phase Is Required
Traditional sequential prediction scripts suffer from rigid fragility:
- **No Autonomous Consensus**: If a supervised model predicts a long Remaining Useful Life but an unsupervised autoencoder detects a severe anomaly, a simple script fails to reconcile the conflict. The multi-agent graph dynamically evaluates cross-model consensus.
- **Lack of Adaptive Context**: Different machines and operational regimes require different analytical pipelines. An agentic supervisor inspects incoming telemetry metadata and selects the optimal diagnostic route.
- **Production State Auditability**: Industrial plants require an inspectable, step-by-step audit trail showing which agent executed which analysis, what evidence was gathered, and why a maintenance action was proposed.

---

## D. Tasks Performed
- Designed and compiled `StateGraph(MaintenanceState)` in `src/graph/workflow.py`.
- Implemented 5 specialized agent classes in `src/agents/`.
- Configured conditional edge routing (`route_supervisor_decision`, `route_after_prediction`, etc.).
- Implemented `ResultValidationNode` ensuring schema compliance and appending mandatory safety disclaimers.
- Executed unit and integration testing suite (`src/test_phase6_orchestration.py`) with 9/9 passing tests.
- Serialized execution logs and workflow documentation into `reports/phase6/`.

---

## E. Detailed Implementation Methodology

```mermaid
graph TD
    START([START: Ingest Telemetry Payload]) --> SUP["supervisor"]
    SUP --> R_SUP{"route_supervisor_decision"}
    R_SUP -- "Valid Payload" --> PRED["prediction_agent"]
    R_SUP -- "Payload Fault" --> VAL["result_validation"]
    
    PRED --> R_PRED{"route_after_prediction"}
    R_PRED -- "Prediction OK" --> ANOM["anomaly_agent"]
    R_PRED -- "Inference Fault" --> VAL
    
    ANOM --> R_ANOM{"route_after_anomaly"}
    R_ANOM -- "Anomaly Scored" --> EXPL["explanation_agent"]
    R_ANOM -- "Scoring Fault" --> VAL
    
    EXPL --> R_EXPL{"route_after_explanation"}
    R_EXPL -- "Attributions Found" --> MAINT["maintenance_agent"]
    R_EXPL -- "Explainer Fault" --> VAL
    
    MAINT --> VAL
    VAL --> END_NODE([END: Return Validated State])
```

---

## F. Algorithms and Techniques Used
- **LangGraph StateGraph**: Deterministic, directed acyclic and cyclical state-machine orchestration.
- **TypedDict State Management**: Strictly typed shared execution dictionary ensuring immutability across nodes.
- **Conditional Edge Routing**: Dynamic functional predicates directing graph flow based on state contents.
- **Decoupled Worker Instantiation**: Lazy singleton worker instantiation avoiding top-level circular imports.

---

## G. Mathematical Explanation of Relevant Algorithms

### 1. Multi-Model Consensus Scoring
To detect conflicting diagnostic signals between supervised Remaining Useful Life ($\hat{y}_{RUL}$) and unsupervised anomaly score ($s_{anom} \in [0, 1]$):
$$\text{Consensus Discrepancy } D = \left| \frac{\min(\hat{y}_{RUL}, 125)}{125} - (1 - s_{anom}) \right|$$
If $D > 0.50$ (e.g., $\hat{y}_{RUL} = 115$ but $s_{anom} = 0.88$), the `SupervisorAgent` flags a **Cross-Model Discrepancy Alert**, overrides automated priority, and mandates human engineering inspection.

### 2. State Transition Function
Let $\mathcal{S}$ denote the space of `MaintenanceState` dictionaries and $\mathcal{N}_k: \mathcal{S} \to \mathcal{S}$ denote the state transformation executed by Agent $k$:
$$S_{k+1} = \mathcal{N}_k(S_k) = S_k \cup \{ \text{key}_k: \text{output}_k \}$$
Subject to invariant validation constraint:
$$\forall S_k, \quad \text{validate}(S_k) = \text{True}$$

---

## H. Libraries and Technologies Used
- **LangGraph 0.2.64**: Multi-agent state-graph orchestration.
- **LangChain-Core 0.3.31**: Base schemas and agent messaging contracts.
- **Pydantic 2.10.5**: Schema validation and serialization.
- **Python Standard Library (`typing`, `unittest`)**: Type definitions and testing harness.

---

## I. Input Datasets and Input Formats
- Telemetry payload: Python dictionary or JSON payload containing:
  - `dataset`: `"NASA"` or `"AI4I"`.
  - `machine_id`: String identifier (e.g., `"ENGINE-024"`, `"CNC-MILL-08"`).
  - `sensor_data`: Numerical dictionary or 2D array of sensor telemetry.
  - `metadata`: Operating settings, cycle count, or machine type.

---

## J. Data Processing and Transformation
- The `SupervisorAgent` normalizes input data into the format required by downstream model services.
- The `MaintenanceState` is progressively enriched at each step without mutating original sensor inputs.

---

## K. Source-Code Files Involved
- [`src/graph/state.py`](file:///C:/Users/HP/OneDrive/Desktop/ML%20Model/agentic-predictive-maintenance/src/graph/state.py): `MaintenanceState` TypedDict definition.
- [`src/graph/workflow.py`](file:///C:/Users/HP/OneDrive/Desktop/ML%20Model/agentic-predictive-maintenance/src/graph/workflow.py): StateGraph assembly, node binding, and compilation.
- [`src/graph/routing.py`](file:///C:/Users/HP/OneDrive/Desktop/ML%20Model/agentic-predictive-maintenance/src/graph/routing.py): Conditional edge routing predicates.
- [`src/agents/supervisor.py`](file:///C:/Users/HP/OneDrive/Desktop/ML%20Model/agentic-predictive-maintenance/src/agents/supervisor.py): Supervisor Agent.
- [`src/agents/prediction_agent.py`](file:///C:/Users/HP/OneDrive/Desktop/ML%20Model/agentic-predictive-maintenance/src/agents/prediction_agent.py): Prediction Agent.
- [`src/agents/anomaly_agent.py`](file:///C:/Users/HP/OneDrive/Desktop/ML%20Model/agentic-predictive-maintenance/src/agents/anomaly_agent.py): Anomaly Detection Agent.
- [`src/agents/explanation_agent.py`](file:///C:/Users/HP/OneDrive/Desktop/ML%20Model/agentic-predictive-maintenance/src/agents/explanation_agent.py): Explanation Agent.
- [`src/agents/maintenance_agent.py`](file:///C:/Users/HP/OneDrive/Desktop/ML%20Model/agentic-predictive-maintenance/src/agents/maintenance_agent.py): Maintenance Recommendation Agent.
- [`src/test_phase6_orchestration.py`](file:///C:/Users/HP/OneDrive/Desktop/ML%20Model/agentic-predictive-maintenance/src/test_phase6_orchestration.py): Automated test suite.

---

## L. Detailed Explanation of Important Functions and Classes
- `create_maintenance_graph()`: Constructs the `StateGraph(MaintenanceState)`, adds all 6 nodes, binds conditional edges, and compiles the executable graph.
- `execute_workflow(input_data)`: Public entrypoint taking telemetry and running graph invocation to completion.
- `route_supervisor_decision(state)`: Checks whether `state["errors"]` contains fatal validation faults; routes to `prediction_agent` or short-circuits to `result_validation`.
- `result_validation_node(state)`: Verifies all expected keys exist, asserts safety disclaimers, and formats final status badges.

---

## M. Execution Procedure
```powershell
# Run the complete Phase 6 multi-agent orchestration test suite
python src/test_phase6_orchestration.py

# Verify deliverables and graph compilation
python src/verify_phase6_deliverables.py
```

---

## N. Actual Output Generated
### Automated Orchestration Test Suite Results (9/9 PASS):
```
  [PASS] Test 1: NASA Nominal Workflow Verified
  [PASS] Test 2: NASA Degraded Workflow & Human Review Verified
  [PASS] Test 3: AI4I Normal Workflow Verified
  [PASS] Test 4: AI4I Breakdown Workflow & Safeguards Verified
  [PASS] Test 5: Missing Input Gracefully Captured
  [PASS] Test 6: Invalid Sensor Data Resiliency Verified
  [PASS] Test 7: Dedicated Service Interfaces Verified
  [PASS] Test 8: Output Schema 100% Compliant
  [PASS] Test 9: Safety Disclaimer & Decision-Support Protocol Verified
Ran 9 tests in 4.244s. OK.
```

### Real Orchestration State Output (Engine #24, Cycle 130):
- **Predicted RUL**: `18.2 cycles`
- **Failure Risk**: `HIGH`
- **Anomaly Score**: `0.8420` (`HIGH anomaly`)
- **Top Sensors Attributed**: `sensor_11 (HPC Temp: +38.4%)`, `sensor_4 (LPT Temp: +24.1%)`
- **Maintenance Priority**: `CRITICAL`
- **Action Plan**: `Immediate turbofan stand-down and borescope inspection of high-pressure compressor stage.`
- **Mandatory Disclaimer**: `"ADVISORY NOTICE: Generated for human-in-the-loop decision support. Mandatory qualified engineering sign-off required."`

---

## O. Screenshots, Charts and Visualizations
Saved in `reports/phase6/`:
- Multi-agent state transition graph rendered via Mermaid.
- Diagnostic routing decision trees documented in `reports/phase6/workflow_documentation.md`.

---

## P. Testing Methodology
- **Nominal Telemetry Test**: Verified that healthy telemetry yields `INFORMATIONAL` / `LOW` priority without false escalation.
- **Degraded Telemetry Test**: Verified that end-of-life telemetry triggers `CRITICAL` priority with borescope work orders.
- **Fault-Tolerance Test**: Tested invalid sensor inputs (`NaN`, missing keys, negative temperatures); verified graceful HTTP 422 / error capture without Python process crashes.

---

## Q. Actual Test Results
- All 9 unit and integration tests passed in 4.244 seconds.
- Mean graph execution latency on CPU: **2.85 seconds** (including full TreeSHAP calculation).
- Zero unhandled exceptions encountered.

---

## R. Problems Encountered and Solutions
1. *Issue:* Top-level circular import between `src.agents` and `src.graph` caused runtime `ImportError`.
   *Solution:* Refactored `workflow.py` to use lazy worker instantiation functions (`get_supervisor()`, `get_predictor()`), breaking the circular dependency.
2. *Issue:* `MaintenanceState` TypedDict was silently dropping `recommendation_id` because it was not explicitly annotated.
   *Solution:* Updated `src/graph/state.py` with complete TypedDict annotations for `recommendation_id`, `priority`, and `recommendation_output`.

---

## S. Technical Limitations
- The current graph runs sequentially through nodes; parallel asynchronous execution of Anomaly and Prediction agents can further reduce latency.
- Graph state is held in-memory during execution and persisted to SQLite upon completion.

---

## T. Final Deliverables
- 5 agent classes in `src/agents/`.
- Compiled graph pipeline in `src/graph/workflow.py`.
- Automated test suite in `src/test_phase6_orchestration.py`.
- 3 technical reports in `reports/phase6/`.

---

## U. Connection with the Next Phase
The multi-agent workflow compiles raw diagnostic evidence into a unified state dictionary. **Phase 7: Intelligent Maintenance Recommendation Engine** formalizes the industrial decision logic, ISO 13381-1 priority rules, and human-in-the-loop review persistence.

---

## V. Phase Completion Assessment
**Status: 100% Complete & Empirically Verified.** All agents implemented, graph compiled, conditional routing verified, and test suite 100% passing.
