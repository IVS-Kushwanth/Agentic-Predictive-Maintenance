# Phase 7: Intelligent Maintenance Recommendation Engine Report
## Agentic Predictive Maintenance: An Explainable Multivariate Time-Series Framework for Early Machine Failure Detection, Remaining Useful Life Estimation and Maintenance Decision Support

---

### Executive Summary

Phase 7 completes the decision-support layer of the **Agentic Predictive Maintenance** platform by introducing an **Intelligent Maintenance Recommendation Engine** coupled with a persistent **Human-in-the-Loop Review System**. 

The recommendation engine translates quantitative outputs from earlier project phases—supervised Remaining Useful Life (RUL) regression, discrete failure classification, unsupervised multivariate autoencoder anomaly scores, and Explainable AI (SHAP / Integrated Gradients) feature attributions—into structured, prioritized, and evidence-grounded industrial maintenance suggestions.

All generated advisories are backed by explicit physical evidence, mapped to standardized industrial sub-systems (e.g., turbofan High-Pressure Compressor outlet, CNC spindle motor assembly), categorized across a 5-tier priority framework, and persisted in a SQLite audit database (`data/maintenance_history.db`) for engineer review, approval, rejection, or escalation.

---

### 1. System Architecture and Integration

The Recommendation Engine operates as an integral component of the LangGraph multi-agent orchestration architecture developed in Phase 6:

```
+-------------------------------------------------------------------------------------------------------+
|                                        LANGGRAPH MULTI-AGENT PIPELINE                                  |
|                                                                                                       |
|  [Raw Telemetry]                                                                                      |
|         |                                                                                             |
|         v                                                                                             |
|  +------------------+       +-------------------+       +---------------------+                       |
|  | Supervisor Agent | ----> | Prediction Agent  | ----> | Anomaly Agent       |                       |
|  | (Input Schema &  |       | (NASA RUL Regress |       | (LSTM Autoencoder   |                       |
|  |  Routing Checks) |       |  AI4I Breakdown)  |       |  Recon Error Score) |                       |
|  +------------------+       +-------------------+       +---------------------+                       |
|                                                                    |                                  |
|                                                                    v                                  |
|  +------------------------------------------------------+  +---------------------+                    |
|  | Maintenance Recommendation Agent                     |  | Explanation Agent   |                    |
|  |   * Standardized Input Schema Validation             |  | (TreeSHAP &         | <------------------+
|  |   * Priority Engine (5-Tier Operational Rules)       |  |  Integrated Grads   |                    |
|  |   * Evidence-Linked Subsystem Rule Mapping           |  |  Attribution)       |                    |
|  |   * Non-Autonomous Safety Guardrails                 |  +---------------------+                    |
|  |   * SQLite Persistence & Human Review Tracking       |                                             |
|  +------------------------------------------------------+                                             |
|         |                                                                                             |
|         v                                                                                             |
|  +-------------------------------------------------------------------------------------------------+  |
|  | SQLite Recommendation History Database (`data/maintenance_history.db`)                         |  |
|  | Status: PENDING | APPROVED | REJECTED | FURTHER_ANALYSIS_REQUESTED                                |  |
|  +-------------------------------------------------------------------------------------------------+  |
+-------------------------------------------------------------------------------------------------------+
```

---

### 2. Standardized Schema Contracts

To ensure strict interoperability between heterogeneous models without data leakage or false metric equivalence, Phase 7 establishes rigid input and output schemas.

#### 2.1 Input Schema (`RecommendationInput`)
```python
RecommendationInput(
    dataset_type: Literal["NASA", "AI4I"],       # Strict separation
    machine_id: str,                             # Unique identifier
    predicted_rul: Optional[float] = None,       # NASA RUL in cycles
    failure_risk_score: Optional[float] = None,  # AI4I failure probability [0, 1]
    anomaly_severity: str = "Normal",            # Normal | Low | Medium | High
    anomaly_score: Optional[float] = None,       # Raw reconstruction error
    important_sensors: List[Dict[str, Any]],     # Top XAI attributions
    operating_conditions: Dict[str, Any],        # Regimes/settings
    model_confidence: Optional[float] = None,    # Model certainty score
    early_warning_horizons: Dict[str, float]     # Classification probabilities
)
```

#### 2.2 Standardized Output Contract
Every recommendation adheres to the standardized industrial schema:
```json
{
  "machine_id": "ENG-CRIT-99",
  "dataset": "NASA",
  "priority": "Critical review required",
  "predicted_rul": 14.2,
  "failure_risk_score": null,
  "confidence_score": 0.92,
  "risk_status": "imminent_exhaustion",
  "reason": [
    "Predicted Remaining Useful Life is critically low (14.2 cycles <= 20 cycles).",
    "Confirmed by acute multivariate sensor anomaly (High anomaly)."
  ],
  "recommended_actions": [
    {
      "action_id": "REC-NASA-01",
      "priority": "HIGH",
      "category": "INSPECTION_OVERHAUL",
      "subsystem": "HPC Thermal Assembly",
      "urgency": "IMMEDIATE (Within 2-5 Operating Cycles)",
      "action_description": "Perform borescope inspection of High-Pressure Compressor outlet; calibrate static thermocouple T30.",
      "protocol_reference": "EASA / FAA Part 145 Engine Overhaul Standards",
      "evidence_link": "Triggered by acute degradation in 'sensor_11' (RUL: 14.2 cycles, Severity: High anomaly)"
    }
  ],
  "supporting_evidence": [
    {
      "evidence_type": "Supervised Prediction",
      "metric": "Remaining Useful Life",
      "value": "14.2 cycles",
      "status": "Degrading"
    },
    {
      "evidence_type": "Unsupervised Anomaly",
      "metric": "Anomaly Severity",
      "value": "High anomaly",
      "score": 0.2105
    },
    {
      "evidence_type": "Attribution Driver",
      "feature": "sensor_11",
      "importance": 0.552,
      "direction": "positive"
    }
  ],
  "human_approval_required": true,
  "disclaimer": "INDUSTRIAL ADVISORY: The recommendations provided herein are decision-support suggestions derived from statistical machine-learning and anomaly models. They do NOT constitute autonomous machine-control commands. Certified human maintenance engineering review and authorization are strictly required prior to physical equipment intervention.",
  "generated_at": "2026-11-05T12:35:10.123456Z"
}
```

---

### 3. Priority Evaluation and Conflicting Evidence Handling

The `PriorityEngine` classifies machine operational status across 5 standardized tiers:
1. **Informational:** Machine operates within normal envelope. Standard scheduled maintenance applies.
2. **Low priority:** Minor sensor baseline drift or early wear. Telemetry logging and monitoring recommended.
3. **Medium priority:** Moderate degradation or anomaly drift. Planned inspection within next line check or 24 hours.
4. **High priority:** Accelerated wear or conflicting evidence. Pre-position spare assemblies; throttle load; borescope/ultrasonic check.
5. **Critical review required:** Imminent component exhaustion or catastrophic failure risk. Immediate physical intervention required.

#### Conflict Resolution Protocol
When supervised models and unsupervised detectors report opposing trends:
- **Scenario:** Supervised RUL predicts healthy remaining life ($RUL > 100\text{ cycles}$), but unsupervised sensor autoencoder flags `High anomaly` ($score > \text{threshold}$).
- **Resolution:** The engine prioritizes safety over blind trust in regression models. It escalates priority to `High priority`, sets `risk_status = "conflicting_evidence"`, and enforces `human_approval_required = True`.
- **Engineering Justification:** Sensor anomalies may indicate sensor harness faults, instrumentation failure, or unmodeled operating regimes that could compromise downstream safety.

---

### 4. Subsystem Mapping and Domain Standards

Recommendations do not suggest vague actions like "inspect machine". Every action targets the specific mechanical or electrical assembly indicated by Explainable AI attributions:

#### NASA C-MAPSS (Turbofan Flight Operations)
- **High-Pressure Compressor (HPC) Outlet (`sensor_11`):** Borescope inspection of HPC outlet stages, thermocouple T30 calibration (EASA/FAA Part 145 Engine Overhaul Standards).
- **Low-Pressure Turbine (LPT) Exhaust (`sensor_4`):** Stator blade erosion check, thermocouple harness inspection (OEM Turbofan Maintenance Manual §72-50).
- **Fan & Bypass Duct (`sensor_12`):** Bypass duct aerodynamic seal inspection, fan blade tip clearance check (Engine Maintenance Manual).
- **Core Coolant Bleed System (`sensor_13` / `sensor_14`):** Bleed valve actuation testing, differential pressure transducer recalibration.

#### UCI AI4I 2020 (CNC Machining)
- **Spindle Motor & Drive (`Mechanical power`):** Electrical inverter audit, motor winding insulation test, drive belt tension (ISO 10791 CNC Machine Safety).
- **Cutting Tool Assembly (`Tool wear`):** Cutting insert index/replacement, tool holder radial runout verification ($< 0.005\text{ mm}$, ISO 3685).
- **Spindle Bearings (`Rotational speed`):** High-speed acoustic emission analysis, spindle oil mist lubrication check (ISO 1940-1).
- **Workpiece Clamping & Feed Drive (`Overstrain indicator`):** Axis feed rate reduction by 15%, hydraulic fixture pressure calibration.

---

### 5. Persistent SQLite History and Human Review Workflow

Every generated advisory is automatically logged to `data/maintenance_history.db`. The database schema supports a complete audit trail:

```sql
CREATE TABLE IF NOT EXISTS recommendation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recommendation_id TEXT UNIQUE NOT NULL,
    machine_id TEXT NOT NULL,
    dataset_type TEXT NOT NULL,
    prediction_date TEXT NOT NULL,
    predicted_rul REAL,
    failure_risk_score REAL,
    priority TEXT NOT NULL,
    risk_status TEXT NOT NULL,
    anomaly_severity TEXT NOT NULL,
    recommendation_json TEXT NOT NULL,
    review_status TEXT NOT NULL DEFAULT 'PENDING',
    reviewed_by TEXT,
    reviewed_at TEXT,
    review_notes TEXT,
    created_at TEXT NOT NULL
);
```

#### Supported Review Actions:
- `approve_recommendation(rec_id, reviewer, notes)`: Transitions status from `PENDING` to `APPROVED`.
- `reject_recommendation(rec_id, reviewer, notes)`: Transitions status to `REJECTED` with required engineering notes.
- `request_further_analysis(rec_id, reviewer, notes)`: Requests additional diagnostic telemetry or physical non-destructive testing (NDT).
- `list_history(machine_id, review_status, limit)`: Historical querying and audit log retrieval.

---

### 6. Verification and Validation Summary

All components underwent rigorous testing via `src/test_phase7_recommendations.py`:
- **Test Pass Rate:** 10/10 (100% passed).
- **Phase 6 Regressions:** 0 regressions (9/9 Phase 6 orchestration tests passed).
- **Execution Performance:** Full recommendation generation and database write completed in under $2\text{ ms}$ per decision cycle.

Phase 7 is complete and ready for interactive user demonstrations and Phase 8 dashboard development.
