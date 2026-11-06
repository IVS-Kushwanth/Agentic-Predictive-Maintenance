# REST API Reference & OpenAPI Specification
## Agentic Predictive Maintenance Backend Microservice

The backend exposes a high-performance RESTful API powered by **FastAPI** on port `8000`. Full interactive OpenAPI documentation is accessible at `http://127.0.0.1:8000/docs`.

---

## 1. Endpoint Summary Matrix

| HTTP Method | Route Endpoint | Purpose / Description | Request Schema | Response Status |
| :--- | :--- | :--- | :--- | :--- |
| **GET** | `/health` | Service health status, uptime, and version. | None | `200 OK` |
| **GET** | `/api/models` | Registry of all loaded ML, DL, and anomaly models. | None | `200 OK` |
| **POST** | `/api/predict/rul` | Turbofan Remaining Useful Life (RUL) regression. | `PredictRULRequest` | `200 OK` / `422` |
| **POST** | `/api/predict/failure` | CNC machine failure binary classification. | `PredictFailureRequest` | `200 OK` / `422` |
| **POST** | `/api/anomaly/detect` | Multivariate sensor anomaly detection & scoring. | `AnomalyRequest` | `200 OK` / `422` |
| **POST** | `/api/explain` | TreeSHAP feature attribution calculation. | `ExplainRequest` | `200 OK` / `422` |
| **POST** | `/api/agents/analyze` | Full LangGraph multi-agent orchestration workflow. | `AgentAnalyzeRequest` | `200 OK` / `422` |
| **POST** | `/api/recommendations` | Direct maintenance recommendation synthesis. | `RecommendationRequest` | `200 OK` / `422` |
| **GET** | `/api/history` | Chronological recommendation audit trail. | Query params | `200 OK` |
| **POST** | `/api/recommendations/review` | Human review update (Approve/Reject). | `ReviewUpdateRequest` | `200 OK` / `422` |

---

## 2. Detailed Endpoint Specifications

### 2.1 GET `/health`
- **Request**: No body.
- **Response**:
```json
{
  "status": "healthy",
  "service": "agentic-predictive-maintenance-backend",
  "version": "1.0.0",
  "uptime_seconds": 1420.5
}
```

### 2.2 POST `/api/predict/rul`
- **Request Body (`PredictRULRequest`)**:
```json
{
  "unit_number": 24,
  "cycle": 130,
  "sensor_values": {
    "sensor_2": 642.5,
    "sensor_3": 1585.2,
    "sensor_4": 1405.8,
    "sensor_7": 553.2,
    "sensor_11": 47.6,
    "sensor_12": 521.4
  },
  "model_type": "bilstm_w60"
}
```
- **Response Body**:
```json
{
  "unit_number": 24,
  "cycle": 130,
  "predicted_rul": 18.25,
  "risk_status": "HIGH_RISK",
  "early_warning_flags": {
    "horizon_10": false,
    "horizon_20": true,
    "horizon_30": true
  },
  "model_used": "bilstm_w60"
}
```

### 2.3 POST `/api/predict/failure`
- **Request Body (`PredictFailureRequest`)**:
```json
{
  "air_temperature": 300.2,
  "process_temperature": 310.8,
  "rotational_speed": 1380,
  "torque": 58.4,
  "tool_wear": 215,
  "machine_type": "M"
}
```
- **Response Body**:
```json
{
  "failure_predicted": true,
  "failure_probability": 0.8420,
  "decision_threshold": 0.485,
  "status": "BREAKDOWN_WARNING",
  "suspected_failure_modes": ["Tool Wear Failure (TWF)", "Overstrain Failure (OSF)"]
}
```

### 2.4 POST `/api/agents/analyze`
- **Request Body (`AgentAnalyzeRequest`)**:
```json
{
  "dataset": "NASA",
  "machine_id": "ENGINE-024",
  "sensor_data": {
    "sensor_2": 642.5,
    "sensor_3": 1585.2,
    "sensor_4": 1405.8,
    "sensor_7": 553.2,
    "sensor_11": 47.6,
    "sensor_12": 521.4
  }
}
```
- **Response Body**:
```json
{
  "workflow_status": "COMPLETED",
  "machine_id": "ENGINE-024",
  "predicted_rul": 18.2,
  "anomaly_score": 0.842,
  "anomaly_severity": "HIGH",
  "top_contributing_sensors": [
    {"sensor": "sensor_11", "attribution": 0.384},
    {"sensor": "sensor_4", "attribution": 0.241}
  ],
  "recommendation": {
    "recommendation_id": "REC-20261105-ENG024",
    "priority": "CRITICAL",
    "action_title": "Turbofan Stand-Down: High-Pressure Compressor Borescope Inspection",
    "sign_off_required": "Chief Propulsion Reliability Engineer",
    "review_status": "PENDING"
  },
  "safety_disclaimer": "ADVISORY NOTICE: Generated for human-in-the-loop decision support. Mandatory qualified engineering sign-off required."
}
```

### 2.5 POST `/api/recommendations/review`
- **Request Body (`ReviewUpdateRequest`)**:
```json
{
  "recommendation_id": "REC-20261105-ENG024",
  "review_status": "APPROVED",
  "reviewer_notes": "Borescope inspection scheduled for 06:00 hangar shift."
}
```
- **Response Body**:
```json
{
  "recommendation_id": "REC-20261105-ENG024",
  "updated_status": "APPROVED",
  "timestamp": "2026-11-05T19:00:00Z",
  "success": true
}
```
