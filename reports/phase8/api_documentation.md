# Phase 8: REST API Documentation
## Agentic Predictive Maintenance: FastAPI Backend Service Specification

---

### 1. Architectural Overview

The backend service is built with **FastAPI** to provide low-latency, strongly typed, and schema-validated endpoints for industrial predictive maintenance inference, anomaly detection, Explainable AI, multi-agent LangGraph execution, and human review persistence.

- **Base URL:** `http://localhost:8000`
- **Interactive Swagger Documentation:** `http://localhost:8000/docs`
- **ReDoc Documentation:** `http://localhost:8000/redoc`
- **Specification Format:** OpenAPI 3.1
- **Persistence Layer:** SQLite (`data/maintenance_history.db`)

#### Performance and Caching Strategy
To prevent excessive disk I/O and latency spikes under concurrent loads, all trained baseline models (XGBoost, Random Forest), deep learning checkpoints, and TreeSHAP explainers are loaded lazily into an in-memory singleton cache (`app.backend.services.model_cache`). Successive inference requests execute in sub-millisecond time without reloading `.joblib` or `.pt` artifacts.

---

### 2. Complete Endpoint Catalog

#### 2.1 System & Catalog Endpoints

##### `GET /health`
Confirms API operational status.
- **Response `200 OK`:**
```json
{
  "status": "healthy",
  "service": "Agentic Predictive Maintenance API",
  "version": "1.0.0",
  "timestamp": "2026-11-05T12:50:00.123456Z"
}
```

##### `GET /api/models`
Returns the verified metadata registry of all trained models across NASA C-MAPSS and UCI AI4I 2020.
- **Response `200 OK`:**
```json
{
  "total_models": 12,
  "models": [
    {
      "model_id": "nasa_rul_xgboost_regressor_FD001",
      "dataset": "NASA",
      "task": "rul_regression",
      "framework": "XGBoost",
      "description": "Gradient boosted trees for turbofan Remaining Useful Life regression on FD001 subset.",
      "metrics": {
        "RMSE": 17.29,
        "MAE": 12.45,
        "R2": 0.812
      }
    },
    {
      "model_id": "ai4i_clf_xgboost_classifier",
      "dataset": "AI4I",
      "task": "failure_classification",
      "framework": "XGBoost",
      "description": "Cost-sensitive classifier predicting imminent CNC breakdown with threshold calibration.",
      "metrics": {
        "F1_score": 0.835,
        "Recall": 0.865,
        "ROC_AUC": 0.978,
        "Threshold": 0.7969
      }
    }
  ]
}
```

---

#### 2.2 Supervised Prediction Endpoints

##### `POST /api/predict/rul`
Executes Remaining Useful Life regression for NASA C-MAPSS turbofan engines.
- **Request Body:**
```json
{
  "machine_id": "ENG-FD001-010",
  "sensor_features": [0.12, -0.45, ...], // Exactly 153 features
  "model_type": "xgboost"
}
```
- **Response `200 OK`:**
```json
{
  "machine_id": "ENG-FD001-010",
  "dataset": "NASA",
  "predicted_rul": 138.45,
  "unit": "cycles",
  "model_used": "nasa_rul_xgboost_regressor_FD001",
  "early_warning_status": {
    "risk_within_10_cycles": 0.012,
    "risk_within_20_cycles": 0.035,
    "risk_within_30_cycles": 0.082
  },
  "confidence": 0.88
}
```

##### `POST /api/predict/failure`
Executes breakdown classification for UCI AI4I 2020 CNC machine tools.
- **Request Body:**
```json
{
  "machine_id": "CNC-Tool-005",
  "operating_features": [298.1, 308.6, 1500, 42.5, 215, ...], // Exactly 14 features
  "model_type": "xgboost"
}
```
- **Response `200 OK`:**
```json
{
  "machine_id": "CNC-Tool-005",
  "dataset": "AI4I",
  "failure_risk_score": 0.8421,
  "failure_prediction": 1,
  "failure_label": "Machine Failure",
  "model_used": "ai4i_clf_xgboost_classifier",
  "is_calibrated": true,
  "confidence": 0.92
}
```

---

#### 2.3 Anomaly Detection & Explainable AI Endpoints

##### `POST /api/anomaly/detect`
Executes unsupervised sensor anomaly detection via Autoencoders or Isolation Forests.
- **Request Body:**
```json
{
  "machine_id": "ENG-FD001-010",
  "dataset_type": "NASA",
  "features": [0.12, -0.45, ...],
  "detector_type": "autoencoder"
}
```
- **Response `200 OK`:**
```json
{
  "machine_id": "ENG-FD001-010",
  "dataset": "NASA",
  "anomaly_score": 0.018245,
  "threshold": 0.037,
  "is_anomaly": false,
  "severity": "Normal",
  "detector_used": "lstm_autoencoder"
}
```

##### `POST /api/explain`
Calculates TreeSHAP feature attributions and ranks contributing sensor channels.
- **Request Body:**
```json
{
  "machine_id": "CNC-Tool-005",
  "dataset_type": "AI4I",
  "features": [298.1, 308.6, ...],
  "method": "shap",
  "top_k": 3
}
```
- **Response `200 OK`:**
```json
{
  "machine_id": "CNC-Tool-005",
  "dataset": "AI4I",
  "method": "shap",
  "top_contributing_sensors": [
    {
      "feature": "Mechanical power",
      "importance": 0.4521,
      "contribution": "positive",
      "value": 6375.0
    },
    {
      "feature": "Tool wear",
      "importance": 0.2814,
      "contribution": "positive",
      "value": 215.0
    },
    {
      "feature": "Torque to speed ratio",
      "importance": 0.1105,
      "contribution": "positive",
      "value": 0.0283
    }
  ],
  "explanation_summary": "Prediction primarily driven by 'Mechanical power' with importance weight 0.4521."
}
```

---

#### 2.4 Multi-Agent Orchestration & Prescriptive Recommendations

##### `POST /api/agents/analyze`
Executes end-to-end multi-agent LangGraph workflow across all specialized agents.
- **Request Body:**
```json
{
  "machine_id": "ENG-FD001-099",
  "dataset_type": "NASA",
  "sensor_data": [0.42, 0.88, ...],
  "workflow_type": "full_analysis"
}
```
- **Response `200 OK`:**
```json
{
  "dataset": "NASA",
  "machine_id": "ENG-FD001-099",
  "workflow_status": "COMPLETED",
  "recommendation_id": "REC-90CA5F90",
  "priority": "Critical review required",
  "risk_status": "imminent_exhaustion",
  "rul_prediction": 14.2,
  "failure_risk_score": null,
  "anomaly_severity": "High anomaly",
  "anomaly_score": 0.2105,
  "important_sensor_contributions": [...],
  "explanation_summary": "ENGINEERING DECISION SUPPORT BRIEF — ENG-FD001-099 ...",
  "suggested_maintenance_actions": [...],
  "human_review_status": {
    "human_review_required": true,
    "disclaimer": "INDUSTRIAL ADVISORY: ... Certified human maintenance engineering review required."
  }
}
```

##### `POST /api/recommendations`
Generates prioritized maintenance action items directly from model evidence and saves to SQLite.
- **Request Body:**
```json
{
  "dataset_type": "NASA",
  "machine_id": "ENG-REC-01",
  "predicted_rul": 15.0,
  "anomaly_severity": "High anomaly",
  "anomaly_score": 0.22,
  "important_sensors": [{"feature": "sensor_11", "importance": 0.45, "contribution": "positive"}]
}
```
- **Response `200 OK`:** Returns standardized recommendation object with unique `recommendation_id`.

---

#### 2.5 Historical Audits & Certified Human Review

##### `GET /api/history`
Queries historical recommendations from SQLite with optional machine and status filters.
- **Parameters:**
  - `machine_id` (optional, string)
  - `review_status` (optional, string: `PENDING`, `APPROVED`, `REJECTED`, `FURTHER_ANALYSIS_REQUESTED`)
  - `limit` (optional, integer: 1–200, default 50)
- **Response `200 OK`:**
```json
{
  "total_records": 1,
  "records": [
    {
      "id": 1,
      "recommendation_id": "REC-90CA5F90",
      "machine_id": "ENG-FD001-099",
      "dataset_type": "NASA",
      "prediction_date": "2026-11-05T12:35:10Z",
      "predicted_rul": 14.2,
      "priority": "Critical review required",
      "risk_status": "imminent_exhaustion",
      "anomaly_severity": "High anomaly",
      "review_status": "PENDING",
      "reviewed_by": null,
      "created_at": "2026-11-05T12:35:10Z",
      "recommendation_data": {...}
    }
  ]
}
```

##### `POST /api/recommendations/review`
Submits certified human engineer authorization, updating status and logging notes.
- **Request Body:**
```json
{
  "recommendation_id": "REC-90CA5F90",
  "status": "APPROVED",
  "reviewed_by": "Lead Propulsion Eng. Miller",
  "review_notes": "Borescope inspection scheduled for next line maintenance cycle."
}
```
- **Response `200 OK`:**
```json
{
  "recommendation_id": "REC-90CA5F90",
  "review_status": "APPROVED",
  "reviewed_by": "Lead Propulsion Eng. Miller",
  "reviewed_at": "2026-11-05T12:45:00.123456Z",
  "review_notes": "Borescope inspection scheduled for next line maintenance cycle.",
  "success": true
}
```

---

### 3. HTTP Status Codes and Error Formats

| Status Code | Description | Trigger Scenario |
| :--- | :--- | :--- |
| `200 OK` | Request succeeded | Valid inference, query, or review commit |
| `422 Unprocessable Content` | Validation failure | Missing required fields, wrong tensor dimension (e.g., 5 features instead of 14) |
| `404 Not Found` | Entity not found | Attempting to review a non-existent `recommendation_id` |
| `500 Internal Server Error` | Execution error | Uncaught service exception with sanitized error payload |

Standard JSON Error Schema:
```json
{
  "error": "UnprocessableContent",
  "message": "NASA C-MAPSS prediction requires exactly 153 sensor features, received 12",
  "path": "/api/predict/rul"
}
```
