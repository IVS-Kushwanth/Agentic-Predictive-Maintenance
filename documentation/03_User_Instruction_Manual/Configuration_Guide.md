# System Configuration Guide
## Agentic Predictive Maintenance Framework

This guide details all configuration options, environment variables, network port bindings, model paths, and operational thresholds across the framework.

---

## 1. Environment Variables

The application reads configuration parameters from operating system environment variables or an optional `.env` file located in the project root directory.

| Variable Name | Default Value | Permitted Values | Architectural Purpose |
| :--- | :--- | :--- | :--- |
| `FASTAPI_HOST` | `127.0.0.1` | Valid IPv4 / `0.0.0.0` | Host IP interface for FastAPI REST server. |
| `FASTAPI_PORT` | `8000` | Port `1024 - 65535` | Listening port for FastAPI microservice. |
| `STREAMLIT_PORT` | `8501` | Port `1024 - 65535` | Listening port for Streamlit web dashboard. |
| `DATABASE_URL` | `sqlite:///data/maintenance_history.db` | SQLite URI | Path to SQLite audit and review database. |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | System logging verbosity across all modules. |
| `DEVICE` | `cpu` | `cpu`, `cuda` | PyTorch inference hardware device. |
| `MAX_RUL_CLIP` | `125` | Positive Integer | Piecewise linear RUL target threshold (cycles). |
| `AI4I_THRESHOLD` | `0.485` | Float $[0.0, 1.0]$ | Calibrated decision threshold for machine failure. |
| `ANOMALY_THRESHOLD` | `0.0370` | Positive Float | Calibrated nominal reconstruction error threshold. |
| `CORS_ORIGINS` | `*` | Comma-separated URLs | Permitted Cross-Origin Resource Sharing origins. |

---

## 2. Configuration Profiles

### 2.1 Development / Local Workstation Profile
Create a `.env` file in the project root:
```ini
FASTAPI_HOST=127.0.0.1
FASTAPI_PORT=8000
STREAMLIT_PORT=8501
LOG_LEVEL=DEBUG
DEVICE=cpu
DATABASE_URL=sqlite:///data/maintenance_history.db
```

### 2.2 Production Industrial Node Profile
For deployment on a dedicated plant reliability server:
```ini
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000
STREAMLIT_PORT=8501
LOG_LEVEL=INFO
DEVICE=cuda
DATABASE_URL=sqlite:///data/maintenance_history.db
CORS_ORIGINS=http://192.168.1.50:8501,http://mes.plant.internal
```

---

## 3. Model & Artifact Paths Configuration

All default file paths are defined in `src/models/registry.py` and can be overridden programmatically:

```python
MODEL_PATHS = {
    "nasa_rul_xgboost": "models/baseline/nasa_rul_xgboost_regressor.joblib",
    "nasa_rul_bilstm_w60": "models/deep_learning/bilstm_rul_w60.pt",
    "nasa_rul_lstm_w30": "models/deep_learning/lstm_rul_w30.pt",
    "ai4i_failure_gbdt": "models/baseline/ai4i_gradient_boosting_classifier.joblib",
    "ai4i_failure_xgboost": "models/baseline/ai4i_xgboost_classifier.joblib",
    "nasa_lstm_autoencoder": "models/anomaly/nasa_lstm_autoencoder.pt",
    "ai4i_isolation_forest": "models/anomaly/ai4i_isolation_forest.joblib",
    "nasa_minmax_scaler": "models/scalers/nasa_minmax_scaler.joblib",
    "ai4i_standard_scaler": "models/scalers/ai4i_standard_scaler.joblib"
}
```

---

## 4. Priority Classification & Threshold Rules

Priority classification thresholds can be adjusted in `src/recommendations/priority_engine.py`:

```python
# Industrial Decision Thresholds (ISO 13381-1 Aligned)
PRIORITY_THRESHOLDS = {
    "RUL_CRITICAL": 20.0,      # Cycles remaining for emergency stand-down
    "RUL_HIGH": 50.0,          # Cycles remaining for scheduled 48-hour service
    "RUL_MEDIUM": 80.0,        # Cycles remaining for next-shift inspection
    "FAILURE_PROB_CRITICAL": 0.75, # Machine failure probability triggering shutdown
    "FAILURE_PROB_HIGH": 0.40,     # Machine failure probability triggering inspection
    "FAILURE_PROB_MEDIUM": 0.20,   # Machine failure probability triggering monitoring
    "ANOMALY_SEVERITY_TIERS": {
        "NOMINAL": 0.00,
        "LOW": 0.50,
        "MEDIUM": 0.65,
        "HIGH": 0.80
    }
}
```
