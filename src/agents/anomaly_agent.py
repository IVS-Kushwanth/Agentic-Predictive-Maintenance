"""
Anomaly Detection Agent:
- Calls Phase 4 anomaly detection models (LSTM Autoencoder, Isolation Forest)
- Retrieves continuous anomaly score and multi-tier calibrated severity
- Extracts anomalous sensor patterns
"""

from typing import Dict, Any
from src.graph.state import MaintenanceState
from src.services.model_service import ModelService


class AnomalyAgent:
    """
    Dedicated agent for unsupervised anomaly detection.
    """

    def __init__(self, model_service: ModelService = None):
        self.model_service = model_service or ModelService()

    def run(self, state: MaintenanceState) -> MaintenanceState:
        new_state = dict(state)
        if new_state.get("workflow_status") == "FAILED":
            return new_state

        ds_type = new_state.get("dataset_type")
        data = new_state.get("input_sensor_data")

        try:
            anomaly_res = self.model_service.detect_anomalies(data, ds_type)
            new_state["anomaly_score"] = anomaly_res.get("anomaly_score")
            new_state["anomaly_status"] = anomaly_res.get("is_anomaly")
            new_state["anomaly_severity"] = anomaly_res.get("severity")
            new_state["anomaly_metadata"] = anomaly_res.get("relevant_sensor_metadata", {})
            new_state["workflow_status"] = "ANOMALY_EVALUATED"
        except Exception as e:
            errors = list(new_state.get("error_information", []))
            errors.append(f"AnomalyAgent execution failed: {str(e)}")
            new_state["error_information"] = errors
            # Allow graceful continuation if anomaly model fails
            new_state["anomaly_severity"] = "Unknown"

        return new_state
