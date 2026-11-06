"""
Prediction Agent:
- Executes trained NASA RUL regression and early-warning classification
- Executes trained AI4I machine-failure classification
- Populates prediction results and confidence metrics
"""

from typing import Dict, Any
from src.graph.state import MaintenanceState
from src.services.model_service import ModelService


class PredictionAgent:
    """
    Dedicated agent for deterministic supervised predictions.
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
            if ds_type == "NASA":
                pred_res = self.model_service.predict_nasa_rul(data)
                new_state["prediction_results"] = pred_res
                new_state["rul_estimate"] = pred_res.get("predicted_rul")
                new_state["early_warning_status"] = pred_res.get("early_warning_horizons", {})
                new_state["workflow_status"] = "PREDICTED"

            elif ds_type == "AI4I":
                pred_res = self.model_service.predict_ai4i_failure(data)
                new_state["prediction_results"] = pred_res
                new_state["failure_risk_score"] = pred_res.get("failure_probability")
                new_state["failure_label"] = pred_res.get("failure_label")
                new_state["workflow_status"] = "PREDICTED"
        except Exception as e:
            errors = list(new_state.get("error_information", []))
            errors.append(f"PredictionAgent execution failed: {str(e)}")
            new_state["error_information"] = errors
            new_state["workflow_status"] = "FAILED"

        return new_state
