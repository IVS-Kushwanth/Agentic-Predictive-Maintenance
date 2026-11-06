"""
Explanation Agent:
- Interfaces with Phase 5 XAI module
- Computes actual SHAP values and Integrated Gradients temporal attributions
- Identifies critical physical features and sensor channels
"""

from typing import Dict, Any
from src.graph.state import MaintenanceState
from src.services.explanation_service import ExplanationService


class ExplanationAgent:
    """
    Dedicated agent for Explainable AI attribution.
    """

    def __init__(self, explanation_service: ExplanationService = None):
        self.explanation_service = explanation_service or ExplanationService()

    def run(self, state: MaintenanceState) -> MaintenanceState:
        new_state = dict(state)
        if new_state.get("workflow_status") == "FAILED":
            return new_state

        ds_type = new_state.get("dataset_type")
        data = new_state.get("input_sensor_data")

        try:
            exp_res = self.explanation_service.explain_case(data, ds_type, top_k=5)
            new_state["explanation_results"] = exp_res
            new_state["top_contributing_sensors"] = exp_res.get("important_features", [])
            new_state["temporal_attribution"] = exp_res.get("temporal_attribution", {})
            new_state["workflow_status"] = "EXPLAINED"
        except Exception as e:
            errors = list(new_state.get("error_information", []))
            errors.append(f"ExplanationAgent execution failed: {str(e)}")
            new_state["error_information"] = errors
            new_state["top_contributing_sensors"] = []

        return new_state
