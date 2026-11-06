"""
Explanation Service: Intermediary layer retrieving SHAP and Integrated Gradients
explanations using actual Phase 5 artifacts.
"""

import os
import sys
from typing import Dict, Any, Optional, List
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.explainability import (
    explain_prediction,
    get_temporal_explanation,
    get_sensor_contributions
)


class ExplanationService:
    """
    Coordinates explanation generation for traditional and deep learning models.
    """

    def explain_case(self, input_data: Any, dataset_type: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Retrieves actual feature attribution and evidence summary.
        """
        arr = np.asarray(input_data)
        ds_code = "FD001" if dataset_type == "NASA" else "AI4I"
        model_id = "lstm" if dataset_type == "NASA" and arr.ndim >= 2 else "xgboost"

        exp = explain_prediction(input_data, model_id=model_id, dataset=ds_code, top_k=top_k)
        return exp

    def get_temporal_profile(self, sequence_data: Any) -> Dict[str, Any]:
        """
        Retrieves temporal decay and peak influence cycle for sequence models.
        """
        return get_temporal_explanation(sequence_data, model_id="lstm", dataset="FD001")

    def get_key_drivers(self, input_data: Any, dataset_type: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Returns ranked list of influential sensor drivers.
        """
        ds_code = "FD001" if dataset_type == "NASA" else "AI4I"
        return get_sensor_contributions(input_data, model_id="xgboost", dataset=ds_code, top_k=top_k)
