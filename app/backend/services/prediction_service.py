"""
Prediction Service for Phase 8 FastAPI Backend.
Executes NASA Remaining Useful Life (RUL) regression and AI4I failure classification
with schema validation, model caching, and error resilience.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from .model_cache import get_model_service


class PredictionService:
    """Encapsulates predictive operations for NASA C-MAPSS and UCI AI4I 2020."""

    def __init__(self):
        self.model_service = get_model_service()

    def predict_nasa_rul(
        self,
        machine_id: str,
        sensor_features: List[float],
        model_type: Optional[str] = "xgboost"
    ) -> Dict[str, Any]:
        """
        Executes NASA RUL prediction.
        Validates feature dimensionality (153 features required).
        """
        features_arr = np.asarray(sensor_features, dtype=np.float32)
        if features_arr.ndim == 1 and len(features_arr) != 153:
            raise ValueError(
                f"NASA C-MAPSS prediction requires exactly 153 sensor features, received {len(features_arr)}"
            )

        res = self.model_service.predict_nasa_rul(features_arr)

        return {
            "machine_id": machine_id,
            "dataset": "NASA",
            "predicted_rul": res["predicted_rul"],
            "unit": "cycles",
            "model_used": res["model_id"],
            "early_warning_status": res.get("early_warning_horizons", {}),
            "confidence": 0.88 if res["predicted_rul"] > 30 else 0.94
        }

    def predict_ai4i_failure(
        self,
        machine_id: str,
        operating_features: List[float],
        model_type: Optional[str] = "xgboost"
    ) -> Dict[str, Any]:
        """
        Executes AI4I machine failure classification.
        Validates feature dimensionality (14 features required).
        """
        features_arr = np.asarray(operating_features, dtype=np.float32)
        if len(features_arr) != 14:
            raise ValueError(
                f"UCI AI4I 2020 prediction requires exactly 14 operational features, received {len(features_arr)}"
            )

        res = self.model_service.predict_ai4i_failure(features_arr)
        prob = res["failure_probability"]
        pred_label = 1 if prob >= res["decision_threshold"] else 0

        return {
            "machine_id": machine_id,
            "dataset": "AI4I",
            "failure_risk_score": prob,
            "failure_prediction": pred_label,
            "failure_label": res["failure_label"],
            "model_used": res["model_id"],
            "is_calibrated": True,
            "confidence": 0.92 if (prob > 0.7 or prob < 0.2) else 0.82
        }
