"""
Anomaly Service for Phase 8 FastAPI Backend.
Executes unsupervised multivariate sensor anomaly detection using
LSTM Autoencoders, Dense Autoencoders, and Isolation Forests.
"""

from typing import Dict, Any, List, Optional, Literal
import numpy as np
from .model_cache import get_model_service


class AnomalyService:
    """Encapsulates unsupervised anomaly detection workflows."""

    def __init__(self):
        self.model_service = get_model_service()

    def detect_anomaly(
        self,
        machine_id: str,
        dataset_type: Literal["NASA", "AI4I"],
        features: List[float],
        detector_type: Optional[str] = "autoencoder"
    ) -> Dict[str, Any]:
        """
        Calculates reconstruction error or isolation score and determines severity.
        """
        arr = np.asarray(features, dtype=np.float32)

        if dataset_type == "NASA":
            if arr.ndim == 1 and len(arr) != 153:
                raise ValueError(f"NASA anomaly detection requires 153 features, received {len(arr)}")
        else:
            if len(arr) != 14:
                raise ValueError(f"AI4I anomaly detection requires 14 features, received {len(arr)}")

        res = self.model_service.detect_anomalies(arr, dataset_type=dataset_type)

        score = float(res.get("anomaly_score", 0.0))
        threshold = float(res.get("threshold", 0.037 if dataset_type == "NASA" else -0.05))
        severity = res.get("severity", "Normal")
        is_anom = bool(res.get("is_anomaly", severity != "Normal"))
        detector_used = res.get("model_id", detector_type or "autoencoder")

        return {
            "machine_id": machine_id,
            "dataset": dataset_type,
            "anomaly_score": round(score, 6),
            "threshold": round(threshold, 6),
            "is_anomaly": is_anom,
            "severity": severity,
            "detector_used": detector_used
        }
