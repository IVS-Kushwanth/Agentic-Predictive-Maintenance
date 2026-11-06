"""
Model Service: Intermediary layer executing predictions and anomaly detection
using actual saved artifacts from Phases 2, 3, and 4.
"""

import os
import sys
import json
import joblib
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.anomaly_detection import predict_anomaly


class ModelService:
    """
    Executes actual deterministic model inference for NASA and AI4I pipelines.
    """

    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = models_dir or os.path.join(BASE_DIR, "models")
        self._cached_models: Dict[str, Any] = {}
        self._cached_features: Dict[str, List[str]] = {}

    def get_features(self, dataset_type: str) -> List[str]:
        if dataset_type in self._cached_features:
            return self._cached_features[dataset_type]

        if dataset_type == "NASA":
            meta_path = os.path.join(self.models_dir, "baseline", "nasa_rul_xgboost_regressor_FD001_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                features = meta["features"]
            else:
                features = [f"sensor_{i}" for i in range(153)]
        else:
            meta_path = os.path.join(self.models_dir, "baseline", "ai4i_clf_xgboost_classifier_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                features = meta["features"]
            else:
                features = [f"feat_{i}" for i in range(14)]

        self._cached_features[dataset_type] = features
        return features

    def predict_nasa_rul(self, input_data: Any) -> Dict[str, Any]:
        """
        Executes NASA RUL prediction using XGBoost regressor (and optional early warning classifier).
        """
        arr = np.asarray(input_data, dtype=np.float32)
        features = self.get_features("NASA")

        # If 3D sequence (1, 30, 153), take the final cycle observation or summary
        if arr.ndim == 3:
            sample_1d = arr[0, -1, :]  # Terminal cycle in window
        elif arr.ndim == 2:
            sample_1d = arr[-1, :] if arr.shape[0] == 30 else arr[0, :]
        else:
            sample_1d = arr.ravel()

        if len(sample_1d) != len(features):
            raise ValueError(f"NASA model requires {len(features)} features, received {len(sample_1d)}")

        # Load XGBoost Regressor
        key = "nasa_xgb_reg"
        if key not in self._cached_models:
            path = os.path.join(self.models_dir, "baseline", "nasa_rul_xgboost_regressor_FD001.joblib")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model checkpoint not found: {path}")
            self._cached_models[key] = joblib.load(path)

        model = self._cached_models[key]
        pred_rul = float(model.predict(sample_1d.reshape(1, -1))[0])

        # Load Early Warning Classifiers (10, 20, 30 cycles)
        ew_status = {}
        for h in [10, 20, 30]:
            ew_key = f"nasa_ew_{h}"
            if ew_key not in self._cached_models:
                ew_path = os.path.join(self.models_dir, "baseline", f"nasa_ew_{h}_xgboost_classifier_FD001.joblib")
                if os.path.exists(ew_path):
                    self._cached_models[ew_key] = joblib.load(ew_path)
            if ew_key in self._cached_models:
                ew_model = self._cached_models[ew_key]
                prob = float(ew_model.predict_proba(sample_1d.reshape(1, -1))[0, 1])
                ew_status[f"risk_within_{h}_cycles"] = round(prob, 4)

        return {
            "predicted_rul": round(pred_rul, 2),
            "units": "operational_cycles",
            "model_id": "nasa_rul_xgboost_regressor_FD001",
            "early_warning_horizons": ew_status
        }

    def predict_ai4i_failure(self, input_data: Any) -> Dict[str, Any]:
        """
        Executes AI4I machine failure classification using XGBoost classifier.
        """
        arr = np.asarray(input_data, dtype=np.float32).ravel()
        features = self.get_features("AI4I")

        if len(arr) != len(features):
            raise ValueError(f"AI4I model requires {len(features)} features, received {len(arr)}")

        key = "ai4i_xgb_clf"
        if key not in self._cached_models:
            path = os.path.join(self.models_dir, "baseline", "ai4i_clf_xgboost_classifier.joblib")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model checkpoint not found: {path}")
            self._cached_models[key] = joblib.load(path)

        model = self._cached_models[key]
        sample_2d = arr.reshape(1, -1)
        prob = float(model.predict_proba(sample_2d)[0, 1])

        # Calibrated optimal decision threshold from Phase 2
        threshold = 0.7969
        label = "Machine Failure" if prob >= threshold else "Normal Operation"

        return {
            "failure_probability": round(prob, 4),
            "decision_threshold": threshold,
            "failure_label": label,
            "model_id": "ai4i_clf_xgboost_classifier"
        }

    def detect_anomalies(self, input_data: Any, dataset_type: str) -> Dict[str, Any]:
        """
        Executes Phase 4 unsupervised anomaly detection.
        """
        ds_code = "FD001" if dataset_type == "NASA" else "AI4I"
        model_id = "lstm_autoencoder" if dataset_type == "NASA" and np.asarray(input_data).ndim >= 2 else "isolation_forest"

        res = predict_anomaly(input_data, model_id=model_id, dataset=ds_code)
        return res
