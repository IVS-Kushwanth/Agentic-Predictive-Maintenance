"""
Unified inference interface for Phase 4 Multivariate Sensor Anomaly Detection.
Provides a standardized function:
    predict_anomaly(input_data, model_id=..., dataset=...)
returning anomaly score, boolean status, severity tier, model identifier, and sensor metadata.
"""

import os
import json
from typing import Dict, Any, Optional, Union, List
import numpy as np
import torch

from .traditional import IsolationForestDetector, OneClassSVMDetector, LOFDetector
from .autoencoder import LSTMAutoencoderDetector, DenseAutoencoderDetector
from .severity import SeverityClassifier
from .attribution import SensorAttributor


class AnomalyInferenceEngine:
    """
    Manages loading and running saved anomaly detectors with transparent caching.
    """

    def __init__(self, models_dir: Optional[str] = None):
        if models_dir is None:
            # Default to project root models/anomaly_detection
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            self.models_dir = os.path.join(base_dir, "models", "anomaly_detection")
        else:
            self.models_dir = models_dir

        self._loaded_models: Dict[str, Any] = {}
        self._attributors: Dict[str, SensorAttributor] = {}
        self._classifiers: Dict[str, SeverityClassifier] = {}

    def _get_model_key(self, model_id: str, dataset: str) -> str:
        return f"{model_id}_{dataset}"

    def load_model(self, model_id: str, dataset: str = "FD001"):
        key = self._get_model_key(model_id, dataset)
        if key in self._loaded_models:
            return self._loaded_models[key]

        # Determine file path based on model_id
        if "lstm" in model_id.lower() or "autoencoder" in model_id.lower() and dataset.startswith("FD"):
            path = os.path.join(self.models_dir, "lstm_autoencoder", f"lstm_autoencoder_{dataset}.pt")
            if not os.path.exists(path):
                # Try generic name
                path = os.path.join(self.models_dir, "lstm_autoencoder", "lstm_autoencoder.pt")
            detector = LSTMAutoencoderDetector(model_id=model_id, dataset_name=dataset)
            detector.load(path)
        elif "dense" in model_id.lower() or ("autoencoder" in model_id.lower() and dataset == "AI4I"):
            path = os.path.join(self.models_dir, "autoencoder", f"dense_autoencoder_{dataset}.pt")
            if not os.path.exists(path):
                path = os.path.join(self.models_dir, "autoencoder", "dense_autoencoder.pt")
            detector = DenseAutoencoderDetector(model_id=model_id, dataset_name=dataset)
            detector.load(path)
        elif "isolation" in model_id.lower() or "forest" in model_id.lower() or "iforest" in model_id.lower():
            path = os.path.join(self.models_dir, "isolation_forest", f"isolation_forest_{dataset}.joblib")
            detector = IsolationForestDetector(model_id=model_id, dataset_name=dataset)
            detector.load(path)
        elif "svm" in model_id.lower():
            path = os.path.join(self.models_dir, "one_class_svm", f"one_class_svm_{dataset}.joblib")
            detector = OneClassSVMDetector(model_id=model_id, dataset_name=dataset)
            detector.load(path)
        elif "lof" in model_id.lower():
            path = os.path.join(self.models_dir, "lof", f"lof_{dataset}.joblib")
            detector = LOFDetector(model_id=model_id, dataset_name=dataset)
            detector.load(path)
        else:
            raise ValueError(f"Unknown anomaly detection model_id: '{model_id}'")

        self._loaded_models[key] = detector
        self._classifiers[key] = SeverityClassifier.from_thresholds(detector.thresholds)
        self._attributors[key] = SensorAttributor(detector.feature_names)
        return detector

    def predict(
        self,
        input_data: Union[np.ndarray, List, torch.Tensor],
        model_id: Optional[str] = None,
        dataset: Optional[str] = None,
        threshold_level: str = "medium",
        custom_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Unified prediction logic.
        """
        # Format input as numpy array
        if isinstance(input_data, torch.Tensor):
            arr = input_data.detach().cpu().numpy()
        else:
            arr = np.asarray(input_data, dtype=np.float32)

        # Infer dataset & model if not provided
        if dataset is None:
            dataset = "AI4I" if (arr.ndim == 2 and arr.shape[-1] == 14) or (arr.ndim == 1 and arr.shape[0] == 14) else "FD001"

        if model_id is None:
            if dataset.startswith("FD"):
                model_id = "lstm_autoencoder"
            else:
                model_id = "isolation_forest"

        detector = self.load_model(model_id, dataset)
        key = self._get_model_key(model_id, dataset)
        classifier = self._classifiers[key]
        attributor = self._attributors[key]

        # Ensure correct batch dimensions
        is_single_sample = False
        if detector.model_type == "LSTMAutoencoder":
            if arr.ndim == 2:
                # Shape (seq_len, features) -> expand to (1, seq_len, features)
                arr = np.expand_dims(arr, axis=0)
                is_single_sample = True
            elif arr.ndim == 3 and arr.shape[0] == 1:
                is_single_sample = True
        else:
            if arr.ndim == 1:
                # Shape (features,) -> expand to (1, features)
                arr = np.expand_dims(arr, axis=0)
                is_single_sample = True
            elif (arr.ndim == 2 or arr.ndim == 3) and arr.shape[0] == 1:
                is_single_sample = True

        scores = detector.score(arr)
        threshold = custom_threshold if custom_threshold is not None else detector.thresholds.get(threshold_level, 0.0)

        # Compute sensor attribution
        sensor_meta = {}
        if hasattr(detector, "feature_errors"):
            feat_errors = detector.feature_errors(arr)
            # Use first sample for attribution
            sample_feat_err = feat_errors[0]
            sensor_meta = attributor.attribute_reconstruction_error(sample_feat_err, top_k=5)
        else:
            # Traditional statistical residual attribution if nominal stats exist
            if "nominal_mean" in detector.metadata and "nominal_std" in detector.metadata:
                mean = np.array(detector.metadata["nominal_mean"])
                std = np.array(detector.metadata["nominal_std"])
                sample_1d = arr[0] if arr.ndim == 2 else np.mean(arr[0], axis=0)
                sensor_meta = attributor.attribute_statistical_deviation(sample_1d, mean, std, top_k=5)
            else:
                sensor_meta = {
                    "top_contributing_sensors": [],
                    "message": "Direct sensor attribution available via Autoencoder models"
                }

        sensor_meta["total_features_evaluated"] = len(detector.feature_names)

        if is_single_sample:
            score_val = float(scores[0])
            severity_str = classifier.classify_score(score_val)
            is_anomaly_bool = bool(score_val >= threshold)

            return {
                "anomaly_score": round(score_val, 6),
                "is_anomaly": is_anomaly_bool,
                "severity": severity_str,
                "severity_code": classifier.get_severity_code(severity_str),
                "model_identifier": detector.model_id,
                "model_type": detector.model_type,
                "dataset": detector.dataset_name,
                "threshold_applied": round(float(threshold), 6),
                "relevant_sensor_metadata": sensor_meta
            }
        else:
            # Batch mode
            severities = classifier.classify_batch(scores)
            is_anomalies = [bool(s >= threshold) for s in scores]

            return {
                "anomaly_scores": [round(float(s), 6) for s in scores],
                "is_anomaly": is_anomalies,
                "severity": severities,
                "model_identifier": detector.model_id,
                "model_type": detector.model_type,
                "dataset": detector.dataset_name,
                "threshold_applied": round(float(threshold), 6),
                "batch_size": len(scores),
                "relevant_sensor_metadata": sensor_meta
            }


# Global engine instance for simple import and execution
_DEFAULT_ENGINE = None


def get_inference_engine() -> AnomalyInferenceEngine:
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = AnomalyInferenceEngine()
    return _DEFAULT_ENGINE


def predict_anomaly(
    input_data: Union[np.ndarray, List, torch.Tensor],
    model_id: Optional[str] = None,
    dataset: Optional[str] = None,
    threshold_level: str = "medium",
    custom_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Standard Phase 4 public API function.

    Example:
        res = predict_anomaly(test_sequence, model_id='lstm_autoencoder')
        print(res['anomaly_score'], res['is_anomaly'], res['severity'])
    """
    engine = get_inference_engine()
    return engine.predict(
        input_data=input_data,
        model_id=model_id,
        dataset=dataset,
        threshold_level=threshold_level,
        custom_threshold=custom_threshold
    )
