"""
Phase 4: Multivariate Sensor Anomaly Detection Package.
Provides traditional and deep learning anomaly detection models, multi-tier severity classification,
sensor-level attribution, and the standard predict_anomaly inference interface.
"""

from .base import BaseAnomalyDetector
from .traditional import IsolationForestDetector, OneClassSVMDetector, LOFDetector
from .autoencoder import LSTMAutoencoderDetector, DenseAutoencoderDetector
from .severity import SeverityClassifier
from .attribution import SensorAttributor
from .evaluation import evaluate_against_labels, compute_score_distribution
from .inference import predict_anomaly, AnomalyInferenceEngine, get_inference_engine

__all__ = [
    "BaseAnomalyDetector",
    "IsolationForestDetector",
    "OneClassSVMDetector",
    "LOFDetector",
    "LSTMAutoencoderDetector",
    "DenseAutoencoderDetector",
    "SeverityClassifier",
    "SensorAttributor",
    "evaluate_against_labels",
    "compute_score_distribution",
    "predict_anomaly",
    "AnomalyInferenceEngine",
    "get_inference_engine"
]
