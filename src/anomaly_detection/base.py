"""
Base anomaly detector interface for Phase 4 Multivariate Sensor Anomaly Detection.
Provides a standardized abstraction for traditional and deep learning anomaly detectors.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
import numpy as np


class BaseAnomalyDetector(ABC):
    """
    Abstract base class for all anomaly detectors in the framework.
    Standardizes scoring, threshold calibration, persistence, and metadata tracking.
    """

    def __init__(self, model_id: str, model_type: str, dataset_name: str):
        self.model_id = model_id
        self.model_type = model_type
        self.dataset_name = dataset_name
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.thresholds: Dict[str, float] = {
            "low": 0.0,
            "medium": 0.0,
            "high": 0.0
        }
        self.metadata: Dict[str, Any] = {}

    @abstractmethod
    def fit(self, X: np.ndarray, **kwargs) -> "BaseAnomalyDetector":
        """
        Fit the anomaly detector on nominal/training data.
        X may be 2D (samples, features) or 3D (samples, window_length, features).
        """
        pass

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute continuous anomaly scores for input data X.
        Higher score indicates higher degree of abnormality.
        Returns a 1D numpy array of shape (samples,).
        """
        pass

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """
        Predict binary anomaly status (1 for anomaly, 0 for normal).
        If threshold is not provided, uses self.thresholds['medium'].
        """
        if not self.is_fitted:
            raise RuntimeError(f"Model {self.model_id} must be fitted before prediction.")
        scores = self.score(X)
        th = threshold if threshold is not None else self.thresholds.get("medium", 0.0)
        return (scores >= th).astype(int)

    def calibrate_thresholds(
        self,
        nominal_scores: np.ndarray,
        percentiles: tuple = (90.0, 97.5, 99.0)
    ) -> Dict[str, float]:
        """
        Calibrate multi-tier thresholds on nominal/validation scores using percentiles.
        """
        low_th = float(np.percentile(nominal_scores, percentiles[0]))
        med_th = float(np.percentile(nominal_scores, percentiles[1]))
        high_th = float(np.percentile(nominal_scores, percentiles[2]))

        self.thresholds = {
            "low": low_th,
            "medium": med_th,
            "high": high_th
        }
        return self.thresholds

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Serialize detector artifacts to disk."""
        pass

    @abstractmethod
    def load(self, filepath: str) -> "BaseAnomalyDetector":
        """Load detector artifacts from disk."""
        pass
