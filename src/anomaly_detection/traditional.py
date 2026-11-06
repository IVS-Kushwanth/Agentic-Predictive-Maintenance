"""
Traditional anomaly detection algorithms for Phase 4:
- Isolation Forest
- One-Class SVM
- Local Outlier Factor (Novelty detection mode)

Supports both 2D tabular inputs and 3D multivariate temporal sequence windows.
"""

import os
import json
import joblib
from typing import Dict, Any, Optional, Union, List
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

from .base import BaseAnomalyDetector


def _prepare_temporal_data(X: np.ndarray) -> np.ndarray:
    """
    If X is 3D (samples, window_length, features), aggregate temporal statistics:
    mean and standard deviation across the temporal window to produce a robust 2D feature matrix.
    If X is 2D/1D with 153 features (static slice), pad with zero variance to match the 306-dim temporal profile.
    If X is already 2D (samples, features), return as is.
    """
    if X.ndim == 3:
        # X shape: (N, L, D)
        mean_feat = np.mean(X, axis=1)
        std_feat = np.std(X, axis=1)
        # Concatenate mean and std -> (N, 2*D)
        return np.concatenate([mean_feat, std_feat], axis=1)
    elif X.ndim == 2:
        if X.shape[1] == 153:
            return np.concatenate([X, np.zeros_like(X)], axis=1)
        return X
    elif X.ndim == 1:
        arr_2d = X.reshape(1, -1)
        if arr_2d.shape[1] == 153:
            return np.concatenate([arr_2d, np.zeros_like(arr_2d)], axis=1)
        return arr_2d
    else:
        raise ValueError(f"Unsupported input dimensions: {X.ndim}. Expected 2D or 3D.")


class IsolationForestDetector(BaseAnomalyDetector):
    """
    Isolation Forest anomaly detector.
    Computes anomaly scores where higher score indicates greater abnormality.
    """

    def __init__(
        self,
        model_id: str = "isolation_forest",
        dataset_name: str = "FD001",
        n_estimators: int = 100,
        max_samples: Union[int, float] = "auto",
        contamination: Union[float, str] = "auto",
        random_state: int = 42
    ):
        super().__init__(model_id=model_id, model_type="IsolationForest", dataset_name=dataset_name)
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )

    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None, **kwargs) -> "IsolationForestDetector":
        X_2d = _prepare_temporal_data(X)
        self.model.fit(X_2d)
        self.is_fitted = True
        if feature_names:
            self.feature_names = feature_names
        elif X.ndim == 3 and hasattr(X, "shape"):
            self.feature_names = [f"feat_{i}" for i in range(X.shape[2])]
        elif X.ndim == 2:
            self.feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"{self.model_id} must be fitted before scoring.")
        X_2d = _prepare_temporal_data(X)
        # decision_function returns negative for outliers, positive for inliers.
        # We invert so higher score = more anomalous.
        return -self.model.decision_function(X_2d)

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        artifact = {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "dataset_name": self.dataset_name,
            "thresholds": self.thresholds,
            "feature_names": self.feature_names,
            "model": self.model,
            "metadata": self.metadata
        }
        joblib.dump(artifact, filepath)

    def load(self, filepath: str) -> "IsolationForestDetector":
        artifact = joblib.load(filepath)
        self.model_id = artifact["model_id"]
        self.model_type = artifact["model_type"]
        self.dataset_name = artifact["dataset_name"]
        self.thresholds = artifact["thresholds"]
        self.feature_names = artifact["feature_names"]
        self.model = artifact["model"]
        self.metadata = artifact.get("metadata", {})
        self.is_fitted = True
        return self


class OneClassSVMDetector(BaseAnomalyDetector):
    """
    One-Class Support Vector Machine for boundary-based anomaly detection.
    Computes distance to support boundary where higher score = more anomalous.
    """

    def __init__(
        self,
        model_id: str = "one_class_svm",
        dataset_name: str = "FD001",
        kernel: str = "rbf",
        gamma: str = "scale",
        nu: float = 0.05
    ):
        super().__init__(model_id=model_id, model_type="OneClassSVM", dataset_name=dataset_name)
        self.kernel = kernel
        self.gamma = gamma
        self.nu = nu

        self.model = OneClassSVM(
            kernel=self.kernel,
            gamma=self.gamma,
            nu=self.nu
        )

    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None, **kwargs) -> "OneClassSVMDetector":
        X_2d = _prepare_temporal_data(X)
        self.model.fit(X_2d)
        self.is_fitted = True
        if feature_names:
            self.feature_names = feature_names
        elif X.ndim == 3 and hasattr(X, "shape"):
            self.feature_names = [f"feat_{i}" for i in range(X.shape[2])]
        elif X.ndim == 2:
            self.feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"{self.model_id} must be fitted before scoring.")
        X_2d = _prepare_temporal_data(X)
        # decision_function returns signed distance: negative for outliers, positive for inliers.
        # Invert so higher score = more anomalous.
        return -self.model.decision_function(X_2d)

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        artifact = {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "dataset_name": self.dataset_name,
            "thresholds": self.thresholds,
            "feature_names": self.feature_names,
            "model": self.model,
            "metadata": self.metadata
        }
        joblib.dump(artifact, filepath)

    def load(self, filepath: str) -> "OneClassSVMDetector":
        artifact = joblib.load(filepath)
        self.model_id = artifact["model_id"]
        self.model_type = artifact["model_type"]
        self.dataset_name = artifact["dataset_name"]
        self.thresholds = artifact["thresholds"]
        self.feature_names = artifact["feature_names"]
        self.model = artifact["model"]
        self.metadata = artifact.get("metadata", {})
        self.is_fitted = True
        return self


class LOFDetector(BaseAnomalyDetector):
    """
    Local Outlier Factor (Novelty mode) for density-based anomaly detection.
    Computes local density ratio where higher score = more anomalous.
    """

    def __init__(
        self,
        model_id: str = "lof",
        dataset_name: str = "FD001",
        n_neighbors: int = 20,
        contamination: Union[float, str] = "auto"
    ):
        super().__init__(model_id=model_id, model_type="LocalOutlierFactor", dataset_name=dataset_name)
        self.n_neighbors = n_neighbors
        self.contamination = contamination

        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=True,
            n_jobs=-1
        )

    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None, **kwargs) -> "LOFDetector":
        X_2d = _prepare_temporal_data(X)
        self.model.fit(X_2d)
        self.is_fitted = True
        if feature_names:
            self.feature_names = feature_names
        elif X.ndim == 3 and hasattr(X, "shape"):
            self.feature_names = [f"feat_{i}" for i in range(X.shape[2])]
        elif X.ndim == 2:
            self.feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"{self.model_id} must be fitted before scoring.")
        X_2d = _prepare_temporal_data(X)
        # score_samples returns negative LOF score. Invert so higher score = more anomalous.
        return -self.model.score_samples(X_2d)

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        artifact = {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "dataset_name": self.dataset_name,
            "thresholds": self.thresholds,
            "feature_names": self.feature_names,
            "model": self.model,
            "metadata": self.metadata
        }
        joblib.dump(artifact, filepath)

    def load(self, filepath: str) -> "LOFDetector":
        artifact = joblib.load(filepath)
        self.model_id = artifact["model_id"]
        self.model_type = artifact["model_type"]
        self.dataset_name = artifact["dataset_name"]
        self.thresholds = artifact["thresholds"]
        self.feature_names = artifact["feature_names"]
        self.model = artifact["model"]
        self.metadata = artifact.get("metadata", {})
        self.is_fitted = True
        return self
