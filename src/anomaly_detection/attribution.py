"""
Sensor-level abnormality attribution module for Phase 4.
Identifies and ranks which sensors/features contribute most significantly to detected anomalies.
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np


class SensorAttributor:
    """
    Analyzes which sensor channels deviate most strongly from nominal patterns.
    Supports Autoencoder reconstruction errors and statistical residual analysis.
    """

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names

    def attribute_reconstruction_error(
        self,
        feature_errors_1d: np.ndarray,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Takes 1D array of per-feature errors for a single sample/sequence.
        Returns top-k ranked sensors and contribution percentages.
        """
        total_error = float(np.sum(feature_errors_1d))
        if total_error <= 1e-9:
            total_error = 1e-9

        # Map indices to names
        n_features = min(len(self.feature_names), len(feature_errors_1d))
        indexed_errors = [
            (self.feature_names[i], float(feature_errors_1d[i]))
            for i in range(n_features)
        ]

        # Sort descending by error magnitude
        ranked = sorted(indexed_errors, key=lambda x: x[1], reverse=True)

        top_sensors = [
            {
                "sensor": name,
                "error": round(err, 6),
                "contribution_pct": round(100.0 * (err / total_error), 2)
            }
            for name, err in ranked[:top_k]
        ]

        all_errors = {name: round(err, 6) for name, err in indexed_errors}

        return {
            "top_contributing_sensors": top_sensors,
            "all_sensor_errors": all_errors,
            "total_reconstruction_error": round(total_error, 6)
        }

    def attribute_statistical_deviation(
        self,
        sample_1d: np.ndarray,
        nominal_mean: np.ndarray,
        nominal_std: np.ndarray,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Z-score based statistical attribution for traditional models.
        """
        std_safe = np.where(nominal_std < 1e-6, 1.0, nominal_std)
        z_scores = np.abs((sample_1d - nominal_mean) / std_safe)
        total_z = float(np.sum(z_scores))
        if total_z <= 1e-9:
            total_z = 1e-9

        n_features = min(len(self.feature_names), len(z_scores))
        indexed_deviations = [
            (self.feature_names[i], float(z_scores[i]))
            for i in range(n_features)
        ]
        ranked = sorted(indexed_deviations, key=lambda x: x[1], reverse=True)

        top_sensors = [
            {
                "sensor": name,
                "z_score": round(z, 4),
                "contribution_pct": round(100.0 * (z / total_z), 2)
            }
            for name, z in ranked[:top_k]
        ]

        return {
            "top_contributing_sensors": top_sensors,
            "all_z_scores": {name: round(z, 4) for name, z in indexed_deviations},
            "mean_z_score": round(float(np.mean(z_scores)), 4)
        }
