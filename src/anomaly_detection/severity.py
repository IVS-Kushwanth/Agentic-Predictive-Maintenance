"""
Severity classification module for Phase 4 Multivariate Sensor Anomaly Detection.
Provides a multi-tier calibrated severity framework distinguishing anomaly deviation from failure probability.
"""

from typing import Dict, Any, Union, List
import numpy as np


class SeverityClassifier:
    """
    Classifies continuous anomaly scores into discrete operational severity levels:
    - Normal (Score < Low Threshold)
    - Low anomaly (Low Threshold <= Score < Medium Threshold)
    - Medium anomaly (Medium Threshold <= Score < High Threshold)
    - High anomaly (Score >= High Threshold)

    IMPORTANT: Anomaly severity measures the mathematical deviation of multivariate
    sensor readings from the learned nominal baseline. It must NOT be equated with
    actual probability of immediate machine breakdown, as benign operating regime
    shifts or sensor noise can produce low-to-medium deviations without structural failure.
    """

    SEVERITY_LEVELS = ["Normal", "Low anomaly", "Medium anomaly", "High anomaly"]
    SEVERITY_CODES = {"Normal": 0, "Low anomaly": 1, "Medium anomaly": 2, "High anomaly": 3}

    def __init__(self, low_th: float = 0.0, med_th: float = 0.0, high_th: float = 0.0):
        self.low_th = float(low_th)
        self.med_th = float(med_th)
        self.high_th = float(high_th)

    @classmethod
    def from_thresholds(cls, thresholds: Dict[str, float]) -> "SeverityClassifier":
        return cls(
            low_th=thresholds.get("low", 0.0),
            med_th=thresholds.get("medium", 0.0),
            high_th=thresholds.get("high", 0.0)
        )

    def calibrate(
        self,
        nominal_scores: np.ndarray,
        percentiles: tuple = (90.0, 97.5, 99.0)
    ) -> Dict[str, float]:
        """
        Calibrate severity thresholds from validation nominal scores.
        """
        self.low_th = float(np.percentile(nominal_scores, percentiles[0]))
        self.med_th = float(np.percentile(nominal_scores, percentiles[1]))
        self.high_th = float(np.percentile(nominal_scores, percentiles[2]))

        # Ensure monotonicity
        if self.med_th < self.low_th:
            self.med_th = self.low_th
        if self.high_th < self.med_th:
            self.high_th = self.med_th

        return self.get_thresholds()

    def get_thresholds(self) -> Dict[str, float]:
        return {
            "low": self.low_th,
            "medium": self.med_th,
            "high": self.high_th
        }

    def classify_score(self, score: float) -> str:
        """Classify a single score into a severity level string."""
        if score >= self.high_th:
            return "High anomaly"
        elif score >= self.med_th:
            return "Medium anomaly"
        elif score >= self.low_th:
            return "Low anomaly"
        else:
            return "Normal"

    def classify_batch(self, scores: Union[np.ndarray, List[float]]) -> List[str]:
        """Classify an array of scores into severity strings."""
        scores_arr = np.asarray(scores, dtype=float)
        results = []
        for s in scores_arr:
            results.append(self.classify_score(float(s)))
        return results

    def get_severity_code(self, severity_label: str) -> int:
        return self.SEVERITY_CODES.get(severity_label, 0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thresholds": self.get_thresholds(),
            "severity_levels": self.SEVERITY_LEVELS,
            "description": "Multi-tier calibrated quantile-based anomaly severity mapping"
        }
