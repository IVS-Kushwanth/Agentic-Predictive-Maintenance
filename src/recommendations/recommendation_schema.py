"""
Standardized schemas for Phase 7 Intelligent Maintenance Recommendation Engine.
Defines input validation models and structured output contracts.
"""

from typing import Dict, Any, List, Optional, Literal, Union
from datetime import datetime, timezone
import numpy as np


class RecommendationInput:
    """
    Standardized input payload validator for the recommendation engine.
    Accommodates dataset-specific differences (NASA vs AI4I) without fabricating missing values.
    """

    VALID_DATASETS = ["NASA", "AI4I"]
    VALID_SEVERITIES = ["Normal", "Low anomaly", "Medium anomaly", "High anomaly", "Unknown"]

    def __init__(
        self,
        dataset_type: str,
        machine_id: str,
        predicted_rul: Optional[float] = None,
        failure_risk_score: Optional[float] = None,
        anomaly_severity: str = "Normal",
        anomaly_score: Optional[float] = None,
        important_sensors: Optional[List[Dict[str, Any]]] = None,
        operating_conditions: Optional[Dict[str, Any]] = None,
        model_confidence: Optional[float] = None,
        early_warning_horizons: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.dataset_type = str(dataset_type).upper()
        self.machine_id = str(machine_id)
        self.predicted_rul = float(predicted_rul) if predicted_rul is not None else None
        self.failure_risk_score = float(failure_risk_score) if failure_risk_score is not None else None
        self.anomaly_severity = anomaly_severity if anomaly_severity in self.VALID_SEVERITIES else "Unknown"
        self.anomaly_score = float(anomaly_score) if anomaly_score is not None else None
        self.important_sensors = important_sensors or []
        self.operating_conditions = operating_conditions or {}
        self.model_confidence = float(model_confidence) if model_confidence is not None else None
        self.early_warning_horizons = early_warning_horizons or {}
        self.metadata = metadata or {}

        self.validate()

    def validate(self) -> None:
        """Validate input consistency without assuming universal fields."""
        if self.dataset_type not in self.VALID_DATASETS:
            raise ValueError(f"Invalid dataset_type: '{self.dataset_type}'. Must be one of {self.VALID_DATASETS}.")

        if not self.machine_id or not self.machine_id.strip():
            raise ValueError("machine_id must be a non-empty string.")

        # Check ranges if provided
        if self.predicted_rul is not None and self.predicted_rul < 0.0:
            raise ValueError(f"predicted_rul cannot be negative: {self.predicted_rul}")

        if self.failure_risk_score is not None and not (0.0 <= self.failure_risk_score <= 1.0):
            raise ValueError(f"failure_risk_score must be between 0.0 and 1.0: {self.failure_risk_score}")

        if self.model_confidence is not None and not (0.0 <= self.model_confidence <= 1.0):
            raise ValueError(f"model_confidence must be between 0.0 and 1.0: {self.model_confidence}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_type": self.dataset_type,
            "machine_id": self.machine_id,
            "predicted_rul": self.predicted_rul,
            "failure_risk_score": self.failure_risk_score,
            "anomaly_severity": self.anomaly_severity,
            "anomaly_score": self.anomaly_score,
            "important_sensors": self.important_sensors,
            "operating_conditions": self.operating_conditions,
            "model_confidence": self.model_confidence,
            "early_warning_horizons": self.early_warning_horizons,
            "metadata": self.metadata
        }


def format_recommendation_output(
    machine_id: str,
    dataset: str,
    priority: str,
    risk_status: str,
    reason: List[str],
    recommended_actions: List[Dict[str, Any]],
    supporting_evidence: List[Dict[str, Any]],
    human_approval_required: bool,
    predicted_rul: Optional[float] = None,
    failure_risk_score: Optional[float] = None,
    confidence: Optional[float] = None
) -> Dict[str, Any]:
    """
    Standardized recommendation output schema conforming to Prompt Requirement 6:
    {
        "machine_id": "example",
        "priority": "medium",
        "predicted_rul": null,
        "risk_status": "requires_review",
        "reason": [],
        "recommended_actions": [],
        "supporting_evidence": [],
        "human_approval_required": true
    }
    """
    disclaimer = (
        "INDUSTRIAL ADVISORY: The recommendations provided herein are decision-support suggestions "
        "derived from statistical machine-learning and anomaly models. They do NOT constitute autonomous "
        "machine-control commands. Certified human maintenance engineering review and authorization "
        "are strictly required prior to physical equipment intervention."
    )

    return {
        "machine_id": machine_id,
        "dataset": dataset,
        "priority": priority,
        "predicted_rul": round(predicted_rul, 2) if predicted_rul is not None else None,
        "failure_risk_score": round(failure_risk_score, 4) if failure_risk_score is not None else None,
        "confidence_score": round(confidence, 4) if confidence is not None else None,
        "risk_status": risk_status,
        "reason": reason,
        "recommended_actions": recommended_actions,
        "supporting_evidence": supporting_evidence,
        "human_approval_required": human_approval_required,
        "disclaimer": disclaimer,
        "generated_at": datetime.now(timezone.utc).isoformat()
    }
