"""
Shared typed state definitions for LangGraph Agent Orchestration.
Maintains context across supervisor, prediction, anomaly, explanation, and maintenance agents.
"""

from typing import Dict, Any, List, Optional, Union, Literal
from typing_extensions import TypedDict


class MaintenanceState(TypedDict, total=False):
    """
    Standardized shared state schema for predictive maintenance orchestration.
    """
    # Request metadata
    request_id: str
    dataset_type: Literal["NASA", "AI4I"]
    machine_id: str
    workflow_type: str  # "full_analysis", "risk_only", "rul_only", "anomaly_only", "explanation_only"

    # Raw input data
    input_sensor_data: Any

    # Agent outputs
    prediction_results: Dict[str, Any]
    rul_estimate: Optional[float]
    early_warning_status: Optional[Dict[str, float]]
    failure_risk_score: Optional[float]
    failure_label: Optional[str]

    anomaly_score: Optional[float]
    anomaly_status: Optional[bool]
    anomaly_severity: Optional[str]
    anomaly_metadata: Optional[Dict[str, Any]]

    explanation_results: Optional[Dict[str, Any]]
    top_contributing_sensors: Optional[List[Dict[str, Any]]]
    temporal_attribution: Optional[Dict[str, Any]]

    recommended_maintenance_actions: Optional[List[Dict[str, Any]]]
    recommendation_id: Optional[str]
    priority: Optional[str]
    risk_status: Optional[str]
    recommendation_output: Optional[Dict[str, Any]]
    llm_summary: Optional[str]

    # Control flow & safety
    workflow_status: str  # "INITIALIZED", "VALIDATED", "PREDICTED", "ANOMALY_EVALUATED", "EXPLAINED", "COMPLETED", "FAILED"
    error_information: List[str]
    human_review_required: bool
    confidence_score: float
    evidence_references: Dict[str, Any]
    disclaimer: str
