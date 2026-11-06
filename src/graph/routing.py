"""
Routing logic and conditional edges for LangGraph predictive maintenance orchestration.
"""

from typing import Literal
from src.graph.state import MaintenanceState


def route_supervisor_decision(state: MaintenanceState) -> Literal["prediction_agent", "result_validation"]:
    """
    Routes from supervisor:
    If validation passed, proceed to prediction_agent;
    If validation failed, skip to result_validation.
    """
    if state.get("workflow_status") == "FAILED":
        return "result_validation"
    return "prediction_agent"


def route_after_prediction(state: MaintenanceState) -> Literal["anomaly_agent", "result_validation"]:
    """
    Routes from prediction_agent:
    If prediction succeeded, proceed to anomaly_agent;
    If failed, skip to result_validation.
    """
    if state.get("workflow_status") == "FAILED":
        return "result_validation"
    return "anomaly_agent"


def route_after_anomaly(state: MaintenanceState) -> Literal["explanation_agent", "result_validation"]:
    """
    Routes from anomaly_agent to explanation_agent.
    """
    if state.get("workflow_status") == "FAILED":
        return "result_validation"
    return "explanation_agent"


def route_after_explanation(state: MaintenanceState) -> Literal["maintenance_agent", "result_validation"]:
    """
    Routes from explanation_agent to maintenance_agent.
    """
    if state.get("workflow_status") == "FAILED":
        return "result_validation"
    return "maintenance_agent"
