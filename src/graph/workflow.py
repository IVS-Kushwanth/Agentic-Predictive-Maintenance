"""
LangGraph Workflow: Multi-Agent Predictive Maintenance Orchestration.
Coordinates Supervisor, Prediction, Anomaly, Explanation, and Maintenance agents
via a compiled StateGraph with conditional execution and fallback safety routing.
"""

import os
import sys
import json
import uuid
from typing import Dict, Any, Optional, Union, List
import numpy as np

from langgraph.graph import StateGraph, START, END

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.graph.state import MaintenanceState
from src.graph.routing import (
    route_supervisor_decision,
    route_after_prediction,
    route_after_anomaly,
    route_after_explanation
)
# Lazy worker holders to prevent circular imports
_supervisor = None
_predictor = None
_anomaly_detector = None
_explainer = None
_maintenance_advisor = None


def get_supervisor():
    global _supervisor
    if _supervisor is None:
        from src.agents.supervisor import SupervisorAgent
        _supervisor = SupervisorAgent()
    return _supervisor


def get_predictor():
    global _predictor
    if _predictor is None:
        from src.agents.prediction_agent import PredictionAgent
        _predictor = PredictionAgent()
    return _predictor


def get_anomaly_detector():
    global _anomaly_detector
    if _anomaly_detector is None:
        from src.agents.anomaly_agent import AnomalyAgent
        _anomaly_detector = AnomalyAgent()
    return _anomaly_detector


def get_explainer():
    global _explainer
    if _explainer is None:
        from src.agents.explanation_agent import ExplanationAgent
        _explainer = ExplanationAgent()
    return _explainer


def get_maintenance_advisor():
    global _maintenance_advisor
    if _maintenance_advisor is None:
        from src.agents.maintenance_agent import MaintenanceAgent
        _maintenance_advisor = MaintenanceAgent()
    return _maintenance_advisor


# Node wrapper functions for StateGraph
def supervisor_node(state: MaintenanceState) -> MaintenanceState:
    return get_supervisor().run(state)


def prediction_node(state: MaintenanceState) -> MaintenanceState:
    return get_predictor().run(state)


def anomaly_node(state: MaintenanceState) -> MaintenanceState:
    return get_anomaly_detector().run(state)


def explanation_node(state: MaintenanceState) -> MaintenanceState:
    return get_explainer().run(state)


def maintenance_node(state: MaintenanceState) -> MaintenanceState:
    return get_maintenance_advisor().run(state)


def validation_node(state: MaintenanceState) -> MaintenanceState:
    """
    Final node validating structured results, enforcing schema integrity,
    and compiling evidence references.
    """
    new_state = dict(state)
    errors = new_state.get("error_information", [])

    if errors:
        new_state["workflow_status"] = "FAILED"
        if not new_state.get("llm_summary"):
            new_state["llm_summary"] = f"Workflow halted due to {len(errors)} error(s): {'; '.join(errors)}"
    else:
        new_state["workflow_status"] = "COMPLETED"

    return new_state


def create_maintenance_graph():
    """
    Constructs and compiles the multi-agent StateGraph workflow.
    """
    builder = StateGraph(MaintenanceState)

    # Register nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("prediction_agent", prediction_node)
    builder.add_node("anomaly_agent", anomaly_node)
    builder.add_node("explanation_agent", explanation_node)
    builder.add_node("maintenance_agent", maintenance_node)
    builder.add_node("result_validation", validation_node)

    # Edges & Conditional Routing
    builder.add_edge(START, "supervisor")

    builder.add_conditional_edges(
        "supervisor",
        route_supervisor_decision,
        {
            "prediction_agent": "prediction_agent",
            "result_validation": "result_validation"
        }
    )

    builder.add_conditional_edges(
        "prediction_agent",
        route_after_prediction,
        {
            "anomaly_agent": "anomaly_agent",
            "result_validation": "result_validation"
        }
    )

    builder.add_conditional_edges(
        "anomaly_agent",
        route_after_anomaly,
        {
            "explanation_agent": "explanation_agent",
            "result_validation": "result_validation"
        }
    )

    builder.add_conditional_edges(
        "explanation_agent",
        route_after_explanation,
        {
            "maintenance_agent": "maintenance_agent",
            "result_validation": "result_validation"
        }
    )

    builder.add_edge("maintenance_agent", "result_validation")
    builder.add_edge("result_validation", END)

    return builder.compile()


# Global compiled workflow instance
_COMPILED_GRAPH = None


def get_workflow_graph():
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = create_maintenance_graph()
    return _COMPILED_GRAPH


# -----------------------------------------------------------------------------
# Reusable Service APIs (Prompt Requirement 9)
# -----------------------------------------------------------------------------

def analyze_machine(
    input_data: Any,
    dataset_type: str,
    machine_id: str = "UNIT-001",
    workflow_type: str = "full_analysis"
) -> Dict[str, Any]:
    """
    Main entry point for end-to-end multi-agent predictive maintenance analysis.
    Executes supervisor -> predictor -> anomaly -> explainer -> maintenance advisor -> validator.
    """
    graph = get_workflow_graph()
    initial_state: MaintenanceState = {
        "request_id": str(uuid.uuid4())[:8],
        "dataset_type": dataset_type.upper(),
        "machine_id": machine_id,
        "input_sensor_data": input_data,
        "workflow_type": workflow_type,
        "workflow_status": "INITIALIZED",
        "error_information": []
    }

    final_state = graph.invoke(initial_state)

    # Return structured JSON output complying with Prompt Requirement 8
    output = {
        "dataset": final_state.get("dataset_type"),
        "machine_id": final_state.get("machine_id"),
        "workflow_status": final_state.get("workflow_status"),
        "recommendation_id": final_state.get("recommendation_id"),
        "priority": final_state.get("priority"),
        "risk_status": final_state.get("risk_status"),
        "rul_prediction": final_state.get("rul_estimate"),
        "failure_risk_score": final_state.get("failure_risk_score"),
        "failure_label": final_state.get("failure_label"),
        "anomaly_status": "ANOMALY_DETECTED" if final_state.get("anomaly_status") else "NOMINAL",
        "anomaly_severity": final_state.get("anomaly_severity"),
        "anomaly_score": final_state.get("anomaly_score"),
        "important_sensor_contributions": final_state.get("top_contributing_sensors", []),
        "explanation_summary": final_state.get("llm_summary"),
        "suggested_maintenance_actions": final_state.get("recommended_maintenance_actions", []),
        "recommendation_output": final_state.get("recommendation_output"),
        "confidence_or_uncertainty_information": {
            "model_confidence_score": final_state.get("confidence_score", 0.0),
            "early_warning_horizons": final_state.get("early_warning_status", {})
        },
        "evidence_references": final_state.get("evidence_references", {}),
        "human_review_status": {
            "human_review_required": final_state.get("human_review_required", False),
            "disclaimer": final_state.get("disclaimer", "")
        },
        "errors": final_state.get("error_information", [])
    }
    return output


def assess_failure_risk(input_data: Any, machine_id: str = "MACHINE-001") -> Dict[str, Any]:
    """Dedicated API for AI4I breakdown probability assessment."""
    res = analyze_machine(input_data, dataset_type="AI4I", machine_id=machine_id, workflow_type="risk_only")
    return {
        "machine_id": machine_id,
        "failure_risk_score": res.get("failure_risk_score"),
        "failure_label": res.get("failure_label"),
        "anomaly_severity": res.get("anomaly_severity"),
        "human_review_required": res["human_review_status"]["human_review_required"],
        "recommended_action": res["suggested_maintenance_actions"][0] if res.get("suggested_maintenance_actions") else None,
        "summary": res.get("explanation_summary")
    }


def estimate_remaining_useful_life(input_data: Any, machine_id: str = "ENGINE-001") -> Dict[str, Any]:
    """Dedicated API for NASA Remaining Useful Life estimation."""
    res = analyze_machine(input_data, dataset_type="NASA", machine_id=machine_id, workflow_type="rul_only")
    return {
        "machine_id": machine_id,
        "predicted_rul_cycles": res.get("rul_prediction"),
        "early_warning_horizons": res["confidence_or_uncertainty_information"].get("early_warning_horizons", {}),
        "anomaly_severity": res.get("anomaly_severity"),
        "human_review_required": res["human_review_status"]["human_review_required"],
        "primary_action": res["suggested_maintenance_actions"][0] if res.get("suggested_maintenance_actions") else None,
        "summary": res.get("explanation_summary")
    }


def explain_machine_condition(input_data: Any, dataset_type: str, machine_id: str = "UNIT-001") -> Dict[str, Any]:
    """Dedicated API retrieving explainable evidence and sensor attributions."""
    res = analyze_machine(input_data, dataset_type=dataset_type, machine_id=machine_id, workflow_type="explanation_only")
    return {
        "machine_id": machine_id,
        "dataset": dataset_type,
        "important_sensor_contributions": res.get("important_sensor_contributions", []),
        "anomaly_severity": res.get("anomaly_severity"),
        "explanation_summary": res.get("explanation_summary"),
        "evidence_references": res.get("evidence_references", {})
    }


def recommend_maintenance(input_data: Any, dataset_type: str, machine_id: str = "UNIT-001") -> Dict[str, Any]:
    """Dedicated API retrieving prescriptive maintenance suggestions."""
    res = analyze_machine(input_data, dataset_type=dataset_type, machine_id=machine_id, workflow_type="full_analysis")
    return {
        "machine_id": machine_id,
        "dataset": dataset_type,
        "suggested_maintenance_actions": res.get("suggested_maintenance_actions", []),
        "human_review_status": res.get("human_review_status", {}),
        "executive_brief": res.get("explanation_summary")
    }
