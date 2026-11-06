"""
Maintenance Recommendation Agent:
- Synthesizes prediction, anomaly severity, and XAI sensor attribution
- Produces structured, prioritized maintenance action items using RecommendationEngine
- Enforces safety guardrails, confidence bounds, and human engineering sign-off
- Persists all decision-support recommendations to SQLite database for audit and review tracking
"""

import os
import json
from typing import Dict, Any, List, Optional
from src.graph.state import MaintenanceState
from src.recommendations import RecommendationEngine, RecommendationInput
from src.database import RecommendationHistoryDB


class MaintenanceAgent:
    """
    Dedicated agent for prescriptive maintenance decision support.
    Integrates RecommendationEngine and RecommendationHistoryDB.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.disclaimer = (
            "INDUSTRIAL ADVISORY: The recommendations provided herein are decision-support suggestions "
            "derived from statistical machine-learning and anomaly models. They do NOT constitute autonomous "
            "machine-control commands. Certified human maintenance engineering review and authorization "
            "are strictly required prior to physical equipment intervention."
        )
        self.engine = RecommendationEngine()
        self.db = RecommendationHistoryDB(db_path=db_path)

    def run(self, state: MaintenanceState) -> MaintenanceState:
        new_state = dict(state)
        if new_state.get("workflow_status") == "FAILED":
            return new_state

        ds_type = new_state.get("dataset_type", "NASA")
        rul = new_state.get("rul_estimate")
        prob = new_state.get("failure_risk_score")
        severity = new_state.get("anomaly_severity", "Normal")
        anomaly_score = new_state.get("anomaly_score")
        top_sensors = new_state.get("top_contributing_sensors", [])
        machine_id = new_state.get("machine_id", "UNIT")
        ew_status = new_state.get("early_warning_status", {})

        # Prepare payload for RecommendationEngine
        payload = RecommendationInput(
            dataset_type=ds_type,
            machine_id=machine_id,
            predicted_rul=rul,
            failure_risk_score=prob,
            anomaly_severity=severity,
            anomaly_score=anomaly_score,
            important_sensors=top_sensors,
            model_confidence=new_state.get("confidence_score", 0.85),
            early_warning_horizons=ew_status
        )

        # Generate recommendation
        rec_output = self.engine.generate_recommendation(payload)

        # Persist to database
        rec_id = self.db.save_recommendation(rec_output)
        rec_output["recommendation_id"] = rec_id

        # Determine urgency label for synthesis brief
        urgency = "ROUTINE"
        if rec_output["priority"] in ["Critical review required", "High priority"]:
            urgency = "CRITICAL"
        elif rec_output["priority"] == "Medium priority":
            urgency = "ELEVATED"
        elif rec_output["priority"] == "Low priority":
            urgency = "ADVISORY"

        # Generate Executive Summary Text
        summary_text = self._synthesize_summary(
            machine_id=machine_id,
            dataset_type=ds_type,
            rul=rul,
            prob=prob,
            severity=severity,
            urgency=urgency,
            top_sensors=top_sensors,
            actions=rec_output["recommended_actions"]
        )

        new_state["recommendation_id"] = rec_id
        new_state["recommendation_output"] = rec_output
        new_state["priority"] = rec_output["priority"]
        new_state["risk_status"] = rec_output["risk_status"]
        new_state["recommended_maintenance_actions"] = rec_output["recommended_actions"]
        new_state["human_review_required"] = rec_output["human_approval_required"]
        new_state["confidence_score"] = rec_output.get("confidence_score") or 0.85
        new_state["disclaimer"] = rec_output.get("disclaimer", self.disclaimer)
        new_state["llm_summary"] = summary_text
        new_state["evidence_references"] = {
            "prediction_evidence": new_state.get("prediction_results", {}),
            "anomaly_evidence": {
                "score": anomaly_score,
                "severity": severity,
                "status": new_state.get("anomaly_status")
            },
            "attribution_evidence": [s["feature"] for s in top_sensors[:3]] if top_sensors else [],
            "supporting_evidence": rec_output.get("supporting_evidence", [])
        }
        new_state["workflow_status"] = "COMPLETED"

        return new_state

    def _synthesize_summary(
        self,
        machine_id: str,
        dataset_type: str,
        rul: Optional[float],
        prob: Optional[float],
        severity: str,
        urgency: str,
        top_sensors: List[Dict[str, Any]],
        actions: List[Dict[str, Any]]
    ) -> str:
        """
        Synthesizes model evidence into a clear, professional engineering decision brief.
        """
        top_feature_str = top_sensors[0]["feature"] if top_sensors else "monitored sensors"
        action_desc = actions[0].get("description") or actions[0].get("action_description", "Standard monitoring") if actions else "Standard monitoring"
        action_urgency = actions[0].get("urgency", "Routine") if actions else "Routine"

        if dataset_type == "NASA":
            rul_str = f"{rul:.1f} cycles" if rul is not None else "N/A"
            brief = (
                f"ENGINEERING DECISION SUPPORT BRIEF — {machine_id} (NASA Turbofan Fleet)\n"
                f"• Status: {urgency} | Anomaly Severity: {severity} | Estimated Remaining Life: {rul_str}\n"
                f"• Primary Physical Degradation Driver: '{top_feature_str}'\n"
                f"• Recommended Protocol: {action_desc}\n"
                f"• Action Urgency: {action_urgency}"
            )
        else:
            prob_str = f"{prob*100:.1f}%" if prob is not None else "N/A"
            brief = (
                f"ENGINEERING DECISION SUPPORT BRIEF — {machine_id} (CNC Manufacturing)\n"
                f"• Status: {urgency} | Anomaly Severity: {severity} | Failure Probability: {prob_str}\n"
                f"• Primary Breakdown Influence: '{top_feature_str}'\n"
                f"• Recommended Protocol: {action_desc}\n"
                f"• Action Urgency: {action_urgency}"
            )

        return brief
