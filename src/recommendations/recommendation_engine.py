"""
Intelligent Maintenance Recommendation Engine:
Translates model predictions, anomaly scores, and explainable feature attributions
into structured, prioritized, and evidence-grounded maintenance suggestions.
"""

from typing import Dict, Any, Optional, Union
from .recommendation_schema import RecommendationInput, format_recommendation_output
from .priority_engine import PriorityEngine
from .recommendation_rules import RecommendationRules


class RecommendationEngine:
    """
    Main decision-support engine.
    Derives actionable maintenance protocols without fabricating values or universal thresholds.
    """

    def __init__(
        self,
        priority_engine: Optional[PriorityEngine] = None,
        recommendation_rules: Optional[RecommendationRules] = None
    ):
        self.priority_engine = priority_engine or PriorityEngine()
        self.rules = recommendation_rules or RecommendationRules()

    def generate_recommendation(
        self,
        input_data: Union[RecommendationInput, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Processes validated input and generates structured decision-support output.
        """
        if isinstance(input_data, dict):
            payload = RecommendationInput(**input_data)
        elif isinstance(input_data, RecommendationInput):
            payload = input_data
        else:
            raise TypeError(f"Expected RecommendationInput or dict, got {type(input_data)}")

        # 1. Determine Priority, Risk Status, and Human Review
        priority, risk_status, reasons, human_approval = self.priority_engine.evaluate_priority(payload)

        # 2. Generate Evidence-Linked Actions
        actions, evidence = self.rules.generate_recommendations(
            payload=payload,
            priority=priority,
            risk_status=risk_status,
            reasons=reasons
        )

        # 3. Format Structured Output
        output = format_recommendation_output(
            machine_id=payload.machine_id,
            dataset=payload.dataset_type,
            priority=priority,
            risk_status=risk_status,
            reason=reasons,
            recommended_actions=actions,
            supporting_evidence=evidence,
            human_approval_required=human_approval,
            predicted_rul=payload.predicted_rul,
            failure_risk_score=payload.failure_risk_score,
            confidence=payload.model_confidence
        )

        return output
