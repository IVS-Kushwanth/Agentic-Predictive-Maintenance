"""
Phase 5: Explainable AI for Predictive Maintenance Package.
Provides SHAP tree explanations, permutation feature importance, deep learning temporal attribution
(Integrated Gradients), and the standardized explanation API for future LangGraph agents.
"""

from .schema import format_explanation_response
from .shap_explainer import TreeSHAPExplainer
from .permutation_explainer import PermutationExplainer
from .deep_explainer import DeepTemporalExplainer
from .inference_api import (
    explain_prediction,
    get_global_feature_importance,
    get_temporal_explanation,
    get_sensor_contributions,
    ExplainabilityManager,
    get_manager
)

__all__ = [
    "format_explanation_response",
    "TreeSHAPExplainer",
    "PermutationExplainer",
    "DeepTemporalExplainer",
    "explain_prediction",
    "get_global_feature_importance",
    "get_temporal_explanation",
    "get_sensor_contributions",
    "ExplainabilityManager",
    "get_manager"
]
