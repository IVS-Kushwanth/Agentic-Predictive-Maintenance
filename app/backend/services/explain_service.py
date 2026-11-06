"""
Explainability Service for Phase 8 FastAPI Backend.
Executes TreeSHAP and temporal attributions to explain machine predictions.
"""

from typing import Dict, Any, List, Optional, Literal
import numpy as np
from .model_cache import get_explainability_manager


class ExplainService:
    """Encapsulates model explainability operations."""

    def __init__(self):
        self.explainer_mgr = get_explainability_manager()

    def explain(
        self,
        machine_id: str,
        dataset_type: Literal["NASA", "AI4I"],
        features: List[float],
        method: Optional[str] = "shap",
        top_k: int = 5
    ) -> Dict[str, Any]:
        arr = np.asarray(features, dtype=np.float32)

        if dataset_type == "NASA":
            if arr.ndim == 1 and len(arr) != 153:
                raise ValueError(f"NASA explanation requires 153 features, received {len(arr)}")
            model_id = "nasa_rul_xgboost_regressor_FD001"
            dataset_code = "FD001"
        else:
            if len(arr) != 14:
                raise ValueError(f"AI4I explanation requires 14 features, received {len(arr)}")
            model_id = "ai4i_clf_xgboost_classifier"
            dataset_code = "AI4I"

        explainer = self.explainer_mgr.get_tree_explainer(model_id=model_id, dataset=dataset_code)
        explanation = explainer.explain_instance(arr)

        top_contribs = []
        raw_contribs = explanation.get("important_features") or explanation.get("top_contributing_sensors", [])
        for c in raw_contribs[:top_k]:
            top_contribs.append({
                "feature": str(c.get("feature", "unknown")),
                "importance": float(c.get("importance", 0.0)),
                "contribution": "positive" if c.get("contribution") == "positive" else "negative",
                "value": float(c.get("value")) if c.get("value") is not None else None
            })

        summary = (
            f"Prediction primarily driven by '{top_contribs[0]['feature']}' "
            f"with importance weight {top_contribs[0]['importance']:.4f}."
            if top_contribs else "Attribution profile within standard parameters."
        )

        return {
            "machine_id": machine_id,
            "dataset": dataset_type,
            "method": method or "shap",
            "top_contributing_sensors": top_contribs,
            "explanation_summary": summary
        }
