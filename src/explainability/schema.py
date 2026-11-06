"""
Standardized schema and data structures for Phase 5 Explainable AI outputs.
Conforms to industrial decision-support and downstream agent consumption.
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np


def format_explanation_response(
    prediction_val: float,
    model_identifier: str,
    model_type: str,
    feature_names: List[str],
    contributions: Union[np.ndarray, List[float]],
    raw_values: Optional[Union[np.ndarray, List[float]]] = None,
    task_type: str = "regression",
    temporal_profile: Optional[Dict[str, Any]] = None,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Standardized explanation interface schema:
    {
        "prediction": {
            "predicted_rul": 24, # or "predicted_probability": 0.85
            "model_identifier": "xgboost_regressor_FD001",
            "model_type": "TreeEnsemble"
        },
        "important_features": [
            {
                "feature": "sensor_11_roll_mean_20",
                "contribution": "positive",
                "importance": 4.125,
                "raw_value": 0.542
            }
        ],
        "temporal_attribution": {...},
        "explanation_summary": "..."
    }
    """
    contribs = np.asarray(contributions, dtype=float).ravel()
    raws = np.asarray(raw_values, dtype=float).ravel() if raw_values is not None else np.zeros_like(contribs)

    n_feats = min(len(feature_names), len(contribs))
    indexed_features = []
    for i in range(n_feats):
        c_val = float(contribs[i])
        r_val = float(raws[i]) if i < len(raws) else 0.0
        indexed_features.append({
            "feature": feature_names[i],
            "contribution": "positive" if c_val >= 0 else "negative",
            "importance": round(c_val, 6),
            "abs_importance": abs(c_val),
            "raw_value": round(r_val, 6)
        })

    # Sort descending by absolute magnitude of importance
    ranked = sorted(indexed_features, key=lambda x: x["abs_importance"], reverse=True)
    top_features = [
        {
            "feature": item["feature"],
            "contribution": item["contribution"],
            "importance": item["importance"],
            "raw_value": item["raw_value"]
        }
        for item in ranked[:top_k]
    ]

    # Build prediction object
    if task_type == "regression":
        pred_dict = {
            "predicted_rul": round(float(prediction_val), 2),
            "model_identifier": model_identifier,
            "model_type": model_type,
            "units": "operating_cycles"
        }
        top_driver = top_features[0]["feature"] if top_features else "unknown"
        top_driver_dir = "extending" if top_features and top_features[0]["contribution"] == "positive" else "reducing"
        summary = (
            f"Model '{model_identifier}' predicts a Remaining Useful Life of {round(float(prediction_val), 1)} cycles. "
            f"The primary driver is '{top_driver}' with a contribution of {top_features[0]['importance']:+.2f} cycles, "
            f"{top_driver_dir} estimated machine operational lifespan."
        )
    else:
        prob = float(prediction_val)
        pred_dict = {
            "predicted_probability": round(prob, 4),
            "predicted_label": "Machine Failure" if prob >= 0.5 else "Normal Operation",
            "model_identifier": model_identifier,
            "model_type": model_type
        }
        status_str = "high risk of machine failure" if prob >= 0.5 else "nominal machine operating status"
        top_driver = top_features[0]["feature"] if top_features else "operating parameters"
        summary = (
            f"Model '{model_identifier}' estimates a {round(prob * 100, 1)}% failure probability ({status_str}). "
            f"The primary operating parameter driving this assessment is '{top_driver}' "
            f"(impact: {top_features[0]['importance']:+.3f})."
        )

    res = {
        "prediction": pred_dict,
        "important_features": top_features,
        "explanation_summary": summary
    }

    if temporal_profile is not None:
        res["temporal_attribution"] = temporal_profile

    return res
