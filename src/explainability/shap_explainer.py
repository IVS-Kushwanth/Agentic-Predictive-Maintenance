"""
SHAP (SHapley Additive exPlanations) module for Phase 5.
Implements TreeExplainer and LinearExplainer for traditional models (NASA RUL and AI4I failure classification).
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import shap

from .schema import format_explanation_response


class TreeSHAPExplainer:
    """
    SHAP TreeExplainer wrapper for tree-based models (XGBoost, Random Forest, Gradient Boosting).
    """

    def __init__(self, model: Any, feature_names: List[str], task_type: str = "regression", model_id: str = "tree_model"):
        self.model = model
        self.feature_names = feature_names
        self.task_type = task_type
        self.model_id = model_id

        # Initialize TreeExplainer
        self.explainer = shap.TreeExplainer(self.model)
        self.expected_value = self.explainer.expected_value
        if isinstance(self.expected_value, np.ndarray) and len(self.expected_value) > 1:
            # Multi-class or binary classification vector -> use positive class
            self.base_value = float(self.expected_value[1])
        elif isinstance(self.expected_value, (list, np.ndarray)):
            self.base_value = float(self.expected_value[0])
        else:
            self.base_value = float(self.expected_value)

    def explain_instance(self, x_sample: Union[np.ndarray, List[float]], top_k: int = 5) -> Dict[str, Any]:
        """
        Compute SHAP attribution for a single instance.
        """
        arr = np.asarray(x_sample, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        shap_vals = self.explainer.shap_values(arr)
        if isinstance(shap_vals, list):
            # Binary classification list [class 0, class 1]
            vals = shap_vals[1][0]
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            vals = shap_vals[0, :, 1]
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 2:
            vals = shap_vals[0]
        else:
            vals = np.asarray(shap_vals).ravel()

        # Compute prediction
        if self.task_type == "regression":
            pred = float(self.model.predict(arr)[0])
        else:
            if hasattr(self.model, "predict_proba"):
                pred = float(self.model.predict_proba(arr)[0, 1])
            else:
                pred = float(self.model.predict(arr)[0])

        return format_explanation_response(
            prediction_val=pred,
            model_identifier=self.model_id,
            model_type=self.model.__class__.__name__,
            feature_names=self.feature_names,
            contributions=vals,
            raw_values=arr[0],
            task_type=self.task_type,
            top_k=top_k
        )

    def compute_dataset_shap(self, X_dataset: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for an entire dataset.
        Returns array of shape (N, D).
        """
        arr = np.asarray(X_dataset, dtype=np.float32)
        shap_vals = self.explainer.shap_values(arr)
        if isinstance(shap_vals, list):
            # Positive class for classification
            return shap_vals[1]
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            return shap_vals[:, :, 1]
        else:
            return np.asarray(shap_vals)

    def get_global_importance(self, X_dataset: np.ndarray, top_k: int = 15) -> List[Dict[str, Any]]:
        """
        Compute mean absolute SHAP value across the dataset for global ranking.
        """
        shap_matrix = self.compute_dataset_shap(X_dataset)
        mean_abs = np.mean(np.abs(shap_matrix), axis=0)

        n_feats = min(len(self.feature_names), len(mean_abs))
        ranked_indices = np.argsort(mean_abs[:n_feats])[::-1]

        results = []
        for idx in ranked_indices[:top_k]:
            results.append({
                "feature": self.feature_names[idx],
                "mean_abs_shap": round(float(mean_abs[idx]), 6),
                "rank": len(results) + 1
            })
        return results
