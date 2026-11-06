"""
Unified Explainability API for Phase 5.
Exposes standardized functions callable by future LangGraph agents and decision-support engines:
- explain_prediction()
- get_global_feature_importance()
- get_temporal_explanation()
- get_sensor_contributions()
"""

import os
import sys
import json
import joblib
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
import torch

from .shap_explainer import TreeSHAPExplainer
from .deep_explainer import DeepTemporalExplainer
from .schema import format_explanation_response

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class ExplainabilityManager:
    """
    Manages loading and caching of explainers across traditional ML and deep learning models.
    """

    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = models_dir or os.path.join(BASE_DIR, "models")
        self._tree_explainers: Dict[str, TreeSHAPExplainer] = {}
        self._deep_explainers: Dict[str, DeepTemporalExplainer] = {}
        self._feature_names_cache: Dict[str, List[str]] = {}

    def get_feature_names(self, dataset: str) -> List[str]:
        if dataset in self._feature_names_cache:
            return self._feature_names_cache[dataset]

        if dataset.startswith("FD"):
            meta_path = os.path.join(self.models_dir, "baseline", f"nasa_rul_xgboost_regressor_{dataset}_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                names = meta["features"]
            else:
                npz = np.load(os.path.join(BASE_DIR, "data", "processed", "nasa", dataset, "sequences_w30_val.npz"))
                names = list(npz["feature_names"])
        else:
            meta_path = os.path.join(self.models_dir, "baseline", "ai4i_clf_xgboost_classifier_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                names = meta["features"]
            else:
                meta_path_raw = os.path.join(BASE_DIR, "data", "processed", "ai4i", "metadata.json")
                with open(meta_path_raw, "r") as f:
                    meta = json.load(f)
                names = meta.get("feature_columns", [f"feat_{i}" for i in range(14)])

        self._feature_names_cache[dataset] = names
        return names

    def get_tree_explainer(self, model_id: str, dataset: str = "FD001") -> TreeSHAPExplainer:
        key = f"{model_id}_{dataset}"
        if key in self._tree_explainers:
            return self._tree_explainers[key]

        features = self.get_feature_names(dataset)

        if dataset.startswith("FD"):
            if "rf" in model_id.lower() or "random_forest" in model_id.lower():
                path = os.path.join(self.models_dir, "baseline", f"nasa_rul_random_forest_regressor_{dataset}.joblib")
            else:
                path = os.path.join(self.models_dir, "baseline", f"nasa_rul_xgboost_regressor_{dataset}.joblib")
            task_type = "regression"
        else:
            if "rf" in model_id.lower() or "random_forest" in model_id.lower():
                path = os.path.join(self.models_dir, "baseline", "ai4i_clf_random_forest_classifier.joblib")
            else:
                path = os.path.join(self.models_dir, "baseline", "ai4i_clf_xgboost_classifier.joblib")
            task_type = "classification"

        model = joblib.load(path)
        explainer = TreeSHAPExplainer(model, features, task_type=task_type, model_id=model_id)
        self._tree_explainers[key] = explainer
        return explainer

    def get_deep_explainer(self, model_id: str = "lstm", dataset: str = "FD001") -> DeepTemporalExplainer:
        key = f"{model_id}_{dataset}"
        if key in self._deep_explainers:
            return self._deep_explainers[key]

        from src.deep_learning.lstm_model import LSTMRULModel
        from src.deep_learning.gru_model import GRURULModel

        features = self.get_feature_names(dataset)

        if "gru" in model_id.lower():
            pt_path = os.path.join(self.models_dir, "deep_learning", "gru", f"nasa_rul_gru_{dataset}_w30.pt")
            model = GRURULModel(input_dim=len(features), hidden_dim=32, num_layers=1, fc_dim=32)
        else:
            pt_path = os.path.join(self.models_dir, "deep_learning", "lstm", f"nasa_rul_lstm_{dataset}_w30.pt")
            model = LSTMRULModel(input_dim=len(features), hidden_dim=32, num_layers=1, fc_dim=32)

        ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt)
        explainer = DeepTemporalExplainer(model, features, model_id=model_id)
        self._deep_explainers[key] = explainer
        return explainer


# Singleton manager instance
_MANAGER = None


def get_manager() -> ExplainabilityManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = ExplainabilityManager()
    return _MANAGER


def explain_prediction(
    input_data: Union[np.ndarray, List, torch.Tensor],
    model_id: Optional[str] = None,
    dataset: Optional[str] = None,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Explains an individual prediction for any traditional ML or deep learning model.
    """
    mgr = get_manager()
    arr = np.asarray(input_data, dtype=np.float32)

    # Infer dataset
    if dataset is None:
        dataset = "AI4I" if (arr.ndim == 1 and arr.shape[0] == 14) or (arr.ndim == 2 and arr.shape[-1] == 14) else "FD001"

    # Infer model_id
    if model_id is None:
        model_id = "lstm" if arr.ndim == 3 or (arr.ndim == 2 and arr.shape[0] == 30 and arr.shape[1] > 14) else "xgboost"

    if "lstm" in model_id.lower() or "gru" in model_id.lower():
        deep_exp = mgr.get_deep_explainer(model_id, dataset)
        return deep_exp.explain_sequence(arr, top_k_sensors=top_k)
    else:
        tree_exp = mgr.get_tree_explainer(model_id, dataset)
        return tree_exp.explain_instance(arr, top_k=top_k)


def get_global_feature_importance(
    model_id: str = "xgboost",
    dataset: str = "FD001",
    top_k: int = 15,
    sample_size: int = 200
) -> List[Dict[str, Any]]:
    """
    Returns global mean absolute SHAP importance for top_k features.
    """
    mgr = get_manager()
    tree_exp = mgr.get_tree_explainer(model_id, dataset)
    features = mgr.get_feature_names(dataset)

    if dataset.startswith("FD"):
        val_df = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "nasa", dataset, "val_tabular.csv"), nrows=sample_size)
        X_eval = val_df[features].to_numpy(dtype=np.float32)
    else:
        X_eval = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "ai4i", "X_test.csv"), nrows=sample_size).to_numpy(dtype=np.float32)

    return tree_exp.get_global_importance(X_eval, top_k=top_k)


def get_temporal_explanation(
    sequence_data: Union[np.ndarray, torch.Tensor],
    model_id: str = "lstm",
    dataset: str = "FD001"
) -> Dict[str, Any]:
    """
    Returns temporal attribution weights and peak influence cycles for sequence models.
    """
    mgr = get_manager()
    deep_exp = mgr.get_deep_explainer(model_id, dataset)
    exp = deep_exp.explain_sequence(sequence_data)
    return exp.get("temporal_attribution", {})


def get_sensor_contributions(
    input_data: Union[np.ndarray, List, torch.Tensor],
    model_id: Optional[str] = None,
    dataset: Optional[str] = None,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Returns ranked sensor contributions (importance, direction, raw value).
    """
    exp = explain_prediction(input_data, model_id=model_id, dataset=dataset, top_k=top_k)
    return exp.get("important_features", [])
