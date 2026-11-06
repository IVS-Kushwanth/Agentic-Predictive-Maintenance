"""
Permutation feature importance module for Phase 5.
Evaluates model performance degradation upon feature shuffling using validation data.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance


class PermutationExplainer:
    """
    Computes permutation feature importance on validation datasets and compares against SHAP.
    """

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        task_type: str = "regression",
        model_id: str = "model"
    ):
        self.model = model
        self.feature_names = feature_names
        self.task_type = task_type
        self.model_id = model_id

    def compute_importance(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_repeats: int = 5,
        scoring: Optional[str] = None,
        random_state: int = 42
    ) -> List[Dict[str, Any]]:
        """
        Calculates permutation importance and returns sorted feature list with mean and std.
        """
        arr_x = np.asarray(X_val, dtype=np.float32)
        arr_y = np.asarray(y_val, dtype=np.float32)

        if scoring is None:
            scoring = "r2" if self.task_type == "regression" else "roc_auc"

        result = permutation_importance(
            self.model,
            arr_x,
            arr_y,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1
        )

        mean_scores = result.importances_mean
        std_scores = result.importances_std

        n_feats = min(len(self.feature_names), len(mean_scores))
        ranked_indices = np.argsort(mean_scores[:n_feats])[::-1]

        features_ranked = []
        for rank, idx in enumerate(ranked_indices, start=1):
            features_ranked.append({
                "feature": self.feature_names[idx],
                "importance_mean": round(float(mean_scores[idx]), 6),
                "importance_std": round(float(std_scores[idx]), 6),
                "rank": rank
            })

        return features_ranked

    @staticmethod
    def compare_with_shap(
        perm_results: List[Dict[str, Any]],
        shap_results: List[Dict[str, Any]],
        top_k: int = 15
    ) -> Dict[str, Any]:
        """
        Compares permutation importance against SHAP global importance:
        computes Spearman rank correlation and concordance across top-k features.
        """
        # Map feature to rank in each method
        perm_map = {item["feature"]: item["rank"] for item in perm_results}
        shap_map = {item["feature"]: item["rank"] for item in shap_results}

        common_features = [f for f in shap_map if f in perm_map]
        if not common_features:
            return {"spearman_correlation": 0.0, "common_features_count": 0}

        perm_ranks = [perm_map[f] for f in common_features]
        shap_ranks = [shap_map[f] for f in common_features]

        corr, p_value = spearmanr(perm_ranks, shap_ranks)

        top_shap_feats = set([item["feature"] for item in shap_results[:top_k]])
        top_perm_feats = set([item["feature"] for item in perm_results[:top_k]])
        intersection = top_shap_feats.intersection(top_perm_feats)
        jaccard_overlap = len(intersection) / max(1, len(top_shap_feats.union(top_perm_feats)))

        return {
            "spearman_rank_correlation": round(float(corr), 4),
            "p_value": round(float(p_value), 6),
            "top_k_evaluated": top_k,
            "concordant_top_features": list(intersection),
            "concordance_jaccard_index": round(float(jaccard_overlap), 4)
        }
