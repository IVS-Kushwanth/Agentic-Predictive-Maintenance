"""Evaluation Metrics Module for Predictive Maintenance.

Implements standard and domain-specific evaluation metrics:
1. NASA C-MAPSS Asymmetric Scoring Function (PHM 2008 benchmark):
   Penalizes late predictions (overestimating RUL) more heavily than early predictions.
   d_i = y_pred_i - y_true_i
   s_i = exp(-d_i / 13) - 1   if d_i < 0 (early prediction / underestimation)
   s_i = exp(d_i / 10) - 1    if d_i >= 0 (late prediction / overestimation)
   Score = sum(s_i)

2. Standard Regression Metrics: MAE, RMSE, R2, Median Absolute Error.

3. Classification Metrics:
   Precision, Recall, F1-Score, PR-AUC (Average Precision), ROC-AUC, Confusion Matrix.
   Optimal decision threshold finder based on F1-score on validation predictions.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def nasa_asymmetric_scoring_function(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute official NASA C-MAPSS asymmetric scoring function.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth remaining useful life.
    y_pred : np.ndarray
        Predicted remaining useful life.
        
    Returns
    -------
    float
        Total asymmetric NASA score.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    d = y_pred - y_true
    
    # Early prediction (d < 0): exp(-d/13) - 1
    # Late prediction (d >= 0): exp(d/10) - 1
    scores = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(scores))


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = ""
) -> Dict[str, float]:
    """Compute comprehensive regression metrics including NASA asymmetric score."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    med_ae = float(np.median(np.abs(y_true - y_pred)))
    nasa_score = nasa_asymmetric_scoring_function(y_true, y_pred)

    pfx = f"{prefix}_" if prefix else ""
    return {
        f"{pfx}mae": round(mae, 4),
        f"{pfx}rmse": round(rmse, 4),
        f"{pfx}r2": round(r2, 4),
        f"{pfx}med_ae": round(med_ae, 4),
        f"{pfx}nasa_score": round(nasa_score, 2),
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1"
) -> Tuple[float, float]:
    """Find decision threshold that maximizes validation F1 score or balanced metric.
    
    Returns
    -------
    best_threshold : float
    best_metric_value : float
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    
    # Avoid division by zero
    numerator = 2 * precisions * recalls
    denominator = precisions + recalls
    f1_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    
    if len(thresholds) == 0:
        return 0.5, 0.0

    best_idx = np.argmax(f1_scores[:-1])
    best_th = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])
    return best_th, round(best_f1, 4)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    prefix: str = ""
) -> Dict[str, Any]:
    """Compute full classification metrics under given decision threshold."""
    y_true = np.asarray(y_true, dtype=np.int32)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(np.int32)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    # PR-AUC and ROC-AUC
    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = 0.0

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.0

    cm = confusion_matrix(y_true, y_pred).tolist()

    pfx = f"{prefix}_" if prefix else ""
    return {
        f"{pfx}threshold": round(threshold, 4),
        f"{pfx}accuracy": round(acc, 4),
        f"{pfx}precision": round(prec, 4),
        f"{pfx}recall": round(rec, 4),
        f"{pfx}f1": round(f1, 4),
        f"{pfx}pr_auc": round(pr_auc, 4),
        f"{pfx}roc_auc": round(roc_auc, 4),
        f"{pfx}confusion_matrix": cm,
    }
