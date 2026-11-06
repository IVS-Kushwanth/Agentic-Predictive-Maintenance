"""
Evaluation module for Phase 4 Multivariate Sensor Anomaly Detection.
Provides post-hoc performance metrics against failure labels, reconstruction error statistics,
and threshold calibration diagnostics.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix
)


def evaluate_against_labels(
    y_true: np.ndarray,
    anomaly_scores: np.ndarray,
    threshold: float
) -> Dict[str, Any]:
    """
    Post-hoc evaluation comparing continuous anomaly scores and binary decisions against true labels.
    Calculates Precision, Recall, F1, PR-AUC, ROC-AUC, and Confusion Matrix.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = (anomaly_scores >= threshold).astype(int)

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        pr_auc = float(average_precision_score(y_true, anomaly_scores))
    except Exception:
        pr_auc = 0.0

    try:
        roc_auc = float(roc_auc_score(y_true, anomaly_scores))
    except Exception:
        roc_auc = 0.5

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "threshold": float(threshold),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },
        "total_samples": int(len(y_true)),
        "true_positives_rate": round(recall, 4),
        "false_positive_rate": round(float(fp) / max(1, (fp + tn)), 4)
    }


def compute_score_distribution(scores: np.ndarray) -> Dict[str, float]:
    """
    Computes statistical distribution metrics for continuous anomaly scores.
    """
    scores = np.asarray(scores, dtype=float)
    return {
        "count": int(len(scores)),
        "mean": round(float(np.mean(scores)), 6),
        "std": round(float(np.std(scores)), 6),
        "min": round(float(np.min(scores)), 6),
        "p25": round(float(np.percentile(scores, 25)), 6),
        "p50_median": round(float(np.median(scores)), 6),
        "p75": round(float(np.percentile(scores, 75)), 6),
        "p90": round(float(np.percentile(scores, 90)), 6),
        "p95": round(float(np.percentile(scores, 95)), 6),
        "p97_5": round(float(np.percentile(scores, 97.5)), 6),
        "p99": round(float(np.percentile(scores, 99)), 6),
        "max": round(float(np.max(scores)), 6)
    }
