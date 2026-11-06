"""Evaluation Engine for Deep Learning Predictive Maintenance Models.

Implements:
1. NASA C-MAPSS RUL Benchmark:
   - Official terminal-cycle test evaluation (1 prediction per test engine vs official RUL vector).
   - Supplementary full-trajectory test evaluation.
   - Asymmetric NASA scoring function (PHM 2008 benchmark).
2. Early-Warning & AI4I Classification:
   - Validation optimal decision threshold tuning (maximizing F1).
   - Precision, Recall, F1, PR-AUC, ROC-AUC, Confusion Matrix.
3. Parameter footprint and inference timing analysis.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation.metrics import (
    compute_regression_metrics,
    compute_classification_metrics,
    find_optimal_threshold,
    nasa_asymmetric_scoring_function,
)


def get_model_size_info(model: nn.Module) -> Dict[str, Union[int, float]]:
    """Compute parameter count and memory footprint."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size_kb = (param_size_bytes + buffer_size_bytes) / 1024.0

    return {
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "model_size_kb": round(float(total_size_kb), 2),
    }


def predict_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
    is_classification: bool = False,
) -> np.ndarray:
    """Generate model predictions across a DataLoader."""
    model.eval()
    model = model.to(device)
    all_preds = []

    with torch.no_grad():
        for X_b, _ in loader:
            X_b = X_b.to(device)
            if is_classification:
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_b)
                else:
                    logits = model(X_b)
                    probs = torch.sigmoid(logits)
                all_preds.extend(probs.cpu().numpy().tolist())
            else:
                preds = model(X_b)
                all_preds.extend(preds.cpu().numpy().tolist())

    preds_arr = np.array(all_preds, dtype=np.float32)
    # Clip negative RUL to 0 for physical plausibility
    if not is_classification:
        preds_arr = np.clip(preds_arr, a_min=0.0, a_max=None)
    return preds_arr


def evaluate_cmapss_rul(
    model: nn.Module,
    val_loader: DataLoader,
    test_loader: DataLoader,
    val_y_true: np.ndarray,
    test_y_true: np.ndarray,
    terminal_test_indices: np.ndarray,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Evaluate deep learning model on validation and official test fleet terminal cycle."""
    # 1. Validation predictions
    y_val_pred = predict_dataset(model, val_loader, device=device, is_classification=False)
    val_metrics = compute_regression_metrics(val_y_true, y_val_pred, prefix="val")

    # 2. Test predictions (all windows)
    y_test_all_pred = predict_dataset(model, test_loader, device=device, is_classification=False)
    test_all_metrics = compute_regression_metrics(test_y_true, y_test_all_pred, prefix="test_all")

    # 3. Test primary benchmark: terminal cycle per test engine
    y_test_terminal_pred = y_test_all_pred[terminal_test_indices]
    y_test_terminal_true = test_y_true[terminal_test_indices]
    test_terminal_metrics = compute_regression_metrics(
        y_test_terminal_true, y_test_terminal_pred, prefix="test_terminal"
    )

    size_info = get_model_size_info(model)

    return {
        "val_metrics": val_metrics,
        "test_terminal_metrics": test_terminal_metrics,
        "test_all_metrics": test_all_metrics,
        "model_size": size_info,
        "y_val_pred": y_val_pred,
        "y_test_terminal_pred": y_test_terminal_pred,
        "y_test_terminal_true": y_test_terminal_true,
        "y_test_all_pred": y_test_all_pred,
    }


def evaluate_classification_model(
    model: nn.Module,
    val_loader: DataLoader,
    test_loader: DataLoader,
    val_y_true: np.ndarray,
    test_y_true: np.ndarray,
    terminal_test_indices: Optional[np.ndarray] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Evaluate classification model with validation threshold tuning."""
    # 1. Validation probabilities
    val_probs = predict_dataset(model, val_loader, device=device, is_classification=True)
    best_th, best_f1 = find_optimal_threshold(val_y_true, val_probs, metric="f1")

    # 2. Test probabilities
    test_probs = predict_dataset(model, test_loader, device=device, is_classification=True)

    # Validation metrics at optimal threshold
    val_metrics = compute_classification_metrics(val_y_true, val_probs, threshold=best_th, prefix="val")

    # Test metrics (all windows / all test records)
    test_metrics = compute_classification_metrics(test_y_true, test_probs, threshold=best_th, prefix="test")

    # Terminal test metrics if indices provided
    test_terminal_metrics = None
    if terminal_test_indices is not None:
        term_probs = test_probs[terminal_test_indices]
        term_true = test_y_true[terminal_test_indices]
        test_terminal_metrics = compute_classification_metrics(
            term_true, term_probs, threshold=best_th, prefix="test_terminal"
        )

    size_info = get_model_size_info(model)

    return {
        "optimal_threshold": round(float(best_th), 4),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_terminal_metrics": test_terminal_metrics,
        "model_size": size_info,
        "val_probs": val_probs,
        "test_probs": test_probs,
    }
