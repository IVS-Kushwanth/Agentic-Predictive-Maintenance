"""Phase 2 Visualization Module for Model Evaluation and Comparison.

Generates:
1. RUL Regression Predicted vs Actual scatter and terminal trajectory plots.
2. RUL Prediction Error Distribution histograms (with asymmetric penalty shading).
3. Precision-Recall curves for NASA Early-Warning (Horizons 10, 20, 30).
4. Confusion matrices for Early-Warning classification.
5. Precision-Recall & ROC curves for AI4I Machine Failure classification.
6. Confusion matrix for AI4I best model.
7. Top Feature Importance charts across tree models.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

# Publication styling
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"


def plot_rul_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    out_dir: Path,
    dataset_tag: str = "Test Terminal Benchmark"
) -> Path:
    """Plot Predicted vs Actual RUL scatter plot with 45-degree reference line."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.7, color="#1f77b4", edgecolors="k", s=40, label="Predictions")
    
    min_val = min(float(y_true.min()), float(y_pred.min()))
    max_val = max(float(y_true.max()), float(y_pred.max()))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideal (y = x)")

    ax.set_title(f"NASA C-MAPSS: Predicted vs Actual RUL ({model_name})\n[{dataset_tag}]", fontsize=11, pad=10)
    ax.set_xlabel("Actual RUL (Cycles)")
    ax.set_ylabel("Predicted RUL (Cycles)")
    ax.legend(loc="upper left")

    out_file = out_dir / f"01_rul_pred_vs_actual_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(out_file)
    plt.close(fig)
    return out_file


def plot_rul_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    out_dir: Path
) -> Path:
    """Plot histogram and KDE of prediction error d = y_pred - y_true."""
    errors = y_pred - y_true
    fig, ax = plt.subplots(figsize=(8, 5))
    
    sns.histplot(errors, kde=True, bins=25, color="#2ca02c", ax=ax)
    ax.axvline(0, color="black", linestyle="-", lw=1.5, label="Zero Error")
    ax.axvline(float(np.mean(errors)), color="red", linestyle="--", label=f"Mean Error: {np.mean(errors):.2f}")
    ax.axvline(float(np.median(errors)), color="blue", linestyle=":", label=f"Median Error: {np.median(errors):.2f}")

    # Shade asymmetric risk regions
    ax.axvspan(0, float(max(errors.max(), 10)), color="red", alpha=0.08, label="Late Prediction (Dangerous)")
    ax.axvspan(float(min(errors.min(), -10)), 0, color="orange", alpha=0.08, label="Early Prediction (Safe)")

    ax.set_title(f"RUL Prediction Error Distribution: {model_name}", fontsize=11, pad=10)
    ax.set_xlabel("Prediction Error (d = Predicted - Actual Cycles)")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper right", fontsize=8)

    out_file = out_dir / f"02_rul_error_dist_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(out_file)
    plt.close(fig)
    return out_file


def plot_early_warning_pr_curves(
    ew_test_predictions: Dict[str, Dict[str, Any]],
    horizons: List[int],
    out_dir: Path
) -> Path:
    """Plot Precision-Recall curves across early-warning horizons for top models."""
    fig, axes = plt.subplots(1, len(horizons), figsize=(16, 5), sharey=True)
    if len(horizons) == 1:
        axes = [axes]

    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]

    for idx, h in enumerate(horizons):
        ax = axes[idx]
        color_idx = 0
        for key, pdata in ew_test_predictions.items():
            if f"_{h}_" in key:
                clean_label = key.split(f"_{h}_")[1].replace("_fd001", "").replace("_", " ").title()
                p, r, _ = precision_recall_curve(pdata["true"], pdata["prob"])
                ax.plot(r, p, lw=1.8, label=clean_label, color=colors[color_idx % len(colors)])
                color_idx += 1

        baseline = float(np.mean([pdata["true"].mean() for k, pdata in ew_test_predictions.items() if f"_{h}_" in k]))
        ax.axhline(baseline, color="gray", linestyle=":", label=f"Prevalence ({baseline:.2f})")
        ax.set_title(f"Horizon H={h} cycles (RUL <= {h})", fontsize=11)
        ax.set_xlabel("Recall")
        if idx == 0:
            ax.set_ylabel("Precision")
        ax.legend(loc="lower left", fontsize=8)

    fig.suptitle("NASA C-MAPSS: Precision-Recall Curves by Early-Warning Horizon", fontsize=12, y=1.02)
    plt.tight_layout()

    out_file = out_dir / "03_early_warning_pr_curves.png"
    fig.savefig(out_file)
    plt.close(fig)
    return out_file


def plot_ai4i_pr_and_roc_curves(
    ai4i_test_predictions: Dict[str, Dict[str, Any]],
    out_dir: Path
) -> Tuple[Path, Path]:
    """Plot PR and ROC curves for AI4I Machine Failure models."""
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]

    # 1. Precision-Recall Curve
    fig_pr, ax_pr = plt.subplots(figsize=(7, 5))
    # 2. ROC Curve
    fig_roc, ax_roc = plt.subplots(figsize=(7, 5))

    for idx, (m_name, pdata) in enumerate(ai4i_test_predictions.items()):
        c = colors[idx % len(colors)]
        # PR
        p, r, _ = precision_recall_curve(pdata["true"], pdata["prob"])
        ax_pr.plot(r, p, lw=2, color=c, label=m_name)

        # ROC
        fpr, tpr, _ = roc_curve(pdata["true"], pdata["prob"])
        ax_roc.plot(fpr, tpr, lw=2, color=c, label=m_name)

    # PR formatting
    test_prevalence = float(list(ai4i_test_predictions.values())[0]["true"].mean())
    ax_pr.axhline(test_prevalence, color="gray", linestyle=":", label=f"No-skill baseline ({test_prevalence*100:.2f}%)")
    ax_pr.set_title("UCI AI4I 2020: Precision-Recall Curves (Test Fleet)", fontsize=11, pad=10)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(loc="lower left", fontsize=9)

    out_pr = out_dir / "04_ai4i_pr_curves.png"
    fig_pr.savefig(out_pr)
    plt.close(fig_pr)

    # ROC formatting
    ax_roc.plot([0, 1], [0, 1], "k--", label="Random Chance")
    ax_roc.set_title("UCI AI4I 2020: ROC Curves (Test Fleet)", fontsize=11, pad=10)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right", fontsize=9)

    out_roc = out_dir / "05_ai4i_roc_curves.png"
    fig_roc.savefig(out_roc)
    plt.close(fig_roc)

    return out_pr, out_roc


def plot_confusion_matrix_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    class_labels: List[str],
    out_path: Path
) -> Path:
    """Plot annotated confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=class_labels, yticklabels=class_labels)
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    title: str,
    out_path: Path,
    top_k: int = 15
) -> Path:
    """Plot top-k feature importances."""
    idx = np.argsort(importances)[::-1][:top_k]
    top_feats = [feature_names[i] for i in idx]
    top_vals = importances[idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(range(len(top_feats)), top_vals[::-1], color="#2b5c8f", align="center")
    ax.set_yticks(range(len(top_feats)))
    ax.set_yticklabels(top_feats[::-1], fontsize=9)
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel("Relative Importance Score")

    fig.savefig(out_path)
    plt.close(fig)
    return out_path
