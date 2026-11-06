"""Visualization Module for Deep Learning Predictive Maintenance Experiments.

Generates high-resolution publication-quality plots:
1. Training and validation loss curves.
2. Predicted vs actual RUL on official test fleet terminal cycles.
3. Residual error distributions.
4. Trajectory degradation tracking across engine operational lifespans.
5. Window length sensitivity analysis (w=30 vs 50 vs 60).
6. Early-warning Precision-Recall curves across horizons.
7. Confusion matrices.
8. AI4I PR/ROC and confusion matrix.
9. Cross-paradigm ML vs DL comparison bar charts.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve

# Apply clean plotting style
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
})


def save_figure_to_both(fig: plt.Figure, filename: str, dirs: List[Path]) -> None:
    """Save matplotlib figure to multiple destination directories at 300 DPI."""
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(
    histories: Dict[str, Dict[str, List[float]]],
    output_dirs: List[Path],
    filename: str = "01_loss_curves_rul_lstm_gru_cnnlstm.png",
) -> None:
    """Plot training and validation loss progression over epochs."""
    fig, axes = plt.subplots(1, len(histories), figsize=(6 * len(histories), 4.5), sharey=False)
    if len(histories) == 1:
        axes = [axes]

    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]

    for ax, (model_name, hist), col in zip(axes, histories.items(), colors):
        epochs = range(1, len(hist["train_loss"]) + 1)
        ax.plot(epochs, hist["train_loss"], label="Train Loss", color=col, linestyle="-", linewidth=2)
        ax.plot(epochs, hist["val_loss"], label="Val Loss", color=col, linestyle="--", linewidth=2, alpha=0.8)
        ax.set_title(f"{model_name} Convergence", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Huber Loss")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle("Deep Learning Training and Validation Convergence (NASA FD001)", fontweight="bold", y=1.02)
    plt.tight_layout()
    save_figure_to_both(fig, filename, output_dirs)


def plot_rul_pred_vs_actual(
    models_preds: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_dirs: List[Path],
    filename: str = "02_rul_pred_vs_actual_deep_learning.png",
) -> None:
    """Plot predicted vs actual RUL scatter for terminal test cycles."""
    n_models = len(models_preds)
    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, (name, (y_true, y_pred)) in zip(axes, models_preds.items()):
        max_val = max(float(np.max(y_true)), float(np.max(y_pred))) + 15
        ax.scatter(y_true, y_pred, alpha=0.75, edgecolors="k", s=45, color="#1f77b4")
        ax.plot([0, max_val], [0, max_val], "r--", linewidth=2, label="Ideal 1:1 Identity")
        ax.set_title(f"{name}\nTest Terminal Fleet (N=100)", fontweight="bold")
        ax.set_xlabel("Official True RUL (Cycles)")
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.legend(loc="upper left")
        ax.grid(True, linestyle=":", alpha=0.6)

    axes[0].set_ylabel("Predicted RUL (Cycles)")
    fig.suptitle("Remaining Useful Life: Predicted vs Actual on NASA Test Fleet", fontweight="bold", y=1.02)
    plt.tight_layout()
    save_figure_to_both(fig, filename, output_dirs)


def plot_rul_error_distribution(
    models_preds: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_dirs: List[Path],
    filename: str = "03_rul_error_dist_deep_learning.png",
) -> None:
    """Plot residual error (pred - true) histograms."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for (name, (y_true, y_pred)), col in zip(models_preds.items(), palette):
        errors = y_pred - y_true
        sns.kdeplot(errors, ax=ax, label=f"{name} (mean={errors.mean():.1f}, std={errors.std():.1f})", color=col, linewidth=2.2)

    ax.axvline(0, color="black", linestyle="--", linewidth=1.5, label="Zero Error Line")
    ax.set_title("Test Terminal Residual Error Distributions (Pred - True)", fontweight="bold")
    ax.set_xlabel("Prediction Error (Cycles) [Negative = Early, Positive = Late]")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    save_figure_to_both(fig, filename, output_dirs)


def plot_trajectory_tracking(
    engine_trajectories: List[Dict[str, Any]],
    output_dirs: List[Path],
    filename: str = "04_rul_trajectory_tracking.png",
) -> None:
    """Plot predicted vs true RUL degradation trajectories over operational cycles."""
    n_engines = len(engine_trajectories)
    fig, axes = plt.subplots(1, n_engines, figsize=(5.5 * n_engines, 4.5), sharey=True)
    if n_engines == 1:
        axes = [axes]

    for ax, traj in zip(axes, engine_trajectories):
        cycles = traj["cycles"]
        true_rul = traj["true_rul"]
        lstm_rul = traj["lstm_rul"]
        gru_rul = traj["gru_rul"]
        unit_id = traj["unit_id"]

        ax.plot(cycles, true_rul, "k--", linewidth=2.2, label="True RUL")
        ax.plot(cycles, lstm_rul, color="#1f77b4", linewidth=2, label="LSTM Pred")
        ax.plot(cycles, gru_rul, color="#2ca02c", linewidth=2, linestyle="-.", label="GRU Pred")
        ax.set_title(f"Test Engine #{unit_id} Degradation", fontweight="bold")
        ax.set_xlabel("Operational Cycle")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right")

    axes[0].set_ylabel("Remaining Useful Life (Cycles)")
    fig.suptitle("Multivariate Temporal Degradation Tracking Over Operational Lifespan", fontweight="bold", y=1.02)
    plt.tight_layout()
    save_figure_to_both(fig, filename, output_dirs)


def plot_window_length_comparison(
    window_results: Dict[str, Dict[str, float]],
    output_dirs: List[Path],
    filename: str = "05_window_length_comparison.png",
) -> None:
    """Plot RMSE, MAE, and NASA score across sequence window lengths (30, 50, 60)."""
    windows = list(window_results.keys())
    rmses = [window_results[w]["rmse"] for w in windows]
    maes = [window_results[w]["mae"] for w in windows]
    scores = [window_results[w]["nasa_score"] for w in windows]

    x = np.arange(len(windows))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    rects1 = ax1.bar(x - width / 2, rmses, width, label="Test Terminal RMSE", color="#1f77b4", alpha=0.85)
    rects2 = ax1.bar(x + width / 2, maes, width, label="Test Terminal MAE", color="#aec7e8", alpha=0.85)
    line1 = ax2.plot(x, scores, color="#d62728", marker="o", linewidth=2.5, markersize=8, label="NASA Asymmetric Score")

    ax1.set_xlabel("Sequence Window Length (Cycles)")
    ax1.set_ylabel("Error in Cycles (RMSE / MAE)")
    ax2.set_ylabel("NASA Asymmetric Scoring Function", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_xticks(x)
    ax1.set_xticklabels(windows)
    ax1.set_title("Impact of Temporal Sequence Window Length on RUL Accuracy (FD001)", fontweight="bold")

    # Combine legends
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    ax1.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    save_figure_to_both(fig, filename, output_dirs)


def plot_early_warning_pr_curves(
    pr_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    output_dirs: List[Path],
    filename: str = "06_early_warning_dl_pr_curves.png",
) -> None:
    """Plot Precision-Recall curves for early warning horizons."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    palette = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]

    for (label, (recalls, precisions, pr_auc)), col in zip(pr_data.items(), palette):
        ax.plot(recalls, precisions, label=f"{label} (PR-AUC = {pr_auc:.4f})", color=col, linewidth=2.2)

    ax.set_title("Early-Warning Classification: Precision-Recall Curves Across Horizons", fontweight="bold")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    save_figure_to_both(fig, filename, output_dirs)


def plot_confusion_matrix(
    cm: np.ndarray,
    title: str,
    output_dirs: List[Path],
    filename: str,
    class_names: List[str] = ["Normal", "Warning"],
) -> None:
    """Plot normalized and annotated confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.1%})"

    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=True, ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Ground Truth Label")

    plt.tight_layout()
    save_figure_to_both(fig, filename, output_dirs)


def plot_ai4i_curves_and_cm(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cm: np.ndarray,
    optimal_th: float,
    output_dirs: List[Path],
    filename: str = "08_ai4i_mlp_curves_cm.png",
) -> None:
    """Plot PR curve, ROC curve, and confusion matrix for AI4I MLP classifier."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # 1. PR Curve
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    from sklearn.metrics import average_precision_score, roc_auc_score
    pr_auc = average_precision_score(y_true, y_prob)
    axes[0].plot(rec, prec, color="#1f77b4", linewidth=2.2, label=f"PR-AUC = {pr_auc:.4f}")
    axes[0].set_title("AI4I MLP Precision-Recall Curve", fontweight="bold")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].legend(loc="lower left")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    axes[1].plot(fpr, tpr, color="#2ca02c", linewidth=2.2, label=f"ROC-AUC = {roc_auc:.4f}")
    axes[1].plot([0, 1], [0, 1], "k--", label="Random Classifier")
    axes[1].set_title("AI4I MLP ROC Curve", fontweight="bold")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    # 3. Confusion Matrix
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.1%})"

    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=False, ax=axes[2],
                xticklabels=["No Failure", "Failure"], yticklabels=["No Failure", "Failure"])
    axes[2].set_title(f"Confusion Matrix (Th={optimal_th:.2f})", fontweight="bold")
    axes[2].set_xlabel("Predicted")
    axes[2].set_ylabel("Actual")

    plt.tight_layout()
    save_figure_to_both(fig, filename, output_dirs)


def plot_ml_vs_dl_comparison(
    comparison_data: Dict[str, Dict[str, float]],
    output_dirs: List[Path],
    filename: str = "09_ml_vs_dl_comparison_bar.png",
) -> None:
    """Side-by-side comparative bar chart: Traditional ML vs Deep Learning."""
    models = list(comparison_data.keys())
    rmses = [comparison_data[m]["rmse"] for m in models]
    maes = [comparison_data[m]["mae"] for m in models]
    scores = [comparison_data[m]["nasa_score"] for m in models]
    paradigms = [comparison_data[m]["paradigm"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)

    # Top plot: RMSE and MAE
    bar_colors_rmse = ["#4a7bb0" if p == "ML" else "#2b8a3e" for p in paradigms]
    bar_colors_mae = ["#90b4ce" if p == "ML" else "#8ce99a" for p in paradigms]

    ax1.bar(x - width / 2, rmses, width, label="Test Terminal RMSE", color=bar_colors_rmse)
    ax1.bar(x + width / 2, maes, width, label="Test Terminal MAE", color=bar_colors_mae)
    ax1.set_ylabel("Error in Cycles (Cycles)")
    ax1.set_title("Machine Learning Baselines vs Deep Learning Architectures (NASA FD001)", fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Custom legend for paradigms
    import matplotlib.patches as mpatches
    ml_patch = mpatches.Patch(color="#4a7bb0", label="Traditional ML (Phase 2)")
    dl_patch = mpatches.Patch(color="#2b8a3e", label="Deep Learning (Phase 3)")
    ax1.legend(handles=[ml_patch, dl_patch], loc="upper right")

    # Bottom plot: NASA Score (log scale recommended due to asymmetric penalty)
    ax2.bar(x, scores, width=0.5, color=["#e056fd" if p == "ML" else "#ff9f43" for p in paradigms], alpha=0.9)
    ax2.set_ylabel("NASA Asymmetric Score")
    ax2.set_xlabel("Predictive Model Architecture")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=20, ha="right")
    ax2.set_yscale("log")
    ax2.grid(True, which="both", linestyle=":", alpha=0.6)

    ml_score_patch = mpatches.Patch(color="#e056fd", label="ML NASA Score (Log Scale)")
    dl_score_patch = mpatches.Patch(color="#ff9f43", label="DL NASA Score (Log Scale)")
    ax2.legend(handles=[ml_score_patch, dl_score_patch], loc="upper right")

    plt.tight_layout()
    save_figure_to_both(fig, filename, output_dirs)
