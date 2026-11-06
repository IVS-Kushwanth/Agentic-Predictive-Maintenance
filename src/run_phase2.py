"""Master Execution Orchestrator for Phase 2: ML Baseline Development & Evaluation.

Coordinates:
1. NASA C-MAPSS RUL Regression (FD001 benchmark, extended to FD002-FD004)
2. NASA C-MAPSS Early-Warning Multi-Horizon Classification (H in [10, 20, 30])
3. UCI AI4I 2020 Machine Failure Classification & Failure Mode Analysis
4. Experiment logging to reports/phase2/experiment_log.csv
5. Publication-quality figure generation to reports/figures/phase2/
6. Generation of Markdown comparison and evaluation reports
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from training.nasa_rul_trainer import train_and_evaluate_nasa_rul
from training.nasa_ew_trainer import train_and_evaluate_nasa_early_warning
from training.ai4i_trainer import train_and_evaluate_ai4i
from evaluation.visualization import (
    plot_rul_predicted_vs_actual,
    plot_rul_error_distribution,
    plot_early_warning_pr_curves,
    plot_ai4i_pr_and_roc_curves,
    plot_confusion_matrix_heatmap,
    plot_feature_importance,
)


def run_phase2():
    print("=" * 75)
    print("PHASE 2: MACHINE LEARNING BASELINE DEVELOPMENT & EVALUATION")
    print("=" * 75)

    base_dir = Path(".")
    reports_p2_dir = base_dir / "reports" / "phase2"
    figures_p2_dir = base_dir / "reports" / "figures" / "phase2"
    models_base_dir = base_dir / "models" / "baseline"

    reports_p2_dir.mkdir(parents=True, exist_ok=True)
    figures_p2_dir.mkdir(parents=True, exist_ok=True)
    models_base_dir.mkdir(parents=True, exist_ok=True)

    all_logs = []

    # -------------------------------------------------------------
    # 1. NASA C-MAPSS RUL REGRESSION (FD001 Benchmark)
    # -------------------------------------------------------------
    nasa_rul_fd1 = train_and_evaluate_nasa_rul(
        subset="FD001",
        processed_dir=base_dir / "data" / "processed" / "nasa",
        models_dir=models_base_dir,
        random_state=42
    )
    all_logs.extend(nasa_rul_fd1["results"])

    # Extend pipeline to FD002, FD003, FD004 as required by Section 2
    for sub in ["FD002", "FD003", "FD004"]:
        sub_res = train_and_evaluate_nasa_rul(
            subset=sub,
            processed_dir=base_dir / "data" / "processed" / "nasa",
            models_dir=models_base_dir,
            random_state=42
        )
        all_logs.extend(sub_res["results"])

    # -------------------------------------------------------------
    # 2. NASA C-MAPSS EARLY-WARNING CLASSIFICATION (FD001 Benchmark)
    # -------------------------------------------------------------
    nasa_ew_fd1 = train_and_evaluate_nasa_early_warning(
        subset="FD001",
        horizons=[10, 20, 30],
        processed_dir=base_dir / "data" / "processed" / "nasa",
        models_dir=models_base_dir,
        random_state=42
    )
    all_logs.extend(nasa_ew_fd1["results"])

    # -------------------------------------------------------------
    # 3. UCI AI4I 2020 MACHINE FAILURE CLASSIFICATION
    # -------------------------------------------------------------
    ai4i_res = train_and_evaluate_ai4i(
        processed_dir=base_dir / "data" / "processed" / "ai4i",
        models_dir=models_base_dir,
        val_size=0.20,
        random_state=42
    )
    all_logs.extend(ai4i_res["results"])

    # -------------------------------------------------------------
    # 4. SAVE COMPLETE EXPERIMENT LOG
    # -------------------------------------------------------------
    log_df = pd.DataFrame(all_logs)
    log_csv_path = reports_p2_dir / "experiment_log.csv"
    log_df.to_csv(log_csv_path, index=False)
    print(f"\nSaved complete experiment log with {len(log_df)} runs to: {log_csv_path}")

    # -------------------------------------------------------------
    # 5. GENERATE PUBLICATION-QUALITY EVALUATION FIGURES
    # -------------------------------------------------------------
    print("\nGenerating Phase 2 Evaluation Visualizations...")

    # (a) RUL Best Model: Predicted vs Actual on Test Terminal Benchmark
    best_rul_name = nasa_rul_fd1["best_model_name"]
    best_terminal_preds = nasa_rul_fd1["test_predictions"][best_rul_name]["terminal_pred"]
    best_terminal_true = nasa_rul_fd1["test_predictions"][best_rul_name]["terminal_true"]

    plot_rul_predicted_vs_actual(
        best_terminal_true,
        best_terminal_preds,
        best_rul_name,
        figures_p2_dir,
        dataset_tag="FD001 Official Test Terminal Benchmark"
    )

    # (b) RUL Error Distribution
    plot_rul_error_distribution(
        best_terminal_true,
        best_terminal_preds,
        best_rul_name,
        figures_p2_dir
    )

    # (c) Early Warning PR Curves
    plot_early_warning_pr_curves(
        nasa_ew_fd1["test_predictions"],
        horizons=[10, 20, 30],
        out_dir=figures_p2_dir
    )

    # (d) Early Warning Confusion Matrix for Best Model (H=30)
    best_ew_key = f"nasa_ew_30_random_forest_classifier_{nasa_ew_fd1['subset']}"
    if best_ew_key in nasa_ew_fd1["test_predictions"]:
        ew_pred_obj = nasa_ew_fd1["test_predictions"][best_ew_key]
        ew_pred_bin = (ew_pred_obj["prob"] >= ew_pred_obj["threshold"]).astype(int)
        plot_confusion_matrix_heatmap(
            ew_pred_obj["true"],
            ew_pred_bin,
            title="NASA C-MAPSS: Early-Warning (H=30) Confusion Matrix",
            class_labels=["Normal (>30)", "Warning (<=30)"],
            out_path=figures_p2_dir / "06_early_warning_h30_cm.png"
        )

    # (e) AI4I PR & ROC Curves
    plot_ai4i_pr_and_roc_curves(
        ai4i_res["test_predictions"],
        figures_p2_dir
    )

    # (f) AI4I Confusion Matrix for Best Model
    best_ai4i_name = ai4i_res["best_model_name"]
    best_ai4i_pred_obj = ai4i_res["test_predictions"][best_ai4i_name]
    best_ai4i_bin = (best_ai4i_pred_obj["prob"] >= best_ai4i_pred_obj["threshold"]).astype(int)
    plot_confusion_matrix_heatmap(
        best_ai4i_pred_obj["true"],
        best_ai4i_bin,
        title=f"UCI AI4I: Machine Failure Confusion Matrix ({best_ai4i_name})",
        class_labels=["No Failure", "Machine Failure"],
        out_path=figures_p2_dir / "07_ai4i_confusion_matrix.png"
    )

    # (g) Feature Importance: NASA RUL Best Tree Model
    clean_rul_name = best_rul_name.lower().replace(" ", "_")
    best_rul_model = joblib.load(nasa_rul_fd1["saved_model_paths"][best_rul_name])
    if hasattr(best_rul_model, "feature_importances_"):
        plot_feature_importance(
            nasa_rul_fd1["feature_cols"],
            best_rul_model.feature_importances_,
            title=f"Top Features for RUL Regression ({best_rul_name})",
            out_path=figures_p2_dir / "08_feature_importance_nasa_rul.png",
            top_k=15
        )

    # (h) Feature Importance: AI4I Best Tree Model
    best_ai4i_model = joblib.load(ai4i_res["saved_model_paths"][best_ai4i_name])
    if hasattr(best_ai4i_model, "feature_importances_"):
        plot_feature_importance(
            ai4i_res["feature_cols"],
            best_ai4i_model.feature_importances_,
            title=f"Top Features for AI4I Failure Classification ({best_ai4i_name})",
            out_path=figures_p2_dir / "09_feature_importance_ai4i.png",
            top_k=14
        )

    print("All Phase 2 evaluation visualizations generated successfully.")
    return {
        "nasa_rul": nasa_rul_fd1,
        "nasa_ew": nasa_ew_fd1,
        "ai4i": ai4i_res,
        "log_path": str(log_csv_path),
    }


if __name__ == "__main__":
    run_phase2()
