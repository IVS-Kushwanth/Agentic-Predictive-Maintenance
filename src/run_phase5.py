"""
Phase 5: Explainable AI Execution Script.
Computes SHAP values, Permutation Feature Importance, and Deep Learning Integrated Gradients;
generates all 8 publication visualizations and serializes explainability artifacts.
"""

import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy.stats import spearmanr
import shap

# Ensure project root is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.explainability.shap_explainer import TreeSHAPExplainer
from src.explainability.permutation_explainer import PermutationExplainer
from src.explainability.deep_explainer import DeepTemporalExplainer
from src.explainability.schema import format_explanation_response
from src.deep_learning.lstm_model import LSTMRULModel
from src.deep_learning.gru_model import GRURULModel

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.size": 10, "figure.autolayout": True})


def setup_directories():
    paths = [
        os.path.join(BASE_DIR, "models", "explainability"),
        os.path.join(BASE_DIR, "reports", "phase5", "figures"),
        os.path.join(BASE_DIR, "reports", "figures", "phase5"),
    ]
    for p in paths:
        os.makedirs(p, exist_ok=True)


def run_phase5():
    print("=" * 80)
    print("STARTING PHASE 5: EXPLAINABLE AI FOR PREDICTIVE MAINTENANCE")
    print("=" * 80)
    setup_directories()
    t_start = time.time()

    summary_metrics = {
        "nasa_rul_shap": {},
        "ai4i_shap": {},
        "permutation_comparison": {},
        "deep_learning_temporal": {},
        "execution_time_seconds": 0.0
    }

    # =========================================================================
    # PART 1: NASA C-MAPSS SHAP EXPLAINABILITY (RUL REGRESSION)
    # =========================================================================
    print("\n--- 1. NASA C-MAPSS FD001 SHAP Analysis ---")
    nasa_meta_path = os.path.join(BASE_DIR, "models", "baseline", "nasa_rul_xgboost_regressor_FD001_meta.json")
    with open(nasa_meta_path, "r") as f:
        nasa_meta = json.load(f)
    nasa_features = nasa_meta["features"]

    val_tabular_path = os.path.join(BASE_DIR, "data", "processed", "nasa", "FD001", "val_tabular.csv")
    val_df = pd.read_csv(val_tabular_path)

    # Sample representative evaluation set (250 samples spanning high and low RUL)
    np.random.seed(42)
    sample_indices = np.random.choice(len(val_df), size=min(250, len(val_df)), replace=False)
    val_sample_df = val_df.iloc[sample_indices]
    X_val_nasa = val_sample_df[nasa_features].to_numpy(dtype=np.float32)
    y_val_nasa = val_sample_df["RUL"].to_numpy(dtype=np.float32)

    # Load trained XGBoost Regressor
    xgb_nasa = joblib.load(os.path.join(BASE_DIR, "models", "baseline", "nasa_rul_xgboost_regressor_FD001.joblib"))
    rf_nasa = joblib.load(os.path.join(BASE_DIR, "models", "baseline", "nasa_rul_random_forest_regressor_FD001.joblib"))

    print("Computing TreeSHAP for NASA XGBoost Regressor...")
    nasa_explainer = TreeSHAPExplainer(xgb_nasa, nasa_features, task_type="regression", model_id="nasa_rul_xgboost")
    shap_matrix_nasa = nasa_explainer.compute_dataset_shap(X_val_nasa)
    global_importance_nasa = nasa_explainer.get_global_importance(X_val_nasa, top_k=20)

    print(f"  NASA Top 5 SHAP Features:")
    for item in global_importance_nasa[:5]:
        print(f"    Rank {item['rank']}: {item['feature']:28s} (Mean |SHAP| = {item['mean_abs_shap']:.4f} cycles)")

    summary_metrics["nasa_rul_shap"]["top_features"] = global_importance_nasa[:15]

    # Individual instance explanations: High RUL vs Low RUL
    high_rul_idx = np.argmax(y_val_nasa)
    low_rul_idx = np.argmin(y_val_nasa)

    exp_high_rul = nasa_explainer.explain_instance(X_val_nasa[high_rul_idx], top_k=5)
    exp_low_rul = nasa_explainer.explain_instance(X_val_nasa[low_rul_idx], top_k=5)

    print(f"\n  Individual Explanation - Early Life (True RUL = {y_val_nasa[high_rul_idx]:.1f}):")
    print(f"    Predicted: {exp_high_rul['prediction']['predicted_rul']} cycles")
    print(f"    Summary:   {exp_high_rul['explanation_summary']}")

    print(f"\n  Individual Explanation - Near Failure (True RUL = {y_val_nasa[low_rul_idx]:.1f}):")
    print(f"    Predicted: {exp_low_rul['prediction']['predicted_rul']} cycles")
    print(f"    Summary:   {exp_low_rul['explanation_summary']}")

    # =========================================================================
    # PART 2: UCI AI4I 2020 SHAP EXPLAINABILITY (FAILURE CLASSIFICATION)
    # =========================================================================
    print("\n" + "=" * 80)
    print("--- 2. UCI AI4I 2020 Machine Failure SHAP Analysis ---")
    ai4i_meta_path = os.path.join(BASE_DIR, "models", "baseline", "ai4i_clf_xgboost_classifier_meta.json")
    with open(ai4i_meta_path, "r") as f:
        ai4i_meta = json.load(f)
    ai4i_features = ai4i_meta["features"]

    X_test_ai4i = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "ai4i", "X_test.csv"))
    y_test_ai4i = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "ai4i", "y_test.csv")).values.ravel()

    # Balance evaluation sample (all 68 failures + 182 normal samples = 250 samples)
    fail_idx = np.where(y_test_ai4i == 1)[0]
    norm_idx = np.where(y_test_ai4i == 0)[0]
    sampled_norm = np.random.choice(norm_idx, size=182, replace=False)
    ai4i_eval_idx = np.concatenate([fail_idx, sampled_norm])

    X_eval_ai4i = X_test_ai4i.iloc[ai4i_eval_idx].to_numpy(dtype=np.float32)
    y_eval_ai4i = y_test_ai4i[ai4i_eval_idx]

    xgb_ai4i = joblib.load(os.path.join(BASE_DIR, "models", "baseline", "ai4i_clf_xgboost_classifier.joblib"))

    print("Computing TreeSHAP for AI4I XGBoost Classifier...")
    ai4i_explainer = TreeSHAPExplainer(xgb_ai4i, ai4i_features, task_type="classification", model_id="ai4i_clf_xgboost")
    shap_matrix_ai4i = ai4i_explainer.compute_dataset_shap(X_eval_ai4i)
    global_importance_ai4i = ai4i_explainer.get_global_importance(X_eval_ai4i, top_k=14)

    print(f"  AI4I Top 5 SHAP Features:")
    for item in global_importance_ai4i[:5]:
        print(f"    Rank {item['rank']}: {item['feature']:28s} (Mean |SHAP| = {item['mean_abs_shap']:.4f})")

    summary_metrics["ai4i_shap"]["top_features"] = global_importance_ai4i

    # Normal vs Failure individual explanation
    exp_fail = ai4i_explainer.explain_instance(X_test_ai4i.iloc[fail_idx[0]].to_numpy(dtype=np.float32), top_k=5)
    exp_norm = ai4i_explainer.explain_instance(X_test_ai4i.iloc[norm_idx[0]].to_numpy(dtype=np.float32), top_k=5)

    print(f"\n  AI4I Failure Instance Explanation (True Failure):")
    print(f"    Predicted Prob: {exp_fail['prediction']['predicted_probability']*100:.1f}%")
    print(f"    Summary:        {exp_fail['explanation_summary']}")

    # =========================================================================
    # PART 3: PERMUTATION FEATURE IMPORTANCE & SHAP COMPARISON
    # =========================================================================
    print("\n" + "=" * 80)
    print("--- 3. Permutation Feature Importance & SHAP Comparison ---")

    # NASA Permutation Importance
    print("Calculating Permutation Importance for NASA XGBoost (5 repeats)...")
    perm_nasa = PermutationExplainer(xgb_nasa, nasa_features, task_type="regression", model_id="nasa_rul_xgboost")
    perm_results_nasa = perm_nasa.compute_importance(X_val_nasa, y_val_nasa, n_repeats=5, scoring="r2", random_state=42)
    comparison_nasa = PermutationExplainer.compare_with_shap(perm_results_nasa, global_importance_nasa, top_k=15)

    print(f"  NASA SHAP vs Permutation Rank Correlation: r = {comparison_nasa['spearman_rank_correlation']:.4f} (p = {comparison_nasa['p_value']:.4e})")
    print(f"  Top-15 Concordant Features Count: {len(comparison_nasa['concordant_top_features'])} / 15 (Jaccard = {comparison_nasa['concordance_jaccard_index']:.2f})")
    summary_metrics["permutation_comparison"]["nasa"] = comparison_nasa

    # AI4I Permutation Importance
    print("Calculating Permutation Importance for AI4I XGBoost (5 repeats)...")
    perm_ai4i = PermutationExplainer(xgb_ai4i, ai4i_features, task_type="classification", model_id="ai4i_clf_xgboost")
    perm_results_ai4i = perm_ai4i.compute_importance(X_eval_ai4i, y_eval_ai4i, n_repeats=5, scoring="roc_auc", random_state=42)
    comparison_ai4i = PermutationExplainer.compare_with_shap(perm_results_ai4i, global_importance_ai4i, top_k=14)

    print(f"  AI4I SHAP vs Permutation Rank Correlation: r = {comparison_ai4i['spearman_rank_correlation']:.4f} (p = {comparison_ai4i['p_value']:.4e})")
    summary_metrics["permutation_comparison"]["ai4i"] = comparison_ai4i

    # =========================================================================
    # PART 4: DEEP LEARNING EXPLAINABILITY (INTEGRATED GRADIENTS)
    # =========================================================================
    print("\n" + "=" * 80)
    print("--- 4. Deep Learning Explainability: Integrated Gradients (LSTM & GRU) ---")
    val_npz = np.load(os.path.join(BASE_DIR, "data", "processed", "nasa", "FD001", "sequences_w30_val.npz"))
    X_val_seq = val_npz["X"]  # (3711, 30, 153)
    y_val_seq_rul = val_npz["y_rul"]

    # Load LSTM model
    lstm_model = LSTMRULModel(input_dim=len(nasa_features), hidden_dim=32, num_layers=1, fc_dim=32)
    lstm_ckpt = torch.load(
        os.path.join(BASE_DIR, "models", "deep_learning", "lstm", "nasa_rul_lstm_FD001_w30.pt"),
        map_location="cpu",
        weights_only=False
    )
    lstm_model.load_state_dict(lstm_ckpt)
    lstm_model.eval()

    deep_exp = DeepTemporalExplainer(lstm_model, nasa_features, model_id="nasa_rul_lstm_FD001")

    # Single sequence attribution near failure
    near_fail_seq_idx = np.argmin(y_val_seq_rul)
    seq_near_fail = X_val_seq[near_fail_seq_idx]
    deep_exp_res = deep_exp.explain_sequence(seq_near_fail, top_k_sensors=5, n_steps=25)

    print(f"  LSTM Near-Failure Explanation (True RUL = {y_val_seq_rul[near_fail_seq_idx]:.1f}):")
    print(f"    Predicted RUL: {deep_exp_res['prediction']['predicted_rul']} cycles")
    print(f"    Peak Influence Cycle: {deep_exp_res['temporal_attribution']['peak_influence_cycle_in_window']} / 30")
    print(f"    Recent vs Early Weight Ratio: {deep_exp_res['temporal_attribution']['recent_vs_early_weight_ratio']}x")
    print(f"    Top Attributed Sensors:")
    for s in deep_exp_res["important_features"]:
        print(f"      - {s['feature']:25s} Impact: {s['importance']:+.4f} (Direction: {s['contribution']})")

    # Temporal decay profile across 50 sequences
    print("Computing average temporal decay profile across 50 sequences...")
    time_decay_profile = deep_exp.compute_dataset_temporal_decay(X_val_seq, sample_size=50, n_steps=20)
    summary_metrics["deep_learning_temporal"] = {
        "time_step_weights": [round(float(w), 4) for w in time_decay_profile],
        "recency_ratio": round(float(np.sum(time_decay_profile[-10:]) / max(1e-9, np.sum(time_decay_profile[:10]))), 2)
    }
    print(f"  Average Recency Ratio (Last 10 vs First 10 cycles): {summary_metrics['deep_learning_temporal']['recency_ratio']:.2f}x")

    # =========================================================================
    # PART 5: GENERATE ALL 8 PUBLICATION VISUALIZATIONS
    # =========================================================================
    print("\n" + "=" * 80)
    print("--- 5. Generating Phase 5 Publication Visualizations ---")
    fig_dirs = [
        os.path.join(BASE_DIR, "reports", "phase5", "figures"),
        os.path.join(BASE_DIR, "reports", "figures", "phase5")
    ]

    def save_fig(fig, filename):
        for d in fig_dirs:
            fig.savefig(os.path.join(d, filename), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved figure: {filename}")

    # Figure 1: SHAP Summary Beeswarm for NASA RUL (Top 15 Features)
    top15_indices = [nasa_features.index(item["feature"]) for item in global_importance_nasa[:15]]
    top15_names = [item["feature"] for item in global_importance_nasa[:15]]
    top15_shap = shap_matrix_nasa[:, top15_indices]
    top15_vals = X_val_nasa[:, top15_indices]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    shap.summary_plot(
        top15_shap,
        top15_vals,
        feature_names=top15_names,
        show=False,
        plot_size=None,
        alpha=0.6
    )
    plt.title("NASA C-MAPSS FD001: SHAP Summary Plot for RUL Regression (Top 15 Drivers)", fontsize=11, fontweight="bold")
    save_fig(plt.gcf(), "01_shap_summary_nasa_rul.png")

    # Figure 2: SHAP Waterfall Plot for Individual Engine Prediction
    fig, ax = plt.subplots(figsize=(10, 5))
    exp_obj = shap.Explanation(
        values=shap_matrix_nasa[low_rul_idx],
        base_values=nasa_explainer.base_value,
        data=X_val_nasa[low_rul_idx],
        feature_names=nasa_features
    )
    shap.plots.waterfall(exp_obj, max_display=10, show=False)
    plt.title(f"SHAP Waterfall: Explanation for Near-Failure Engine Cycle (Pred RUL: {exp_low_rul['prediction']['predicted_rul']} cycles)", fontsize=11, fontweight="bold")
    save_fig(plt.gcf(), "02_shap_waterfall_nasa_individual.png")

    # Figure 3: SHAP Summary for AI4I Machine Failure Classification
    fig, ax = plt.subplots(figsize=(10, 5.5))
    shap.summary_plot(
        shap_matrix_ai4i,
        X_eval_ai4i,
        feature_names=ai4i_features,
        show=False,
        plot_size=None,
        alpha=0.6
    )
    plt.title("UCI AI4I 2020: SHAP Summary Plot for Machine Failure Classification", fontsize=11, fontweight="bold")
    save_fig(plt.gcf(), "03_shap_summary_ai4i_failure.png")

    # Figure 4: Feature Importance Comparison: SHAP vs Permutation Importance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    top10_shap_nasa = global_importance_nasa[:10]
    s_names = [item["feature"] for item in top10_shap_nasa][::-1]
    s_vals = [item["mean_abs_shap"] for item in top10_shap_nasa][::-1]

    ax1.barh(s_names, s_vals, color="#2980b9", edgecolor="#1b4f72", height=0.6)
    ax1.set_title("NASA RUL: Global SHAP Importance (Mean |SHAP|)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Mean |SHAP| (Cycles)")

    top10_perm_nasa = perm_results_nasa[:10]
    p_names = [item["feature"] for item in top10_perm_nasa][::-1]
    p_vals = [item["importance_mean"] for item in top10_perm_nasa][::-1]

    ax2.barh(p_names, p_vals, color="#27ae60", edgecolor="#145a32", height=0.6)
    ax2.set_title(f"NASA RUL: Permutation Importance (Drop in R²)\nSpearman Correlation = {comparison_nasa['spearman_rank_correlation']:.4f}", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Permutation Importance Mean")

    save_fig(fig, "04_feature_importance_comparison_shap_vs_permutation.png")

    # Figure 5: Deep Learning Temporal Attribution Heatmap (30 Time Steps x Top 12 Sensors)
    attr_mat_near_fail, _ = deep_exp.attribute_sequence(seq_near_fail, n_steps=25)
    top12_sensor_idx = [nasa_features.index(item["feature"]) for item in global_importance_nasa[:12]]
    top12_sensor_names = [item["feature"] for item in global_importance_nasa[:12]]

    heatmap_data = attr_mat_near_fail[:, top12_sensor_idx].T  # (12, 30)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap="coolwarm",
        center=0.0,
        xticklabels=range(1, 31),
        yticklabels=top12_sensor_names,
        cbar_kws={"label": "Integrated Gradients Attribution"}
    )
    ax.set_title("Deep Learning: Integrated Gradients Spatiotemporal Attribution Heatmap (Near Failure)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Operational Cycle in Temporal Window (1 to 30)")
    ax.set_ylabel("Sensor / Temporal Feature")
    save_fig(fig, "05_deep_learning_temporal_attribution_heatmap.png")

    # Figure 6: Historical Time Step Decay Profile
    fig, ax = plt.subplots(figsize=(9, 4.5))
    cycles_x = range(1, 31)
    ax.plot(cycles_x, time_decay_profile * 100, marker="o", color="#8e44ad", linewidth=2.0, label="Normalized Attribution Weight")
    ax.fill_between(cycles_x, 0, time_decay_profile * 100, color="#9b59b6", alpha=0.25)
    ax.axvspan(21, 30, color="#e74c3c", alpha=0.15, label=f"Recent Cycles (21-30): {summary_metrics['deep_learning_temporal']['recency_ratio']}x weight")
    ax.set_title("Temporal Decay Profile: Attribution Distribution Across Historical Cycles in Window", fontsize=11, fontweight="bold")
    ax.set_xlabel("Cycle within Observation Window (1 = Oldest, 30 = Most Recent)")
    ax.set_ylabel("Attribution Share (%)")
    ax.set_ylim(0, max(time_decay_profile * 100) * 1.25)
    ax.legend(loc="upper left")
    save_fig(fig, "06_time_step_decay_profile.png")

    # Figure 7: Lifecycle SHAP Progression Across Operating Cycles (Validation Engine #1)
    id_col = "unit_id" if "unit_id" in val_df.columns else "engine_id"
    eng1_mask = val_df[id_col] == val_df[id_col].unique()[0]
    eng1_df = val_df[eng1_mask].sort_values("cycle")
    eng1_X = eng1_df[nasa_features].to_numpy(dtype=np.float32)
    eng1_shap = nasa_explainer.compute_dataset_shap(eng1_X)

    fig, ax = plt.subplots(figsize=(11, 5))
    top4_lifecycle_feats = [item["feature"] for item in global_importance_nasa[:4]]
    palette = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    for idx, f_name in enumerate(top4_lifecycle_feats):
        f_idx = nasa_features.index(f_name)
        ax.plot(eng1_df["cycle"], eng1_shap[:, f_idx], label=f"SHAP: {f_name}", color=palette[idx], linewidth=2.0)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax.set_title(f"Engine #{eng1_df[id_col].iloc[0]}: SHAP Contribution Trajectory Across Lifecycle", fontsize=11, fontweight="bold")
    ax.set_xlabel("Operational Cycle")
    ax.set_ylabel("SHAP Impact on Predicted RUL (Cycles)")
    ax.legend(loc="best", fontsize=9)
    save_fig(fig, "07_lifecycle_shap_progression.png")

    # Figure 8: Model Explanation Comparison (XGBoost vs Random Forest vs LSTM Sensor Attribution)
    # Compute RF dataset SHAP
    rf_explainer = TreeSHAPExplainer(rf_nasa, nasa_features, task_type="regression", model_id="nasa_rul_rf")
    rf_importance = rf_explainer.get_global_importance(X_val_nasa, top_k=10)
    rf_imp_map = {item["feature"]: item["mean_abs_shap"] for item in rf_importance}

    # LSTM sensor attribution (sum of absolute IG across time steps for the dataset sample)
    lstm_sensor_attrs = []
    for i in range(min(50, len(X_val_seq))):
        a_mat, _ = deep_exp.attribute_sequence(X_val_seq[i], n_steps=15)
        lstm_sensor_attrs.append(np.sum(np.abs(a_mat), axis=0))
    lstm_mean_sensor_attr = np.mean(lstm_sensor_attrs, axis=0)
    lstm_imp_map = {nasa_features[i]: float(lstm_mean_sensor_attr[i]) for i in range(len(nasa_features))}

    top6_comp_features = [item["feature"] for item in global_importance_nasa[:6]]

    fig, ax = plt.subplots(figsize=(12, 5))
    x_pos = np.arange(len(top6_comp_features))
    width = 0.25

    # Normalize values for fair multi-model comparison
    xgb_vals = [item["mean_abs_shap"] for item in global_importance_nasa[:6]]
    xgb_norm = [v / max(xgb_vals) for v in xgb_vals]

    rf_vals = [rf_imp_map.get(f, 0.0) for f in top6_comp_features]
    rf_norm = [v / max(max(rf_vals), 1e-6) for v in rf_vals]

    lstm_vals = [lstm_imp_map.get(f, 0.0) for f in top6_comp_features]
    lstm_norm = [v / max(max(lstm_vals), 1e-6) for v in lstm_vals]

    ax.bar(x_pos - width, xgb_norm, width, label="XGBoost Regressor (SHAP)", color="#2980b9")
    ax.bar(x_pos, rf_norm, width, label="Random Forest Regressor (SHAP)", color="#27ae60")
    ax.bar(x_pos + width, lstm_norm, width, label="LSTM Regressor (Integrated Gradients)", color="#8e44ad")

    ax.set_title("Cross-Model Attribution Comparison on Top Degradation Features", fontsize=11, fontweight="bold")
    ax.set_ylabel("Normalized Feature Attribution Magnitude")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(top6_comp_features, rotation=15, ha="right", fontsize=9)
    ax.legend(loc="upper right")
    save_fig(fig, "08_model_explanation_comparison.png")

    # =========================================================================
    # PART 6: SERIALIZE EXPLAINABILITY ARTIFACTS
    # =========================================================================
    print("\n" + "=" * 80)
    print("--- 6. Serializing Explainability Artifacts ---")
    exp_dir = os.path.join(BASE_DIR, "models", "explainability")

    # Save NASA SHAP table
    with open(os.path.join(exp_dir, "nasa_shap_importance.json"), "w") as f:
        json.dump(global_importance_nasa, f, indent=4)

    # Save AI4I SHAP table
    with open(os.path.join(exp_dir, "ai4i_shap_importance.json"), "w") as f:
        json.dump(global_importance_ai4i, f, indent=4)

    # Save Permutation results
    with open(os.path.join(exp_dir, "permutation_importance.json"), "w") as f:
        json.dump({
            "nasa_permutation": perm_results_nasa[:20],
            "ai4i_permutation": perm_results_ai4i,
            "concordance_metrics": summary_metrics["permutation_comparison"]
        }, f, indent=4)

    # Save Deep Learning temporal decay
    with open(os.path.join(exp_dir, "deep_learning_temporal_profile.json"), "w") as f:
        json.dump(summary_metrics["deep_learning_temporal"], f, indent=4)

    # Save master summary metrics
    summary_metrics["execution_time_seconds"] = round(time.time() - t_start, 2)
    with open(os.path.join(exp_dir, "metrics_summary.json"), "w") as f:
        json.dump(summary_metrics, f, indent=4)

    print(f"  Saved explainability artifacts to: {exp_dir}")
    print("\n" + "=" * 80)
    print(f"PHASE 5 EXECUTION COMPLETE IN {summary_metrics['execution_time_seconds']} SECONDS")
    print("=" * 80)


if __name__ == "__main__":
    run_phase5()
