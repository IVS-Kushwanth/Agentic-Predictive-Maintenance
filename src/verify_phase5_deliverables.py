"""
Phase 5 Deliverables Verification Script.
Audits all Phase 5 explainability artifacts, figures, reports, notebook,
and executes functional tests on all standardized explainability API endpoints.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.explainability import (
    explain_prediction,
    get_global_feature_importance,
    get_temporal_explanation,
    get_sensor_contributions
)


def verify_phase5():
    print("=" * 80)
    print("PHASE 5 DELIVERABLES VERIFICATION AUDIT")
    print("=" * 80)

    errors = []

    # 1. Audit Explainability Artifacts
    print("\n1. Auditing Explainability Artifacts in models/explainability/...")
    exp_dir = os.path.join(BASE_DIR, "models", "explainability")
    expected_artifacts = [
        os.path.join(exp_dir, "nasa_shap_importance.json"),
        os.path.join(exp_dir, "ai4i_shap_importance.json"),
        os.path.join(exp_dir, "permutation_importance.json"),
        os.path.join(exp_dir, "deep_learning_temporal_profile.json"),
        os.path.join(exp_dir, "metrics_summary.json"),
    ]

    for p in expected_artifacts:
        if os.path.exists(p) and os.path.getsize(p) > 0:
            print(f"  [OK] Found artifact: {os.path.basename(p)} ({os.path.getsize(p):,} bytes)")
        else:
            errors.append(f"Missing or empty artifact: {p}")
            print(f"  [FAIL] Missing artifact: {p}")

    # 2. Audit Visualizations
    print("\n2. Auditing Visualizations in reports/phase5/figures/...")
    fig_dir = os.path.join(BASE_DIR, "reports", "phase5", "figures")
    expected_figs = [
        "01_shap_summary_nasa_rul.png",
        "02_shap_waterfall_nasa_individual.png",
        "03_shap_summary_ai4i_failure.png",
        "04_feature_importance_comparison_shap_vs_permutation.png",
        "05_deep_learning_temporal_attribution_heatmap.png",
        "06_time_step_decay_profile.png",
        "07_lifecycle_shap_progression.png",
        "08_model_explanation_comparison.png"
    ]

    for f in expected_figs:
        fig_path = os.path.join(fig_dir, f)
        if os.path.exists(fig_path) and os.path.getsize(fig_path) > 0:
            print(f"  [OK] Found figure: {f} ({os.path.getsize(fig_path):,} bytes)")
        else:
            errors.append(f"Missing or empty figure: {fig_path}")
            print(f"  [FAIL] Missing figure: {f}")

    # 3. Audit Technical Reports
    print("\n3. Auditing Technical Reports in reports/phase5/...")
    reports_dir = os.path.join(BASE_DIR, "reports", "phase5")
    expected_reports = [
        "explainable_ai_report.md",
        "shap_analysis.md",
        "feature_importance_comparison.md",
        "temporal_explanation_report.md"
    ]

    for r in expected_reports:
        rep_path = os.path.join(reports_dir, r)
        if os.path.exists(rep_path) and os.path.getsize(rep_path) > 0:
            print(f"  [OK] Found report: {r} ({os.path.getsize(rep_path):,} bytes)")
        else:
            errors.append(f"Missing or empty report: {rep_path}")
            print(f"  [FAIL] Missing report: {r}")

    # 4. Audit Jupyter Notebook
    print("\n4. Auditing Jupyter Notebook...")
    nb_path = os.path.join(BASE_DIR, "notebooks", "14_explainable_ai.ipynb")
    if os.path.exists(nb_path) and os.path.getsize(nb_path) > 0:
        print(f"  [OK] Found notebook: {os.path.basename(nb_path)} ({os.path.getsize(nb_path):,} bytes)")
    else:
        errors.append(f"Missing or empty notebook: {nb_path}")
        print(f"  [FAIL] Missing notebook: {nb_path}")

    # 5. Functional Testing of Standardized Explainability API
    print("\n5. Testing Reusable Public API Functions...")

    # Load test inputs
    val_tabular = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "nasa", "FD001", "val_tabular.csv"), nrows=5)
    with open(os.path.join(BASE_DIR, "models", "baseline", "nasa_rul_xgboost_regressor_FD001_meta.json"), "r") as f:
        meta = json.load(f)
    nasa_features = meta["features"]
    nasa_row = val_tabular[nasa_features].iloc[0].to_numpy(dtype=np.float32)

    ai4i_csv = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "ai4i", "X_test.csv"), nrows=5)
    ai4i_row = ai4i_csv.iloc[0].to_numpy(dtype=np.float32)

    val_npz = np.load(os.path.join(BASE_DIR, "data", "processed", "nasa", "FD001", "sequences_w30_val.npz"))
    nasa_seq = val_npz["X"][0]  # (30, 153)

    # Test 5.1: explain_prediction NASA XGBoost
    try:
        exp_nasa_xgb = explain_prediction(nasa_row, model_id="xgboost", dataset="FD001", top_k=5)
        for k in ["prediction", "important_features", "explanation_summary"]:
            if k not in exp_nasa_xgb:
                raise KeyError(f"Missing key '{k}' in result")
        print(f"  [OK] explain_prediction(NASA XGBoost) -> Pred: {exp_nasa_xgb['prediction']['predicted_rul']} cycles | Top driver: {exp_nasa_xgb['important_features'][0]['feature']}")
    except Exception as e:
        errors.append(f"explain_prediction NASA XGBoost failed: {str(e)}")
        print(f"  [FAIL] explain_prediction NASA XGBoost: {str(e)}")

    # Test 5.2: explain_prediction AI4I XGBoost
    try:
        exp_ai4i_xgb = explain_prediction(ai4i_row, model_id="xgboost", dataset="AI4I", top_k=5)
        for k in ["prediction", "important_features", "explanation_summary"]:
            if k not in exp_ai4i_xgb:
                raise KeyError(f"Missing key '{k}' in result")
        print(f"  [OK] explain_prediction(AI4I XGBoost) -> Prob: {exp_ai4i_xgb['prediction']['predicted_probability']} | Top driver: {exp_ai4i_xgb['important_features'][0]['feature']}")
    except Exception as e:
        errors.append(f"explain_prediction AI4I XGBoost failed: {str(e)}")
        print(f"  [FAIL] explain_prediction AI4I XGBoost: {str(e)}")

    # Test 5.3: explain_prediction NASA Deep Learning (LSTM)
    try:
        exp_lstm = explain_prediction(nasa_seq, model_id="lstm", dataset="FD001", top_k=5)
        for k in ["prediction", "important_features", "temporal_attribution", "explanation_summary"]:
            if k not in exp_lstm:
                raise KeyError(f"Missing key '{k}' in result")
        print(f"  [OK] explain_prediction(NASA LSTM) -> Pred: {exp_lstm['prediction']['predicted_rul']} cycles | Recency ratio: {exp_lstm['temporal_attribution']['recent_vs_early_weight_ratio']}x")
    except Exception as e:
        errors.append(f"explain_prediction NASA LSTM failed: {str(e)}")
        print(f"  [FAIL] explain_prediction NASA LSTM: {str(e)}")

    # Test 5.4: get_global_feature_importance
    try:
        global_imp = get_global_feature_importance(model_id="xgboost", dataset="FD001", top_k=5, sample_size=50)
        if len(global_imp) != 5:
            raise ValueError(f"Expected 5 features, got {len(global_imp)}")
        print(f"  [OK] get_global_feature_importance(NASA) -> Top 1: {global_imp[0]['feature']} ({global_imp[0]['mean_abs_shap']:.4f})")
    except Exception as e:
        errors.append(f"get_global_feature_importance failed: {str(e)}")
        print(f"  [FAIL] get_global_feature_importance: {str(e)}")

    # Test 5.5: get_temporal_explanation
    try:
        temp_exp = get_temporal_explanation(nasa_seq, model_id="lstm", dataset="FD001")
        if "time_step_weights" not in temp_exp or len(temp_exp["time_step_weights"]) != 30:
            raise ValueError(f"Invalid temporal weights length: {len(temp_exp.get('time_step_weights', []))}")
        print(f"  [OK] get_temporal_explanation(LSTM) -> Peak cycle: {temp_exp['peak_influence_cycle_in_window']} / 30")
    except Exception as e:
        errors.append(f"get_temporal_explanation failed: {str(e)}")
        print(f"  [FAIL] get_temporal_explanation: {str(e)}")

    # Test 5.6: get_sensor_contributions
    try:
        sensor_contribs = get_sensor_contributions(nasa_row, model_id="xgboost", dataset="FD001", top_k=3)
        if len(sensor_contribs) != 3:
            raise ValueError(f"Expected 3 contributions, got {len(sensor_contribs)}")
        print(f"  [OK] get_sensor_contributions(NASA) -> {len(sensor_contribs)} ranked items: {[c['feature'] for c in sensor_contribs]}")
    except Exception as e:
        errors.append(f"get_sensor_contributions failed: {str(e)}")
        print(f"  [FAIL] get_sensor_contributions: {str(e)}")

    print("\n" + "=" * 80)
    if errors:
        print(f"AUDIT FAILED WITH {len(errors)} ERROR(S):")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("ALL PHASE 5 DELIVERABLES VERIFIED SUCCESSFULLY! (100% PASS)")
        print("=" * 80)
        sys.exit(0)


if __name__ == "__main__":
    verify_phase5()
