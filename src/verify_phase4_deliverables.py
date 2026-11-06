"""
Phase 4 Deliverables Verification Script.
Audits all Phase 4 models, threshold files, reports, figures, notebooks, and tests the predict_anomaly() API.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.anomaly_detection import predict_anomaly


def verify_phase4():
    print("=" * 80)
    print("PHASE 4 DELIVERABLES VERIFICATION AUDIT")
    print("=" * 80)

    errors = []

    # 1. Audit Model Artifacts
    print("\n1. Auditing Model Artifacts...")
    models_dir = os.path.join(BASE_DIR, "models", "anomaly_detection")
    expected_models = [
        os.path.join(models_dir, "lstm_autoencoder", "lstm_autoencoder_FD001.pt"),
        os.path.join(models_dir, "autoencoder", "dense_autoencoder_AI4I.pt"),
        os.path.join(models_dir, "isolation_forest", "isolation_forest_FD001.joblib"),
        os.path.join(models_dir, "isolation_forest", "isolation_forest_AI4I.joblib"),
        os.path.join(models_dir, "one_class_svm", "one_class_svm_FD001.joblib"),
        os.path.join(models_dir, "one_class_svm", "one_class_svm_AI4I.joblib"),
        os.path.join(models_dir, "lof", "lof_FD001.joblib"),
        os.path.join(models_dir, "lof", "lof_AI4I.joblib"),
        os.path.join(models_dir, "thresholds.json"),
        os.path.join(models_dir, "metadata.json"),
        os.path.join(models_dir, "experiment_metrics.json"),
    ]

    for p in expected_models:
        if os.path.exists(p) and os.path.getsize(p) > 0:
            print(f"  [OK] Found model artifact: {os.path.basename(p)} ({os.path.getsize(p):,} bytes)")
        else:
            errors.append(f"Missing or empty model artifact: {p}")
            print(f"  [FAIL] Missing artifact: {p}")

    # 2. Audit Figures
    print("\n2. Auditing Visualizations...")
    fig_dir = os.path.join(BASE_DIR, "reports", "phase4", "figures")
    expected_figs = [
        "01_nasa_anomaly_score_distribution.png",
        "02_nasa_engine_trajectories.png",
        "03_nasa_sensor_attribution_breakdown.png",
        "04_nasa_lstm_reconstruction_error.png",
        "05_ai4i_score_distributions_comparison.png",
        "06_ai4i_pr_roc_curves.png",
        "07_ai4i_confusion_matrices.png",
        "08_severity_tier_breakdown.png"
    ]

    for f in expected_figs:
        fig_path = os.path.join(fig_dir, f)
        if os.path.exists(fig_path) and os.path.getsize(fig_path) > 0:
            print(f"  [OK] Found figure: {f} ({os.path.getsize(fig_path):,} bytes)")
        else:
            errors.append(f"Missing or empty figure: {fig_path}")
            print(f"  [FAIL] Missing figure: {f}")

    # 3. Audit Technical Reports
    print("\n3. Auditing Technical Reports...")
    reports_dir = os.path.join(BASE_DIR, "reports", "phase4")
    expected_reports = [
        "anomaly_detection_report.md",
        "algorithm_comparison.md",
        "anomaly_threshold_report.md"
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
    nb_path = os.path.join(BASE_DIR, "notebooks", "13_multivariate_anomaly_detection.ipynb")
    if os.path.exists(nb_path) and os.path.getsize(nb_path) > 0:
        print(f"  [OK] Found notebook: {os.path.basename(nb_path)} ({os.path.getsize(nb_path):,} bytes)")
    else:
        errors.append(f"Missing or empty notebook: {nb_path}")
        print(f"  [FAIL] Missing notebook: {nb_path}")

    # 5. Test Unified Inference API `predict_anomaly`
    print("\n5. Testing Unified Inference API (predict_anomaly)...")
    val_npz = np.load(os.path.join(BASE_DIR, "data", "processed", "nasa", "FD001", "sequences_w30_val.npz"))
    nasa_seq = val_npz["X"][:1]  # (1, 30, 153)

    ai4i_csv = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "ai4i", "X_test.csv"))
    ai4i_row = ai4i_csv.iloc[0].values  # (14,)

    test_cases = [
        ("NASA FD001", "lstm_autoencoder", "FD001", nasa_seq),
        ("NASA FD001", "isolation_forest", "FD001", nasa_seq),
        ("NASA FD001", "one_class_svm", "FD001", nasa_seq),
        ("NASA FD001", "lof", "FD001", nasa_seq),
        ("AI4I 2020", "isolation_forest", "AI4I", ai4i_row),
        ("AI4I 2020", "dense_autoencoder", "AI4I", ai4i_row),
        ("AI4I 2020", "one_class_svm", "AI4I", ai4i_row),
        ("AI4I 2020", "lof", "AI4I", ai4i_row),
    ]

    for dataset_label, model_id, ds_key, sample_input in test_cases:
        try:
            res = predict_anomaly(sample_input, model_id=model_id, dataset=ds_key)
            required_keys = ["anomaly_score", "is_anomaly", "severity", "model_identifier", "relevant_sensor_metadata"]
            for k in required_keys:
                if k not in res:
                    raise KeyError(f"Missing key '{k}' in result dict")
            valid_severities = ["Normal", "Low anomaly", "Medium anomaly", "High anomaly"]
            if res["severity"] not in valid_severities:
                raise ValueError(f"Invalid severity '{res['severity']}'")

            score = res["anomaly_score"]
            status = "ANOMALOUS" if res["is_anomaly"] else "NORMAL"
            sev = res["severity"]
            print(f"  [OK] {dataset_label:10s} | {model_id:18s} -> Score: {score:8.4f} | Status: {status:9s} | Severity: {sev}")
        except Exception as e:
            errors.append(f"Inference failure for {dataset_label} {model_id}: {str(e)}")
            print(f"  [FAIL] Inference failed for {dataset_label} {model_id}: {str(e)}")

    print("\n" + "=" * 80)
    if errors:
        print(f"AUDIT FAILED WITH {len(errors)} ERROR(S):")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("ALL PHASE 4 DELIVERABLES VERIFIED SUCCESSFULLY! (100% PASS)")
        print("=" * 80)
        sys.exit(0)


if __name__ == "__main__":
    verify_phase4()
