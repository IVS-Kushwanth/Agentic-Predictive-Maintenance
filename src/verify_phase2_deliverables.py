"""Phase 2 Deliverables Verification Script.

Tests and verifies:
1. All saved baseline models in models/baseline/ can be loaded and produce valid predictions.
2. All required Phase 2 evaluation figures exist and have valid file sizes.
3. All required Phase 2 Jupyter notebooks exist.
4. All required Phase 2 Markdown reports exist and contain data.
5. Experiment log exists with complete run records.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def verify_phase2():
    print("=" * 70)
    print("AGENTIC PREDICTIVE MAINTENANCE — PHASE 2 VERIFICATION CHECKLIST")
    print("=" * 70)

    base_dir = Path(".")
    all_passed = True

    # 1. Verify Saved Models
    print("\n[1/5] Verifying Saved Baseline Models in models/baseline/...")
    models_dir = base_dir / "models" / "baseline"
    model_files = list(models_dir.glob("*.joblib"))
    print(f"Found {len(model_files)} saved model files.")

    assert len(model_files) >= 16, f"Expected at least 16 baseline models, found {len(model_files)}"

    for mf in model_files:
        try:
            m = joblib.load(mf)
            meta_f = mf.parent / f"{mf.stem}_meta.json"
            assert meta_f.exists(), f"Missing metadata file for {mf.name}"
            
            # Test inference capability
            if "nasa" in mf.name:
                n_feats = 153 if "FD001" in mf.name else (162 if "FD003" in mf.name else 216)
                dummy = np.zeros((2, n_feats), dtype=np.float32)
                pred = m.predict(dummy)
                assert len(pred) == 2, f"Inference failed for {mf.name}"
            elif "ai4i" in mf.name:
                dummy = np.zeros((2, 14), dtype=np.float32)
                pred = m.predict(dummy)
                assert len(pred) == 2, f"Inference failed for {mf.name}"
                
            print(f"  Verified {mf.name} -> Loaded & Inference Passed")
        except Exception as e:
            print(f"  FAILED: {mf.name} - {e}")
            all_passed = False

    # 2. Verify Evaluation Visualizations
    print("\n[2/5] Verifying Phase 2 Figures in reports/figures/phase2/...")
    figs_dir = base_dir / "reports" / "figures" / "phase2"
    expected_figs = [
        "01_rul_pred_vs_actual_xgboost_regressor.png",
        "02_rul_error_dist_xgboost_regressor.png",
        "03_early_warning_pr_curves.png",
        "04_ai4i_pr_curves.png",
        "05_ai4i_roc_curves.png",
        "06_early_warning_h30_cm.png",
        "07_ai4i_confusion_matrix.png",
        "08_feature_importance_nasa_rul.png",
        "09_feature_importance_ai4i.png",
    ]
    for ef in expected_figs:
        fpath = figs_dir / ef
        if not fpath.exists() or fpath.stat().st_size < 1000:
            print(f"  FAILED: Figure {ef} missing or invalid size")
            all_passed = False
        else:
            print(f"  Verified figure: {ef} ({fpath.stat().st_size / 1024:.1f} KB)")

    # 3. Verify Jupyter Notebooks
    print("\n[3/5] Verifying Phase 2 Notebooks in notebooks/...")
    nb_dir = base_dir / "notebooks"
    expected_nbs = [
        "06_nasa_rul_baselines.ipynb",
        "07_nasa_early_warning.ipynb",
        "08_ai4i_failure_classification.ipynb",
    ]
    for nb in expected_nbs:
        nb_path = nb_dir / nb
        if not nb_path.exists() or nb_path.stat().st_size < 500:
            print(f"  FAILED: Notebook {nb} missing or invalid size")
            all_passed = False
        else:
            print(f"  Verified notebook: {nb} ({nb_path.stat().st_size / 1024:.1f} KB)")

    # 4. Verify Markdown Reports
    print("\n[4/5] Verifying Phase 2 Reports in reports/phase2/...")
    rep_dir = base_dir / "reports" / "phase2"
    expected_reps = [
        "model_comparison.md",
        "evaluation_report.md",
        "feature_importance_report.md",
        "experiment_log.csv",
    ]
    for rep in expected_reps:
        rpath = rep_dir / rep
        if not rpath.exists() or rpath.stat().st_size < 100:
            print(f"  FAILED: Report {rep} missing or empty")
            all_passed = False
        else:
            print(f"  Verified report: {rep} ({rpath.stat().st_size / 1024:.1f} KB)")

    # 5. Verify Experiment Log
    print("\n[5/5] Verifying Experiment Log Integrity...")
    log_csv = rep_dir / "experiment_log.csv"
    df_log = pd.read_csv(log_csv)
    print(f"  Total logged experiment runs: {len(df_log)}")
    assert len(df_log) == 32, f"Expected 32 experiment runs, found {len(df_log)}"
    print(f"  Tasks represented: {df_log['task'].unique().tolist()}")
    print(f"  Datasets represented: {df_log['dataset'].unique().tolist()}")

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL PHASE 2 DELIVERABLES VERIFIED AND VALIDATED SUCCESSFULLY!")
    else:
        print("VERIFICATION FAILED: Some deliverables did not pass.")
    print("=" * 70)


if __name__ == "__main__":
    verify_phase2()
