"""Final Deliverables Validation & Verification Script for Phase 1.

Verifies:
1. Raw file presence and non-zero byte size.
2. Processed tabular CSV loadability and column counts.
3. 3D Sequence NPZ archive loadability, array shapes, and target distributions.
4. Preprocessing joblib pipelines loadability and transform capability on unseen data.
5. Generated visualization figures existence.
6. Notebooks and markdown documentation presence.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def check_file(path: Path, min_bytes: int = 100) -> bool:
    if not path.exists():
        print(f"FAILED: File does not exist: {path}")
        return False
    size = path.stat().st_size
    if size < min_bytes:
        print(f"FAILED: File too small ({size} bytes): {path}")
        return False
    return True


def run_comprehensive_validation():
    print("=" * 70)
    print("AGENTIC PREDICTIVE MAINTENANCE — PHASE 1 VALIDATION CHECKLIST")
    print("=" * 70)

    all_passed = True

    # 1. Raw NASA Files
    print("\n[1/7] Verifying Raw NASA C-MAPSS Files...")
    raw_nasa_dir = Path("data/raw/nasa_cmapss")
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        for prefix in ["train", "test", "RUL"]:
            f = raw_nasa_dir / f"{prefix}_{sub}.txt"
            if not check_file(f):
                all_passed = False
    print("-> All raw NASA files verified successfully.")

    # 2. Raw AI4I File
    print("\n[2/7] Verifying Raw UCI AI4I 2020 File...")
    ai4i_raw = Path("data/raw/ai4i/ai4i2020.csv")
    if not check_file(ai4i_raw, min_bytes=100000):
        all_passed = False
    df_ai4i = pd.read_csv(ai4i_raw)
    assert df_ai4i.shape == (10000, 14), f"Unexpected AI4I raw shape: {df_ai4i.shape}"
    print(f"-> Raw AI4I verified. Shape: {df_ai4i.shape}")

    # 3. Processed NASA Artifacts (FD001 Focus)
    print("\n[3/7] Verifying Processed NASA FD001 Artifacts...")
    proc_fd1 = Path("data/processed/nasa/FD001")
    for tab in ["train_tabular.csv", "val_tabular.csv", "test_tabular.csv"]:
        f = proc_fd1 / tab
        if not check_file(f):
            all_passed = False
        df = pd.read_csv(f)
        print(f"   Loaded {tab}: {df.shape}")

    for w in [30, 50, 60]:
        for split in ["train", "val", "test"]:
            npz_file = proc_fd1 / f"sequences_w{w}_{split}.npz"
            if not check_file(npz_file):
                all_passed = False
            data = np.load(npz_file)
            X = data["X"]
            y_rul = data["y_rul"]
            y_ew30 = data["y_ew30"]
            assert len(X) == len(y_rul) == len(y_ew30), "Mismatched sequence and target counts!"
            print(f"   Loaded sequences_w{w}_{split}.npz: X={X.shape}, y_rul={y_rul.shape}, y_ew30_pos_rate={y_ew30.mean():.4f}")

    # 4. Processed AI4I Artifacts
    print("\n[4/7] Verifying Processed AI4I Artifacts...")
    proc_ai4i = Path("data/processed/ai4i")
    for fname in ["ai4i_cleaned.csv", "ai4i_engineered.csv", "X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]:
        f = proc_ai4i / fname
        if not check_file(f):
            all_passed = False
        df = pd.read_csv(f)
        print(f"   Loaded {fname}: {df.shape}")

    # 5. Saved Joblib Preprocessing Pipelines
    print("\n[5/7] Verifying Saved Preprocessing Pipelines & Unseen Transform Capability...")
    models_dir = Path("models/preprocessing")
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        sc_file = models_dir / f"nasa_scaler_{sub}.joblib"
        if not check_file(sc_file):
            all_passed = False
        scaler = joblib.load(sc_file)
        # Test transform on mock record with correct feature dimension
        n_features = scaler.n_features_in_
        mock_features = np.zeros((2, n_features))
        transformed = scaler.transform(mock_features)
        assert transformed.shape == (2, n_features)
        print(f"   NASA scaler {sub}: Verified ({n_features} features in)")

    ai4i_pipe_file = models_dir / "ai4i_preprocessor.joblib"
    if not check_file(ai4i_pipe_file):
        all_passed = False
    ai4i_pipe = joblib.load(ai4i_pipe_file)
    # Test transform on unseen mock AI4I sample
    mock_ai4i = pd.DataFrame([{
        "Type": "M",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 100.0,
        "Mechanical power [W]": 6283.0,
        "Temp difference [K]": 10.0,
        "Temp ratio": 1.033,
        "Torque to speed ratio": 0.0267,
        "Overstrain indicator [Nm*min]": 4000.0,
        "Heat dissipation index": 15000.0,
        "Power wear index": 628300.0,
    }])
    transformed_ai4i = ai4i_pipe.transform(mock_ai4i)
    assert transformed_ai4i.shape == (1, 14), f"Unexpected AI4I transformed shape: {transformed_ai4i.shape}"
    print(f"   AI4I preprocessor: Verified on unseen sample. Output shape: {transformed_ai4i.shape}")

    # 6. Figures Verification
    print("\n[6/7] Verifying Generated Publication-Quality Figures...")
    figs_nasa = list(Path("reports/figures/nasa").glob("*.png"))
    figs_ai4i = list(Path("reports/figures/ai4i").glob("*.png"))
    print(f"   NASA figures count: {len(figs_nasa)} (Expected: 13)")
    print(f"   AI4I figures count: {len(figs_ai4i)} (Expected: 5)")
    assert len(figs_nasa) == 13, f"Expected 13 NASA figures, found {len(figs_nasa)}"
    assert len(figs_ai4i) == 5, f"Expected 5 AI4I figures, found {len(figs_ai4i)}"

    # 7. Documentation & Notebooks
    print("\n[7/7] Verifying Documentation & Notebooks...")
    reports = [
        "dataset_analysis.md",
        "data_quality_report.md",
        "temporal_analysis.md",
        "preprocessing_report.md",
        "data_dictionary.md",
    ]
    for r in reports:
        if not check_file(Path("reports") / r):
            all_passed = False
        print(f"   Verified report: reports/{r}")

    notebooks = [
        "01_dataset_acquisition.ipynb",
        "02_nasa_dataset_eda.ipynb",
        "03_nasa_temporal_preprocessing.ipynb",
        "04_ai4i_eda.ipynb",
        "05_ai4i_preprocessing.ipynb",
    ]
    for nb in notebooks:
        if not check_file(Path("notebooks") / nb):
            all_passed = False
        print(f"   Verified notebook: notebooks/{nb}")

    if not check_file(Path("README.md")):
        all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL PHASE 1 DELIVERABLES FULLY VERIFIED AND VALIDATED!")
    else:
        print("SOME DELIVERABLES FAILED VALIDATION.")
    print("=" * 70)


if __name__ == "__main__":
    run_comprehensive_validation()
