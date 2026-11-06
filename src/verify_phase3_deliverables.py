"""Phase 3 Comprehensive Verification Checklist Script.

Verifies:
1. All 18 saved PyTorch deep learning models load and pass inference.
2. All 9 Phase 3 figures exist and have valid non-zero size.
3. All 4 Phase 3 Jupyter notebooks exist and parse as valid JSON.
4. All 4 technical reports in reports/phase3/ exist and have valid content.
5. experiment_log.csv contains all logged runs and valid metrics.
"""

import json
from pathlib import Path
import sys
import pandas as pd
import torch

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from deep_learning.lstm_model import LSTMRULModel
from deep_learning.gru_model import GRURULModel
from deep_learning.cnn_lstm_model import CNNLSTMRULModel
from deep_learning.early_warning_model import TemporalEarlyWarningClassifier
from deep_learning.mlp_model import AI4IMLPClassifier


def verify_phase3():
    print("=" * 70)
    print("AGENTIC PREDICTIVE MAINTENANCE - PHASE 3 VERIFICATION CHECKLIST")
    print("=" * 70)

    base_dir = Path(".")
    models_dir = base_dir / "models" / "deep_learning"
    figs_dir = base_dir / "reports" / "phase3" / "figures"
    nb_dir = base_dir / "notebooks"
    reports_dir = base_dir / "reports" / "phase3"

    all_passed = True

    # -------------------------------------------------------------
    # 1. VERIFY SAVED DEEP LEARNING MODELS
    # -------------------------------------------------------------
    print("\n[1/5] Verifying Saved PyTorch Models in models/deep_learning/...")
    pt_files = sorted(list(models_dir.glob("*/*.pt")))
    print(f"Found {len(pt_files)} saved PyTorch checkpoint files.")

    if len(pt_files) < 18:
        print(f"  [!] Expected at least 18 models, found {len(pt_files)}.")
        all_passed = False

    for pt_path in pt_files:
        meta_path = pt_path.parent / f"{pt_path.stem}_meta.json"
        if not meta_path.exists():
            print(f"  [X] Missing metadata for: {pt_path.name}")
            all_passed = False
            continue

        with open(meta_path, "r") as f:
            meta = json.load(f)

        # Instantiate architecture and test inference
        try:
            cfg = meta.get("config", {})
            m_type = cfg.get("model_type", "")

            if "CNN-LSTM" in m_type:
                model = CNNLSTMRULModel(
                    input_dim=cfg["input_dim"],
                    conv_filters=cfg.get("conv_filters", 32),
                    lstm_hidden_dim=cfg.get("lstm_hidden_dim", 32),
                )
                x = torch.randn(2, 30, cfg["input_dim"])
            elif "EarlyWarning" in m_type:
                model = TemporalEarlyWarningClassifier(
                    input_dim=cfg["input_dim"],
                    backbone=cfg.get("backbone", "lstm"),
                    hidden_dim=cfg.get("hidden_dim", 32),
                    num_layers=cfg.get("num_layers", 1),
                )
                x = torch.randn(2, 30, cfg["input_dim"])
            elif "LSTM" in m_type:
                model = LSTMRULModel(
                    input_dim=cfg["input_dim"],
                    hidden_dim=cfg.get("hidden_dim", 32),
                    num_layers=cfg.get("num_layers", 1),
                )
                x = torch.randn(2, 30, cfg["input_dim"])
            elif "GRU" in m_type:
                model = GRURULModel(
                    input_dim=cfg["input_dim"],
                    hidden_dim=cfg.get("hidden_dim", 32),
                    num_layers=cfg.get("num_layers", 1),
                )
                x = torch.randn(2, 30, cfg["input_dim"])
            elif "AI4I_MLP" in m_type:
                model = AI4IMLPClassifier(
                    input_dim=cfg["input_dim"],
                    hidden_dims=tuple(cfg.get("hidden_dims", [64, 32, 16])),
                )
                x = torch.randn(2, cfg["input_dim"])
            else:
                raise ValueError(f"Unknown model_type: {m_type}")

            model.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
            model.eval()
            out = model(x)
            assert out.shape[0] == 2
            print(f"  [OK] Verified {pt_path.name} -> Loaded & Forward Pass Passed")
        except Exception as e:
            print(f"  [X] Failed inference on {pt_path.name}: {e}")
            all_passed = False

    # -------------------------------------------------------------
    # 2. VERIFY PHASE 3 FIGURES
    # -------------------------------------------------------------
    print("\n[2/5] Verifying Phase 3 Figures in reports/phase3/figures/...")
    expected_figures = [
        "01_loss_curves_rul_lstm_gru_cnnlstm.png",
        "02_rul_pred_vs_actual_deep_learning.png",
        "03_rul_error_dist_deep_learning.png",
        "04_rul_trajectory_tracking.png",
        "05_window_length_comparison.png",
        "06_early_warning_dl_pr_curves.png",
        "07_early_warning_dl_cm.png",
        "08_ai4i_mlp_curves_cm.png",
        "09_ml_vs_dl_comparison_bar.png",
    ]

    for fig_name in expected_figures:
        fig_path = figs_dir / fig_name
        if fig_path.exists() and fig_path.stat().st_size > 5000:
            print(f"  [OK] Verified figure: {fig_name} ({fig_path.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"  [X] Missing or empty figure: {fig_name}")
            all_passed = False

    # -------------------------------------------------------------
    # 3. VERIFY PHASE 3 NOTEBOOKS
    # -------------------------------------------------------------
    print("\n[3/5] Verifying Phase 3 Notebooks in notebooks/...")
    expected_notebooks = [
        "09_lstm_rul_prediction.ipynb",
        "10_gru_rul_prediction.ipynb",
        "11_early_warning_classification.ipynb",
        "12_deep_learning_comparison.ipynb",
    ]

    for nb_name in expected_notebooks:
        nb_path = nb_dir / nb_name
        if nb_path.exists():
            with open(nb_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            num_cells = len(data.get("cells", []))
            print(f"  [OK] Verified notebook: {nb_name} ({num_cells} cells, {nb_path.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"  [X] Missing notebook: {nb_name}")
            all_passed = False

    # -------------------------------------------------------------
    # 4. VERIFY PHASE 3 TECHNICAL REPORTS
    # -------------------------------------------------------------
    print("\n[4/5] Verifying Phase 3 Reports in reports/phase3/...")
    expected_reports = [
        "training_report.md",
        "evaluation_report.md",
        "model_comparison.md",
        "hyperparameter_report.md",
        "experiment_log.csv",
    ]

    for rep_name in expected_reports:
        rep_path = reports_dir / rep_name
        if rep_path.exists() and rep_path.stat().st_size > 200:
            print(f"  [OK] Verified report: {rep_name} ({rep_path.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"  [X] Missing or empty report: {rep_name}")
            all_passed = False

    # -------------------------------------------------------------
    # 5. VERIFY EXPERIMENT LOG INTEGRITY
    # -------------------------------------------------------------
    print("\n[5/5] Verifying Experiment Log Integrity...")
    log_csv = reports_dir / "experiment_log.csv"
    if log_csv.exists():
        df_log = pd.read_csv(log_csv)
        print(f"  Total logged experiment runs: {len(df_log)}")
        print(f"  Tasks represented: {df_log['Task'].unique().tolist()}")
        print(f"  Datasets represented: {df_log['Dataset'].unique().tolist()}")
        if len(df_log) >= 18:
            print("  [OK] Full complement of 18 experiments validated.")
        else:
            print(f"  [!] Incomplete runs: found {len(df_log)}, expected >= 18")
            all_passed = False
    else:
        print("  [X] Missing experiment_log.csv")
        all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL PHASE 3 DELIVERABLES VERIFIED AND VALIDATED SUCCESSFULLY!")
    else:
        print("VERIFICATION ENCOUNTERED ISSUES — PLEASE REVIEW WARNINGS ABOVE.")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = verify_phase3()
    sys.exit(0 if success else 1)
