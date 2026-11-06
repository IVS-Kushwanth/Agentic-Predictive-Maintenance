"""Phase 3 Artifact & Figure Generation Script.

Loads saved PyTorch checkpoints and metadata from models/deep_learning/,
generates publication figures 01 to 09, and writes reports/phase3/experiment_log.csv.
Uses standard ASCII characters for cross-platform Windows compatibility.
"""

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

# Ensure src in module search path
sys.path.insert(0, str(Path(__file__).parent))

from deep_learning.dataset_loader import (
    get_cmapss_dataloaders,
    get_ai4i_dataloaders,
)
from deep_learning.lstm_model import LSTMRULModel
from deep_learning.gru_model import GRURULModel
from deep_learning.cnn_lstm_model import CNNLSTMRULModel
from deep_learning.early_warning_model import TemporalEarlyWarningClassifier
from deep_learning.mlp_model import AI4IMLPClassifier
from deep_learning.evaluate import evaluate_cmapss_rul, evaluate_classification_model
from deep_learning.visualization import (
    plot_loss_curves,
    plot_rul_pred_vs_actual,
    plot_rul_error_distribution,
    plot_trajectory_tracking,
    plot_window_length_comparison,
    plot_early_warning_pr_curves,
    plot_confusion_matrix,
    plot_ai4i_curves_and_cm,
    plot_ml_vs_dl_comparison,
)
from sklearn.metrics import precision_recall_curve


def generate_all_artifacts():
    print("=" * 70)
    print("GENERATING PHASE 3 FIGURES AND EXPERIMENT LOG")
    print("=" * 70)

    device = "cpu"
    base_models_dir = Path("models/deep_learning")
    fig_dirs = [Path("reports/phase3/figures"), Path("reports/figures/phase3")]
    exp_log_path = Path("reports/phase3/experiment_log.csv")
    exp_log_path.parent.mkdir(parents=True, exist_ok=True)

    experiment_records = []

    # 1. Load FD001 w=30 models
    fd001_w30 = get_cmapss_dataloaders("FD001", window_length=30, batch_size=256, val_batch_size=256)
    in_dim_fd001 = fd001_w30["num_features"]

    # LSTM
    lstm_model = LSTMRULModel(input_dim=in_dim_fd001, hidden_dim=32, num_layers=1, fc_dim=32)
    lstm_model.load_state_dict(torch.load(base_models_dir / "lstm" / "nasa_rul_lstm_FD001_w30.pt", map_location=device, weights_only=True))
    with open(base_models_dir / "lstm" / "nasa_rul_lstm_FD001_w30_meta.json", "r") as f:
        lstm_meta = json.load(f)
    lstm_eval = evaluate_cmapss_rul(
        lstm_model, fd001_w30["val_loader"], fd001_w30["test_loader"],
        fd001_w30["val_data"]["y_rul"], fd001_w30["test_data"]["y_rul"],
        fd001_w30["terminal_test_indices"], device=device
    )
    experiment_records.append({
        "Model Name": "LSTM Regressor",
        "Architecture": "LSTM (1-layer, h=32)",
        "Dataset": "NASA C-MAPSS",
        "Subset": "FD001",
        "Window Length": 30,
        "Task": "RUL Regression",
        "Val RMSE": lstm_eval["val_metrics"]["val_rmse"],
        "Val MAE": lstm_eval["val_metrics"]["val_mae"],
        "Test Terminal RMSE": lstm_eval["test_terminal_metrics"]["test_terminal_rmse"],
        "Test Terminal MAE": lstm_eval["test_terminal_metrics"]["test_terminal_mae"],
        "Test Terminal NASA Score": lstm_eval["test_terminal_metrics"]["test_terminal_nasa_score"],
        "Test All RMSE": lstm_eval["test_all_metrics"]["test_all_rmse"],
        "Params": lstm_eval["model_size"]["total_parameters"],
        "Train Time (s)": lstm_meta["train"]["train_duration_sec"],
    })

    # GRU
    gru_model = GRURULModel(input_dim=in_dim_fd001, hidden_dim=32, num_layers=1, fc_dim=32)
    gru_model.load_state_dict(torch.load(base_models_dir / "gru" / "nasa_rul_gru_FD001_w30.pt", map_location=device, weights_only=True))
    with open(base_models_dir / "gru" / "nasa_rul_gru_FD001_w30_meta.json", "r") as f:
        gru_meta = json.load(f)
    gru_eval = evaluate_cmapss_rul(
        gru_model, fd001_w30["val_loader"], fd001_w30["test_loader"],
        fd001_w30["val_data"]["y_rul"], fd001_w30["test_data"]["y_rul"],
        fd001_w30["terminal_test_indices"], device=device
    )
    experiment_records.append({
        "Model Name": "GRU Regressor",
        "Architecture": "GRU (1-layer, h=32)",
        "Dataset": "NASA C-MAPSS",
        "Subset": "FD001",
        "Window Length": 30,
        "Task": "RUL Regression",
        "Val RMSE": gru_eval["val_metrics"]["val_rmse"],
        "Val MAE": gru_eval["val_metrics"]["val_mae"],
        "Test Terminal RMSE": gru_eval["test_terminal_metrics"]["test_terminal_rmse"],
        "Test Terminal MAE": gru_eval["test_terminal_metrics"]["test_terminal_mae"],
        "Test Terminal NASA Score": gru_eval["test_terminal_metrics"]["test_terminal_nasa_score"],
        "Test All RMSE": gru_eval["test_all_metrics"]["test_all_rmse"],
        "Params": gru_eval["model_size"]["total_parameters"],
        "Train Time (s)": gru_meta["train"]["train_duration_sec"],
    })

    # CNN-LSTM
    cnn_model = CNNLSTMRULModel(input_dim=in_dim_fd001, conv_filters=32, kernel_size=3, lstm_hidden_dim=32, fc_dim=32)
    cnn_model.load_state_dict(torch.load(base_models_dir / "cnn_lstm" / "nasa_rul_cnn_lstm_FD001_w30.pt", map_location=device, weights_only=True))
    with open(base_models_dir / "cnn_lstm" / "nasa_rul_cnn_lstm_FD001_w30_meta.json", "r") as f:
        cnn_meta = json.load(f)
    cnn_eval = evaluate_cmapss_rul(
        cnn_model, fd001_w30["val_loader"], fd001_w30["test_loader"],
        fd001_w30["val_data"]["y_rul"], fd001_w30["test_data"]["y_rul"],
        fd001_w30["terminal_test_indices"], device=device
    )
    experiment_records.append({
        "Model Name": "CNN-LSTM Regressor",
        "Architecture": "Conv1D(32) + LSTM(32)",
        "Dataset": "NASA C-MAPSS",
        "Subset": "FD001",
        "Window Length": 30,
        "Task": "RUL Regression",
        "Val RMSE": cnn_eval["val_metrics"]["val_rmse"],
        "Val MAE": cnn_eval["val_metrics"]["val_mae"],
        "Test Terminal RMSE": cnn_eval["test_terminal_metrics"]["test_terminal_rmse"],
        "Test Terminal MAE": cnn_eval["test_terminal_metrics"]["test_terminal_mae"],
        "Test Terminal NASA Score": cnn_eval["test_terminal_metrics"]["test_terminal_nasa_score"],
        "Test All RMSE": cnn_eval["test_all_metrics"]["test_all_rmse"],
        "Params": cnn_eval["model_size"]["total_parameters"],
        "Train Time (s)": cnn_meta["train"]["train_duration_sec"],
    })

    # 2. Window Length: w=50 and w=60
    fd001_w50 = get_cmapss_dataloaders("FD001", window_length=50, batch_size=256, val_batch_size=256)
    lstm_w50 = LSTMRULModel(input_dim=fd001_w50["num_features"], hidden_dim=32, num_layers=1)
    lstm_w50.load_state_dict(torch.load(base_models_dir / "lstm" / "nasa_rul_lstm_FD001_w50.pt", map_location=device, weights_only=True))
    with open(base_models_dir / "lstm" / "nasa_rul_lstm_FD001_w50_meta.json", "r") as f:
        w50_meta = json.load(f)
    w50_eval = evaluate_cmapss_rul(
        lstm_w50, fd001_w50["val_loader"], fd001_w50["test_loader"],
        fd001_w50["val_data"]["y_rul"], fd001_w50["test_data"]["y_rul"],
        fd001_w50["terminal_test_indices"], device=device
    )
    experiment_records.append({
        "Model Name": "LSTM Regressor (w=50)",
        "Architecture": "LSTM (1-layer, h=32)",
        "Dataset": "NASA C-MAPSS",
        "Subset": "FD001",
        "Window Length": 50,
        "Task": "RUL Regression",
        "Val RMSE": w50_eval["val_metrics"]["val_rmse"],
        "Val MAE": w50_eval["val_metrics"]["val_mae"],
        "Test Terminal RMSE": w50_eval["test_terminal_metrics"]["test_terminal_rmse"],
        "Test Terminal MAE": w50_eval["test_terminal_metrics"]["test_terminal_mae"],
        "Test Terminal NASA Score": w50_eval["test_terminal_metrics"]["test_terminal_nasa_score"],
        "Test All RMSE": w50_eval["test_all_metrics"]["test_all_rmse"],
        "Params": w50_eval["model_size"]["total_parameters"],
        "Train Time (s)": w50_meta["train"]["train_duration_sec"],
    })

    fd001_w60 = get_cmapss_dataloaders("FD001", window_length=60, batch_size=256, val_batch_size=256)
    lstm_w60 = LSTMRULModel(input_dim=fd001_w60["num_features"], hidden_dim=32, num_layers=1)
    lstm_w60.load_state_dict(torch.load(base_models_dir / "lstm" / "nasa_rul_lstm_FD001_w60.pt", map_location=device, weights_only=True))
    with open(base_models_dir / "lstm" / "nasa_rul_lstm_FD001_w60_meta.json", "r") as f:
        w60_meta = json.load(f)
    w60_eval = evaluate_cmapss_rul(
        lstm_w60, fd001_w60["val_loader"], fd001_w60["test_loader"],
        fd001_w60["val_data"]["y_rul"], fd001_w60["test_data"]["y_rul"],
        fd001_w60["terminal_test_indices"], device=device
    )
    experiment_records.append({
        "Model Name": "LSTM Regressor (w=60)",
        "Architecture": "LSTM (1-layer, h=32)",
        "Dataset": "NASA C-MAPSS",
        "Subset": "FD001",
        "Window Length": 60,
        "Task": "RUL Regression",
        "Val RMSE": w60_eval["val_metrics"]["val_rmse"],
        "Val MAE": w60_eval["val_metrics"]["val_mae"],
        "Test Terminal RMSE": w60_eval["test_terminal_metrics"]["test_terminal_rmse"],
        "Test Terminal MAE": w60_eval["test_terminal_metrics"]["test_terminal_mae"],
        "Test Terminal NASA Score": w60_eval["test_terminal_metrics"]["test_terminal_nasa_score"],
        "Test All RMSE": w60_eval["test_all_metrics"]["test_all_rmse"],
        "Params": w60_eval["model_size"]["total_parameters"],
        "Train Time (s)": w60_meta["train"]["train_duration_sec"],
    })

    # 3. Subsets FD002, FD003, FD004
    for sub in ["FD002", "FD003", "FD004"]:
        bs = 512 if sub in ["FD002", "FD004"] else 256
        sub_loader = get_cmapss_dataloaders(sub, window_length=30, batch_size=bs, val_batch_size=bs)
        in_d = sub_loader["num_features"]

        # LSTM
        sub_lstm = LSTMRULModel(input_dim=in_d, hidden_dim=32, num_layers=1)
        sub_lstm.load_state_dict(torch.load(base_models_dir / "lstm" / f"nasa_rul_lstm_{sub}_w30.pt", map_location=device, weights_only=True))
        with open(base_models_dir / "lstm" / f"nasa_rul_lstm_{sub}_w30_meta.json", "r") as f:
            sub_lstm_meta = json.load(f)
        sub_lstm_ev = evaluate_cmapss_rul(
            sub_lstm, sub_loader["val_loader"], sub_loader["test_loader"],
            sub_loader["val_data"]["y_rul"], sub_loader["test_data"]["y_rul"],
            sub_loader["terminal_test_indices"], device=device
        )
        experiment_records.append({
            "Model Name": f"LSTM Regressor ({sub})",
            "Architecture": "LSTM (1-layer, h=32)",
            "Dataset": "NASA C-MAPSS",
            "Subset": sub,
            "Window Length": 30,
            "Task": "RUL Regression",
            "Val RMSE": sub_lstm_ev["val_metrics"]["val_rmse"],
            "Val MAE": sub_lstm_ev["val_metrics"]["val_mae"],
            "Test Terminal RMSE": sub_lstm_ev["test_terminal_metrics"]["test_terminal_rmse"],
            "Test Terminal MAE": sub_lstm_ev["test_terminal_metrics"]["test_terminal_mae"],
            "Test Terminal NASA Score": sub_lstm_ev["test_terminal_metrics"]["test_terminal_nasa_score"],
            "Test All RMSE": sub_lstm_ev["test_all_metrics"]["test_all_rmse"],
            "Params": sub_lstm_ev["model_size"]["total_parameters"],
            "Train Time (s)": sub_lstm_meta["train"]["train_duration_sec"],
        })

        # GRU
        sub_gru = GRURULModel(input_dim=in_d, hidden_dim=32, num_layers=1)
        sub_gru.load_state_dict(torch.load(base_models_dir / "gru" / f"nasa_rul_gru_{sub}_w30.pt", map_location=device, weights_only=True))
        with open(base_models_dir / "gru" / f"nasa_rul_gru_{sub}_w30_meta.json", "r") as f:
            sub_gru_meta = json.load(f)
        sub_gru_ev = evaluate_cmapss_rul(
            sub_gru, sub_loader["val_loader"], sub_loader["test_loader"],
            sub_loader["val_data"]["y_rul"], sub_loader["test_data"]["y_rul"],
            sub_loader["terminal_test_indices"], device=device
        )
        experiment_records.append({
            "Model Name": f"GRU Regressor ({sub})",
            "Architecture": "GRU (1-layer, h=32)",
            "Dataset": "NASA C-MAPSS",
            "Subset": sub,
            "Window Length": 30,
            "Task": "RUL Regression",
            "Val RMSE": sub_gru_ev["val_metrics"]["val_rmse"],
            "Val MAE": sub_gru_ev["val_metrics"]["val_mae"],
            "Test Terminal RMSE": sub_gru_ev["test_terminal_metrics"]["test_terminal_rmse"],
            "Test Terminal MAE": sub_gru_ev["test_terminal_metrics"]["test_terminal_mae"],
            "Test Terminal NASA Score": sub_gru_ev["test_terminal_metrics"]["test_terminal_nasa_score"],
            "Test All RMSE": sub_gru_ev["test_all_metrics"]["test_all_rmse"],
            "Params": sub_gru_ev["model_size"]["total_parameters"],
            "Train Time (s)": sub_gru_meta["train"]["train_duration_sec"],
        })

    # 4. Early Warning
    ew_pr_curves_data = {}
    last_ew_cm = None
    last_ew_title = ""

    for h in [10, 20, 30]:
        ew_loaders = get_cmapss_dataloaders("FD001", window_length=30, target_type=f"ew{h}", batch_size=256, val_batch_size=256)
        in_d = ew_loaders["num_features"]

        # LSTM
        l_ew = TemporalEarlyWarningClassifier(input_dim=in_d, backbone="lstm", hidden_dim=32, num_layers=1)
        l_ew.load_state_dict(torch.load(base_models_dir / "early_warning" / f"nasa_ew_{h}_lstm_FD001.pt", map_location=device, weights_only=True))
        with open(base_models_dir / "early_warning" / f"nasa_ew_{h}_lstm_FD001_meta.json", "r") as f:
            l_ew_meta = json.load(f)
        l_ew_ev = evaluate_classification_model(
            l_ew, ew_loaders["val_loader"], ew_loaders["test_loader"],
            ew_loaders["val_data"][f"y_ew{h}"], ew_loaders["test_data"][f"y_ew{h}"],
            terminal_test_indices=ew_loaders["terminal_test_indices"], device=device
        )
        experiment_records.append({
            "Model Name": f"LSTM Early Warning (H={h})",
            "Architecture": "LSTM Classifier (h=32)",
            "Dataset": "NASA C-MAPSS",
            "Subset": "FD001",
            "Window Length": 30,
            "Task": f"Early Warning (H={h})",
            "Threshold": l_ew_ev["optimal_threshold"],
            "Test Precision": l_ew_ev["test_metrics"]["test_precision"],
            "Test Recall": l_ew_ev["test_metrics"]["test_recall"],
            "Test F1": l_ew_ev["test_metrics"]["test_f1"],
            "Test PR-AUC": l_ew_ev["test_metrics"]["test_pr_auc"],
            "Test ROC-AUC": l_ew_ev["test_metrics"]["test_roc_auc"],
            "Params": l_ew_ev["model_size"]["total_parameters"],
            "Train Time (s)": l_ew_meta["train"]["train_duration_sec"],
        })

        # GRU
        g_ew = TemporalEarlyWarningClassifier(input_dim=in_d, backbone="gru", hidden_dim=32, num_layers=1)
        g_ew.load_state_dict(torch.load(base_models_dir / "early_warning" / f"nasa_ew_{h}_gru_FD001.pt", map_location=device, weights_only=True))
        with open(base_models_dir / "early_warning" / f"nasa_ew_{h}_gru_FD001_meta.json", "r") as f:
            g_ew_meta = json.load(f)
        g_ew_ev = evaluate_classification_model(
            g_ew, ew_loaders["val_loader"], ew_loaders["test_loader"],
            ew_loaders["val_data"][f"y_ew{h}"], ew_loaders["test_data"][f"y_ew{h}"],
            terminal_test_indices=ew_loaders["terminal_test_indices"], device=device
        )
        experiment_records.append({
            "Model Name": f"GRU Early Warning (H={h})",
            "Architecture": "GRU Classifier (h=32)",
            "Dataset": "NASA C-MAPSS",
            "Subset": "FD001",
            "Window Length": 30,
            "Task": f"Early Warning (H={h})",
            "Threshold": g_ew_ev["optimal_threshold"],
            "Test Precision": g_ew_ev["test_metrics"]["test_precision"],
            "Test Recall": g_ew_ev["test_metrics"]["test_recall"],
            "Test F1": g_ew_ev["test_metrics"]["test_f1"],
            "Test PR-AUC": g_ew_ev["test_metrics"]["test_pr_auc"],
            "Test ROC-AUC": g_ew_ev["test_metrics"]["test_roc_auc"],
            "Params": g_ew_ev["model_size"]["total_parameters"],
            "Train Time (s)": g_ew_meta["train"]["train_duration_sec"],
        })

        p_l, r_l, _ = precision_recall_curve(ew_loaders["test_data"][f"y_ew{h}"], l_ew_ev["test_probs"])
        p_g, r_g, _ = precision_recall_curve(ew_loaders["test_data"][f"y_ew{h}"], g_ew_ev["test_probs"])
        ew_pr_curves_data[f"LSTM H={h}"] = (r_l, p_l, l_ew_ev["test_metrics"]["test_pr_auc"])
        ew_pr_curves_data[f"GRU H={h}"] = (r_g, p_g, g_ew_ev["test_metrics"]["test_pr_auc"])

        if h == 30:
            last_ew_cm = np.array(l_ew_ev["test_metrics"]["test_confusion_matrix"])
            last_ew_title = f"NASA FD001 Early Warning Confusion Matrix (LSTM H=30, Th={l_ew_ev['optimal_threshold']:.2f})"

    # 5. AI4I MLP
    ai4i_loaders = get_ai4i_dataloaders(val_ratio=0.2, batch_size=128, random_state=42)
    ai4i_mlp = AI4IMLPClassifier(input_dim=ai4i_loaders["input_dim"], hidden_dims=(64, 32, 16), dropout=0.25)
    ai4i_mlp.load_state_dict(torch.load(base_models_dir / "ai4i_mlp" / "ai4i_clf_deep_mlp.pt", map_location=device, weights_only=True))
    with open(base_models_dir / "ai4i_mlp" / "ai4i_clf_deep_mlp_meta.json", "r") as f:
        ai4i_meta = json.load(f)
    ai4i_ev = evaluate_classification_model(
        ai4i_mlp, ai4i_loaders["val_loader"], ai4i_loaders["test_loader"],
        ai4i_loaders["y_val"], ai4i_loaders["y_test"], device=device
    )
    experiment_records.append({
        "Model Name": "Deep Tabular MLP",
        "Architecture": "MLP (64-32-16 BatchNorm + Dropout)",
        "Dataset": "UCI AI4I 2020",
        "Subset": "Full",
        "Window Length": 1,
        "Task": "Machine Failure Classification",
        "Threshold": ai4i_ev["optimal_threshold"],
        "Test Precision": ai4i_ev["test_metrics"]["test_precision"],
        "Test Recall": ai4i_ev["test_metrics"]["test_recall"],
        "Test F1": ai4i_ev["test_metrics"]["test_f1"],
        "Test PR-AUC": ai4i_ev["test_metrics"]["test_pr_auc"],
        "Test ROC-AUC": ai4i_ev["test_metrics"]["test_roc_auc"],
        "Params": ai4i_ev["model_size"]["total_parameters"],
        "Train Time (s)": ai4i_meta["train"]["train_duration_sec"],
    })

    # 6. Trajectory tracking
    sample_engine_ids = [31, 49, 81]
    engine_trajectories = []
    test_uids = fd001_w30["test_data"]["engine_ids"]
    test_cycles = fd001_w30["test_data"]["end_cycles"]
    test_true = fd001_w30["test_data"]["y_rul"]
    test_lstm_p = lstm_eval["y_test_all_pred"]
    test_gru_p = gru_eval["y_test_all_pred"]

    for uid in sample_engine_ids:
        mask = (test_uids == uid)
        if np.any(mask):
            engine_trajectories.append({
                "unit_id": uid,
                "cycles": test_cycles[mask],
                "true_rul": test_true[mask],
                "lstm_rul": test_lstm_p[mask],
                "gru_rul": test_gru_p[mask],
            })

    # 7. Generate Figures (01 to 09)
    print("Generating Figure 01: Loss curves...")
    plot_loss_curves(
        {
            "LSTM (FD001)": lstm_meta["train"]["history"],
            "GRU (FD001)": gru_meta["train"]["history"],
            "CNN-LSTM (FD001)": cnn_meta["train"]["history"],
        },
        fig_dirs, filename="01_loss_curves_rul_lstm_gru_cnnlstm.png"
    )
    print("  [OK] Saved 01_loss_curves_rul_lstm_gru_cnnlstm.png")

    print("Generating Figure 02: Predicted vs Actual RUL...")
    plot_rul_pred_vs_actual(
        {
            "LSTM Regressor": (lstm_eval["y_test_terminal_true"], lstm_eval["y_test_terminal_pred"]),
            "GRU Regressor": (gru_eval["y_test_terminal_true"], gru_eval["y_test_terminal_pred"]),
            "CNN-LSTM Regressor": (cnn_eval["y_test_terminal_true"], cnn_eval["y_test_terminal_pred"]),
        },
        fig_dirs, filename="02_rul_pred_vs_actual_deep_learning.png"
    )
    print("  [OK] Saved 02_rul_pred_vs_actual_deep_learning.png")

    print("Generating Figure 03: Error distributions...")
    plot_rul_error_distribution(
        {
            "LSTM": (lstm_eval["y_test_terminal_true"], lstm_eval["y_test_terminal_pred"]),
            "GRU": (gru_eval["y_test_terminal_true"], gru_eval["y_test_terminal_pred"]),
            "CNN-LSTM": (cnn_eval["y_test_terminal_true"], cnn_eval["y_test_terminal_pred"]),
        },
        fig_dirs, filename="03_rul_error_dist_deep_learning.png"
    )
    print("  [OK] Saved 03_rul_error_dist_deep_learning.png")

    print("Generating Figure 04: Trajectory tracking...")
    plot_trajectory_tracking(engine_trajectories, fig_dirs, filename="04_rul_trajectory_tracking.png")
    print("  [OK] Saved 04_rul_trajectory_tracking.png")

    print("Generating Figure 05: Window length comparison...")
    plot_window_length_comparison(
        {
            "L=30 Cycles": {
                "rmse": lstm_eval["test_terminal_metrics"]["test_terminal_rmse"],
                "mae": lstm_eval["test_terminal_metrics"]["test_terminal_mae"],
                "nasa_score": lstm_eval["test_terminal_metrics"]["test_terminal_nasa_score"],
            },
            "L=50 Cycles": {
                "rmse": w50_eval["test_terminal_metrics"]["test_terminal_rmse"],
                "mae": w50_eval["test_terminal_metrics"]["test_terminal_mae"],
                "nasa_score": w50_eval["test_terminal_metrics"]["test_terminal_nasa_score"],
            },
            "L=60 Cycles": {
                "rmse": w60_eval["test_terminal_metrics"]["test_terminal_rmse"],
                "mae": w60_eval["test_terminal_metrics"]["test_terminal_mae"],
                "nasa_score": w60_eval["test_terminal_metrics"]["test_terminal_nasa_score"],
            },
        },
        fig_dirs, filename="05_window_length_comparison.png"
    )
    print("  [OK] Saved 05_window_length_comparison.png")

    print("Generating Figure 06: Early warning PR curves...")
    plot_early_warning_pr_curves(ew_pr_curves_data, fig_dirs, filename="06_early_warning_dl_pr_curves.png")
    print("  [OK] Saved 06_early_warning_dl_pr_curves.png")

    print("Generating Figure 07: Early warning confusion matrix...")
    if last_ew_cm is not None:
        plot_confusion_matrix(last_ew_cm, last_ew_title, fig_dirs, filename="07_early_warning_dl_cm.png")
        print("  [OK] Saved 07_early_warning_dl_cm.png")

    print("Generating Figure 08: AI4I curves and confusion matrix...")
    ai4i_cm = np.array(ai4i_ev["test_metrics"]["test_confusion_matrix"])
    plot_ai4i_curves_and_cm(
        ai4i_loaders["y_test"], ai4i_ev["test_probs"], ai4i_cm,
        ai4i_ev["optimal_threshold"], fig_dirs, filename="08_ai4i_mlp_curves_cm.png"
    )
    print("  [OK] Saved 08_ai4i_mlp_curves_cm.png")

    print("Generating Figure 09: ML vs DL comparison...")
    plot_ml_vs_dl_comparison(
        {
            "Linear Reg (ML)": {"rmse": 26.54, "mae": 19.57, "nasa_score": 11181.9, "paradigm": "ML"},
            "Random Forest (ML)": {"rmse": 24.31, "mae": 17.65, "nasa_score": 22754.7, "paradigm": "ML"},
            "Gradient Boost (ML)": {"rmse": 22.50, "mae": 16.09, "nasa_score": 3202.4, "paradigm": "ML"},
            "XGBoost (ML)": {"rmse": 22.50, "mae": 15.35, "nasa_score": 9525.5, "paradigm": "ML"},
            "LSTM (DL)": {
                "rmse": lstm_eval["test_terminal_metrics"]["test_terminal_rmse"],
                "mae": lstm_eval["test_terminal_metrics"]["test_terminal_mae"],
                "nasa_score": lstm_eval["test_terminal_metrics"]["test_terminal_nasa_score"],
                "paradigm": "DL",
            },
            "GRU (DL)": {
                "rmse": gru_eval["test_terminal_metrics"]["test_terminal_rmse"],
                "mae": gru_eval["test_terminal_metrics"]["test_terminal_mae"],
                "nasa_score": gru_eval["test_terminal_metrics"]["test_terminal_nasa_score"],
                "paradigm": "DL",
            },
            "CNN-LSTM (DL)": {
                "rmse": cnn_eval["test_terminal_metrics"]["test_terminal_rmse"],
                "mae": cnn_eval["test_terminal_metrics"]["test_terminal_mae"],
                "nasa_score": cnn_eval["test_terminal_metrics"]["test_terminal_nasa_score"],
                "paradigm": "DL",
            },
        },
        fig_dirs, filename="09_ml_vs_dl_comparison_bar.png"
    )
    print("  [OK] Saved 09_ml_vs_dl_comparison_bar.png")

    # 8. Save CSV experiment log
    df_log = pd.DataFrame(experiment_records)
    df_log.to_csv(exp_log_path, index=False)
    print(f"[OK] Saved Phase 3 experiment log to: {exp_log_path} ({len(df_log)} records)")


if __name__ == "__main__":
    generate_all_artifacts()
