"""Comprehensive Deep Learning Pipeline Execution Script for Phase 3.

Executes:
1. NASA C-MAPSS FD001 Primary Benchmarks (LSTM, GRU, CNN-LSTM at w=30).
2. Window length sensitivity experiments (w=50 and w=60).
3. NASA C-MAPSS Subsets FD002, FD003, FD004 extensions.
4. Early-warning classification experiments for horizons H in {10, 20, 30}.
5. UCI AI4I 2020 Deep Tabular MLP machine failure classification.
6. Multi-cycle degradation trajectory tracking for test engines.
7. Publication-quality figures (01 through 09).
8. Experiment tracking log and model checkpoint persistence.
"""

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import torch

# Ensure src in module search path
sys.path.insert(0, str(Path(__file__).parent))

from deep_learning.dataset_loader import (
    get_cmapss_dataloaders,
    get_ai4i_dataloaders,
    load_cmapss_sequence_split,
    get_terminal_test_indices,
)
from deep_learning.lstm_model import LSTMRULModel
from deep_learning.gru_model import GRURULModel
from deep_learning.cnn_lstm_model import CNNLSTMRULModel
from deep_learning.early_warning_model import TemporalEarlyWarningClassifier
from deep_learning.mlp_model import AI4IMLPClassifier
from deep_learning.train import set_seed, train_rul_regressor, train_binary_classifier
from deep_learning.evaluate import (
    evaluate_cmapss_rul,
    evaluate_classification_model,
    predict_dataset,
    get_model_size_info,
)
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
from sklearn.metrics import precision_recall_curve, average_precision_score


def save_dl_model(
    model: torch.nn.Module,
    model_name: str,
    output_dir: Path,
    metadata: Dict[str, Any],
) -> Tuple[Path, Path]:
    """Persist PyTorch model state_dict and associated metadata JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pt_path = output_dir / f"{model_name}.pt"
    json_path = output_dir / f"{model_name}_meta.json"

    torch.save(model.state_dict(), pt_path)
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return pt_path, json_path


def run_phase3_pipeline() -> None:
    """Execute complete Phase 3 deep learning experiments."""
    print("=" * 80, flush=True)
    print("STARTING PHASE 3: DEEP LEARNING MODEL DEVELOPMENT & BENCHMARKING", flush=True)
    print("=" * 80, flush=True)

    device = "cpu"
    set_seed(42)

    # Output paths
    base_models_dir = Path("models/deep_learning")
    fig_dirs = [Path("reports/phase3/figures"), Path("reports/figures/phase3")]
    exp_log_path = Path("reports/phase3/experiment_log.csv")
    exp_log_path.parent.mkdir(parents=True, exist_ok=True)

    experiment_records: List[Dict[str, Any]] = []

    # =========================================================================
    # PART 1: NASA FD001 RUL DEEP LEARNING BENCHMARKS (w=30)
    # =========================================================================
    print("\n" + "#" * 70, flush=True)
    print("PART 1: NASA C-MAPSS FD001 RUL REGRESSION BENCHMARK (L=30)", flush=True)
    print("#" * 70, flush=True)

    fd001_w30 = get_cmapss_dataloaders("FD001", window_length=30, batch_size=256, val_batch_size=256)
    in_dim_fd001 = fd001_w30["num_features"]

    # 1.1 LSTM Regressor
    print("\n--- Training Model 1: LSTM Regressor (FD001, w=30) ---", flush=True)
    lstm_model = LSTMRULModel(input_dim=in_dim_fd001, hidden_dim=32, num_layers=1, dropout=0.2, fc_dim=32)
    lstm_train_res = train_rul_regressor(
        lstm_model, fd001_w30["train_loader"], fd001_w30["val_loader"],
        epochs=18, lr=2e-3, patience=5, device=device, verbose=True
    )
    lstm_eval_res = evaluate_cmapss_rul(
        lstm_model, fd001_w30["val_loader"], fd001_w30["test_loader"],
        fd001_w30["val_data"]["y_rul"], fd001_w30["test_data"]["y_rul"],
        fd001_w30["terminal_test_indices"], device=device
    )
    save_dl_model(
        lstm_model, "nasa_rul_lstm_FD001_w30", base_models_dir / "lstm",
        {
            "config": lstm_model.get_config(),
            "train": lstm_train_res,
            "val_metrics": lstm_eval_res["val_metrics"],
            "test_terminal_metrics": lstm_eval_res["test_terminal_metrics"],
            "test_all_metrics": lstm_eval_res["test_all_metrics"],
            "size": lstm_eval_res["model_size"],
        }
    )
    experiment_records.append({
        "Model Name": "LSTM Regressor",
        "Architecture": "LSTM (1-layer, h=32)",
        "Dataset": "NASA C-MAPSS",
        "Subset": "FD001",
        "Window Length": 30,
        "Task": "RUL Regression",
        "Val RMSE": lstm_eval_res["val_metrics"]["val_rmse"],
        "Val MAE": lstm_eval_res["val_metrics"]["val_mae"],
        "Test Terminal RMSE": lstm_eval_res["test_terminal_metrics"]["test_terminal_rmse"],
        "Test Terminal MAE": lstm_eval_res["test_terminal_metrics"]["test_terminal_mae"],
        "Test Terminal NASA Score": lstm_eval_res["test_terminal_metrics"]["test_terminal_nasa_score"],
        "Test All RMSE": lstm_eval_res["test_all_metrics"]["test_all_rmse"],
        "Params": lstm_eval_res["model_size"]["total_parameters"],
        "Train Time (s)": lstm_train_res["train_duration_sec"],
    })
    print(f"--> LSTM FD001 (w=30) Test Terminal RMSE: {lstm_eval_res['test_terminal_metrics']['test_terminal_rmse']:.2f} | MAE: {lstm_eval_res['test_terminal_metrics']['test_terminal_mae']:.2f} | NASA Score: {lstm_eval_res['test_terminal_metrics']['test_terminal_nasa_score']:.1f}", flush=True)

    # 1.2 GRU Regressor
    print("\n--- Training Model 2: GRU Regressor (FD001, w=30) ---", flush=True)
    gru_model = GRURULModel(input_dim=in_dim_fd001, hidden_dim=32, num_layers=1, dropout=0.2, fc_dim=32)
    gru_train_res = train_rul_regressor(
        gru_model, fd001_w30["train_loader"], fd001_w30["val_loader"],
        epochs=18, lr=2e-3, patience=5, device=device, verbose=True
    )
    gru_eval_res = evaluate_cmapss_rul(
        gru_model, fd001_w30["val_loader"], fd001_w30["test_loader"],
        fd001_w30["val_data"]["y_rul"], fd001_w30["test_data"]["y_rul"],
        fd001_w30["terminal_test_indices"], device=device
    )
    save_dl_model(
        gru_model, "nasa_rul_gru_FD001_w30", base_models_dir / "gru",
        {
            "config": gru_model.get_config(),
            "train": gru_train_res,
            "val_metrics": gru_eval_res["val_metrics"],
            "test_terminal_metrics": gru_eval_res["test_terminal_metrics"],
            "test_all_metrics": gru_eval_res["test_all_metrics"],
            "size": gru_eval_res["model_size"],
        }
    )
    experiment_records.append({
        "Model Name": "GRU Regressor",
        "Architecture": "GRU (1-layer, h=32)",
        "Dataset": "NASA C-MAPSS",
        "Subset": "FD001",
        "Window Length": 30,
        "Task": "RUL Regression",
        "Val RMSE": gru_eval_res["val_metrics"]["val_rmse"],
        "Val MAE": gru_eval_res["val_metrics"]["val_mae"],
        "Test Terminal RMSE": gru_eval_res["test_terminal_metrics"]["test_terminal_rmse"],
        "Test Terminal MAE": gru_eval_res["test_terminal_metrics"]["test_terminal_mae"],
        "Test Terminal NASA Score": gru_eval_res["test_terminal_metrics"]["test_terminal_nasa_score"],
        "Test All RMSE": gru_eval_res["test_all_metrics"]["test_all_rmse"],
        "Params": gru_eval_res["model_size"]["total_parameters"],
        "Train Time (s)": gru_train_res["train_duration_sec"],
    })
    print(f"--> GRU FD001 (w=30) Test Terminal RMSE: {gru_eval_res['test_terminal_metrics']['test_terminal_rmse']:.2f} | MAE: {gru_eval_res['test_terminal_metrics']['test_terminal_mae']:.2f} | NASA Score: {gru_eval_res['test_terminal_metrics']['test_terminal_nasa_score']:.1f}", flush=True)

    # 1.3 CNN-LSTM Regressor
    print("\n--- Training Model 3: CNN-LSTM Regressor (FD001, w=30) ---", flush=True)
    cnn_lstm_model = CNNLSTMRULModel(input_dim=in_dim_fd001, conv_filters=32, kernel_size=3, lstm_hidden_dim=32, lstm_layers=1, fc_dim=32)
    cnn_train_res = train_rul_regressor(
        cnn_lstm_model, fd001_w30["train_loader"], fd001_w30["val_loader"],
        epochs=18, lr=2e-3, patience=5, device=device, verbose=True
    )
    cnn_eval_res = evaluate_cmapss_rul(
        cnn_lstm_model, fd001_w30["val_loader"], fd001_w30["test_loader"],
        fd001_w30["val_data"]["y_rul"], fd001_w30["test_data"]["y_rul"],
        fd001_w30["terminal_test_indices"], device=device
    )
    save_dl_model(
        cnn_lstm_model, "nasa_rul_cnn_lstm_FD001_w30", base_models_dir / "cnn_lstm",
        {
            "config": cnn_lstm_model.get_config(),
            "train": cnn_train_res,
            "val_metrics": cnn_eval_res["val_metrics"],
            "test_terminal_metrics": cnn_eval_res["test_terminal_metrics"],
            "test_all_metrics": cnn_eval_res["test_all_metrics"],
            "size": cnn_eval_res["model_size"],
        }
    )
    experiment_records.append({
        "Model Name": "CNN-LSTM Regressor",
        "Architecture": "Conv1D(32) + LSTM(32)",
        "Dataset": "NASA C-MAPSS",
        "Subset": "FD001",
        "Window Length": 30,
        "Task": "RUL Regression",
        "Val RMSE": cnn_eval_res["val_metrics"]["val_rmse"],
        "Val MAE": cnn_eval_res["val_metrics"]["val_mae"],
        "Test Terminal RMSE": cnn_eval_res["test_terminal_metrics"]["test_terminal_rmse"],
        "Test Terminal MAE": cnn_eval_res["test_terminal_metrics"]["test_terminal_mae"],
        "Test Terminal NASA Score": cnn_eval_res["test_terminal_metrics"]["test_terminal_nasa_score"],
        "Test All RMSE": cnn_eval_res["test_all_metrics"]["test_all_rmse"],
        "Params": cnn_eval_res["model_size"]["total_parameters"],
        "Train Time (s)": cnn_train_res["train_duration_sec"],
    })
    print(f"--> CNN-LSTM FD001 (w=30) Test Terminal RMSE: {cnn_eval_res['test_terminal_metrics']['test_terminal_rmse']:.2f} | MAE: {cnn_eval_res['test_terminal_metrics']['test_terminal_mae']:.2f} | NASA Score: {cnn_eval_res['test_terminal_metrics']['test_terminal_nasa_score']:.1f}", flush=True)

    # =========================================================================
    # PART 2: WINDOW LENGTH SENSITIVITY ANALYSIS (w=50 and w=60)
    # =========================================================================
    print("\n" + "#" * 70, flush=True)
    print("PART 2: WINDOW LENGTH SENSITIVITY EXPERIMENTS (L=50, L=60)", flush=True)
    print("#" * 70, flush=True)

    # 2.1 LSTM (w=50)
    print("\n--- Training Model 4: LSTM Regressor (FD001, w=50) ---", flush=True)
    fd001_w50 = get_cmapss_dataloaders("FD001", window_length=50, batch_size=256, val_batch_size=256)
    lstm_w50 = LSTMRULModel(input_dim=fd001_w50["num_features"], hidden_dim=32, num_layers=1, dropout=0.2)
    w50_train_res = train_rul_regressor(
        lstm_w50, fd001_w50["train_loader"], fd001_w50["val_loader"],
        epochs=18, lr=2e-3, patience=5, device=device, verbose=True
    )
    w50_eval_res = evaluate_cmapss_rul(
        lstm_w50, fd001_w50["val_loader"], fd001_w50["test_loader"],
        fd001_w50["val_data"]["y_rul"], fd001_w50["test_data"]["y_rul"],
        fd001_w50["terminal_test_indices"], device=device
    )
    save_dl_model(
        lstm_w50, "nasa_rul_lstm_FD001_w50", base_models_dir / "lstm",
        {
            "config": lstm_w50.get_config(),
            "train": w50_train_res,
            "val_metrics": w50_eval_res["val_metrics"],
            "test_terminal_metrics": w50_eval_res["test_terminal_metrics"],
            "test_all_metrics": w50_eval_res["test_all_metrics"],
            "size": w50_eval_res["model_size"],
        }
    )
    experiment_records.append({
        "Model Name": "LSTM Regressor (w=50)",
        "Architecture": "LSTM (1-layer, h=32)",
        "Dataset": "NASA C-MAPSS",
        "Subset": "FD001",
        "Window Length": 50,
        "Task": "RUL Regression",
        "Val RMSE": w50_eval_res["val_metrics"]["val_rmse"],
        "Val MAE": w50_eval_res["val_metrics"]["val_mae"],
        "Test Terminal RMSE": w50_eval_res["test_terminal_metrics"]["test_terminal_rmse"],
        "Test Terminal MAE": w50_eval_res["test_terminal_metrics"]["test_terminal_mae"],
        "Test Terminal NASA Score": w50_eval_res["test_terminal_metrics"]["test_terminal_nasa_score"],
        "Test All RMSE": w50_eval_res["test_all_metrics"]["test_all_rmse"],
        "Params": w50_eval_res["model_size"]["total_parameters"],
        "Train Time (s)": w50_train_res["train_duration_sec"],
    })

    # 2.2 LSTM (w=60)
    print("\n--- Training Model 5: LSTM Regressor (FD001, w=60) ---", flush=True)
    fd001_w60 = get_cmapss_dataloaders("FD001", window_length=60, batch_size=256, val_batch_size=256)
    lstm_w60 = LSTMRULModel(input_dim=fd001_w60["num_features"], hidden_dim=32, num_layers=1, dropout=0.2)
    w60_train_res = train_rul_regressor(
        lstm_w60, fd001_w60["train_loader"], fd001_w60["val_loader"],
        epochs=18, lr=2e-3, patience=5, device=device, verbose=True
    )
    w60_eval_res = evaluate_cmapss_rul(
        lstm_w60, fd001_w60["val_loader"], fd001_w60["test_loader"],
        fd001_w60["val_data"]["y_rul"], fd001_w60["test_data"]["y_rul"],
        fd001_w60["terminal_test_indices"], device=device
    )
    save_dl_model(
        lstm_w60, "nasa_rul_lstm_FD001_w60", base_models_dir / "lstm",
        {
            "config": lstm_w60.get_config(),
            "train": w60_train_res,
            "val_metrics": w60_eval_res["val_metrics"],
            "test_terminal_metrics": w60_eval_res["test_terminal_metrics"],
            "test_all_metrics": w60_eval_res["test_all_metrics"],
            "size": w60_eval_res["model_size"],
        }
    )
    experiment_records.append({
        "Model Name": "LSTM Regressor (w=60)",
        "Architecture": "LSTM (1-layer, h=32)",
        "Dataset": "NASA C-MAPSS",
        "Subset": "FD001",
        "Window Length": 60,
        "Task": "RUL Regression",
        "Val RMSE": w60_eval_res["val_metrics"]["val_rmse"],
        "Val MAE": w60_eval_res["val_metrics"]["val_mae"],
        "Test Terminal RMSE": w60_eval_res["test_terminal_metrics"]["test_terminal_rmse"],
        "Test Terminal MAE": w60_eval_res["test_terminal_metrics"]["test_terminal_mae"],
        "Test Terminal NASA Score": w60_eval_res["test_terminal_metrics"]["test_terminal_nasa_score"],
        "Test All RMSE": w60_eval_res["test_all_metrics"]["test_all_rmse"],
        "Params": w60_eval_res["model_size"]["total_parameters"],
        "Train Time (s)": w60_train_res["train_duration_sec"],
    })

    # =========================================================================
    # PART 3: EXTENSION TO FD002, FD003, FD004 (w=30)
    # =========================================================================
    print("\n" + "#" * 70, flush=True)
    print("PART 3: EXTENSION TO C-MAPSS SUBSETS FD002, FD003, FD004", flush=True)
    print("#" * 70, flush=True)

    for subset in ["FD002", "FD003", "FD004"]:
        print(f"\n--- Processing Subset: {subset} (w=30) ---", flush=True)
        bs = 512 if subset in ["FD002", "FD004"] else 256
        sub_loader = get_cmapss_dataloaders(subset, window_length=30, batch_size=bs, val_batch_size=bs)
        in_dim_sub = sub_loader["num_features"]

        # LSTM
        print(f"Training LSTM for {subset}...", flush=True)
        sub_lstm = LSTMRULModel(input_dim=in_dim_sub, hidden_dim=32, num_layers=1, dropout=0.2)
        sub_lstm_train = train_rul_regressor(
            sub_lstm, sub_loader["train_loader"], sub_loader["val_loader"],
            epochs=15, lr=2e-3, patience=4, device=device, verbose=False
        )
        sub_lstm_eval = evaluate_cmapss_rul(
            sub_lstm, sub_loader["val_loader"], sub_loader["test_loader"],
            sub_loader["val_data"]["y_rul"], sub_loader["test_data"]["y_rul"],
            sub_loader["terminal_test_indices"], device=device
        )
        save_dl_model(
            sub_lstm, f"nasa_rul_lstm_{subset}_w30", base_models_dir / "lstm",
            {
                "config": sub_lstm.get_config(),
                "train": sub_lstm_train,
                "val_metrics": sub_lstm_eval["val_metrics"],
                "test_terminal_metrics": sub_lstm_eval["test_terminal_metrics"],
                "test_all_metrics": sub_lstm_eval["test_all_metrics"],
                "size": sub_lstm_eval["model_size"],
            }
        )
        experiment_records.append({
            "Model Name": f"LSTM Regressor ({subset})",
            "Architecture": "LSTM (1-layer, h=32)",
            "Dataset": "NASA C-MAPSS",
            "Subset": subset,
            "Window Length": 30,
            "Task": "RUL Regression",
            "Val RMSE": sub_lstm_eval["val_metrics"]["val_rmse"],
            "Val MAE": sub_lstm_eval["val_metrics"]["val_mae"],
            "Test Terminal RMSE": sub_lstm_eval["test_terminal_metrics"]["test_terminal_rmse"],
            "Test Terminal MAE": sub_lstm_eval["test_terminal_metrics"]["test_terminal_mae"],
            "Test Terminal NASA Score": sub_lstm_eval["test_terminal_metrics"]["test_terminal_nasa_score"],
            "Test All RMSE": sub_lstm_eval["test_all_metrics"]["test_all_rmse"],
            "Params": sub_lstm_eval["model_size"]["total_parameters"],
            "Train Time (s)": sub_lstm_train["train_duration_sec"],
        })
        print(f"  {subset} LSTM Test Terminal RMSE: {sub_lstm_eval['test_terminal_metrics']['test_terminal_rmse']:.2f} | NASA Score: {sub_lstm_eval['test_terminal_metrics']['test_terminal_nasa_score']:.1f}", flush=True)

        # GRU
        print(f"Training GRU for {subset}...", flush=True)
        sub_gru = GRURULModel(input_dim=in_dim_sub, hidden_dim=32, num_layers=1, dropout=0.2)
        sub_gru_train = train_rul_regressor(
            sub_gru, sub_loader["train_loader"], sub_loader["val_loader"],
            epochs=15, lr=2e-3, patience=4, device=device, verbose=False
        )
        sub_gru_eval = evaluate_cmapss_rul(
            sub_gru, sub_loader["val_loader"], sub_loader["test_loader"],
            sub_loader["val_data"]["y_rul"], sub_loader["test_data"]["y_rul"],
            sub_loader["terminal_test_indices"], device=device
        )
        save_dl_model(
            sub_gru, f"nasa_rul_gru_{subset}_w30", base_models_dir / "gru",
            {
                "config": sub_gru.get_config(),
                "train": sub_gru_train,
                "val_metrics": sub_gru_eval["val_metrics"],
                "test_terminal_metrics": sub_gru_eval["test_terminal_metrics"],
                "test_all_metrics": sub_gru_eval["test_all_metrics"],
                "size": sub_gru_eval["model_size"],
            }
        )
        experiment_records.append({
            "Model Name": f"GRU Regressor ({subset})",
            "Architecture": "GRU (1-layer, h=32)",
            "Dataset": "NASA C-MAPSS",
            "Subset": subset,
            "Window Length": 30,
            "Task": "RUL Regression",
            "Val RMSE": sub_gru_eval["val_metrics"]["val_rmse"],
            "Val MAE": sub_gru_eval["val_metrics"]["val_mae"],
            "Test Terminal RMSE": sub_gru_eval["test_terminal_metrics"]["test_terminal_rmse"],
            "Test Terminal MAE": sub_gru_eval["test_terminal_metrics"]["test_terminal_mae"],
            "Test Terminal NASA Score": sub_gru_eval["test_terminal_metrics"]["test_terminal_nasa_score"],
            "Test All RMSE": sub_gru_eval["test_all_metrics"]["test_all_rmse"],
            "Params": sub_gru_eval["model_size"]["total_parameters"],
            "Train Time (s)": sub_gru_train["train_duration_sec"],
        })
        print(f"  {subset} GRU Test Terminal RMSE: {sub_gru_eval['test_terminal_metrics']['test_terminal_rmse']:.2f} | NASA Score: {sub_gru_eval['test_terminal_metrics']['test_terminal_nasa_score']:.1f}", flush=True)

    # =========================================================================
    # PART 4: EARLY WARNING CLASSIFICATION (H in {10, 20, 30})
    # =========================================================================
    print("\n" + "#" * 70, flush=True)
    print("PART 4: TEMPORAL EARLY-WARNING CLASSIFICATION (FD001, w=30)", flush=True)
    print("#" * 70, flush=True)

    ew_pr_curves_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}
    last_ew_cm = None
    last_ew_title = ""

    for horizon in [10, 20, 30]:
        target_name = f"ew{horizon}"
        print(f"\n--- Early Warning Horizon: {horizon} Cycles ---", flush=True)
        ew_loaders = get_cmapss_dataloaders("FD001", window_length=30, target_type=target_name, batch_size=256, val_batch_size=256)
        in_dim = ew_loaders["num_features"]

        y_tr = ew_loaders["train_data"][f"y_{target_name}"]
        num_pos = int(np.sum(y_tr == 1))
        num_neg = int(np.sum(y_tr == 0))
        pos_weight = float(num_neg / max(num_pos, 1))

        # LSTM Classifier
        lstm_ew = TemporalEarlyWarningClassifier(input_dim=in_dim, backbone="lstm", hidden_dim=32, num_layers=1)
        lstm_ew_train = train_binary_classifier(
            lstm_ew, ew_loaders["train_loader"], ew_loaders["val_loader"],
            pos_weight=pos_weight, epochs=15, lr=2e-3, patience=4, device=device, verbose=False
        )
        lstm_ew_eval = evaluate_classification_model(
            lstm_ew, ew_loaders["val_loader"], ew_loaders["test_loader"],
            ew_loaders["val_data"][f"y_{target_name}"], ew_loaders["test_data"][f"y_{target_name}"],
            terminal_test_indices=ew_loaders["terminal_test_indices"], device=device
        )
        save_dl_model(
            lstm_ew, f"nasa_ew_{horizon}_lstm_FD001", base_models_dir / "early_warning",
            {
                "config": lstm_ew.get_config(),
                "horizon": horizon,
                "train": lstm_ew_train,
                "optimal_threshold": lstm_ew_eval["optimal_threshold"],
                "val_metrics": lstm_ew_eval["val_metrics"],
                "test_metrics": lstm_ew_eval["test_metrics"],
                "test_terminal_metrics": lstm_ew_eval["test_terminal_metrics"],
                "size": lstm_ew_eval["model_size"],
            }
        )
        experiment_records.append({
            "Model Name": f"LSTM Early Warning (H={horizon})",
            "Architecture": "LSTM Classifier (h=32)",
            "Dataset": "NASA C-MAPSS",
            "Subset": "FD001",
            "Window Length": 30,
            "Task": f"Early Warning (H={horizon})",
            "Threshold": lstm_ew_eval["optimal_threshold"],
            "Test Precision": lstm_ew_eval["test_metrics"]["test_precision"],
            "Test Recall": lstm_ew_eval["test_metrics"]["test_recall"],
            "Test F1": lstm_ew_eval["test_metrics"]["test_f1"],
            "Test PR-AUC": lstm_ew_eval["test_metrics"]["test_pr_auc"],
            "Test ROC-AUC": lstm_ew_eval["test_metrics"]["test_roc_auc"],
            "Params": lstm_ew_eval["model_size"]["total_parameters"],
            "Train Time (s)": lstm_ew_train["train_duration_sec"],
        })
        print(f"  LSTM (H={horizon}) Test PR-AUC: {lstm_ew_eval['test_metrics']['test_pr_auc']:.4f} | F1: {lstm_ew_eval['test_metrics']['test_f1']:.4f}", flush=True)

        # GRU Classifier
        gru_ew = TemporalEarlyWarningClassifier(input_dim=in_dim, backbone="gru", hidden_dim=32, num_layers=1)
        gru_ew_train = train_binary_classifier(
            gru_ew, ew_loaders["train_loader"], ew_loaders["val_loader"],
            pos_weight=pos_weight, epochs=15, lr=2e-3, patience=4, device=device, verbose=False
        )
        gru_ew_eval = evaluate_classification_model(
            gru_ew, ew_loaders["val_loader"], ew_loaders["test_loader"],
            ew_loaders["val_data"][f"y_{target_name}"], ew_loaders["test_data"][f"y_{target_name}"],
            terminal_test_indices=ew_loaders["terminal_test_indices"], device=device
        )
        save_dl_model(
            gru_ew, f"nasa_ew_{horizon}_gru_FD001", base_models_dir / "early_warning",
            {
                "config": gru_ew.get_config(),
                "horizon": horizon,
                "train": gru_ew_train,
                "optimal_threshold": gru_ew_eval["optimal_threshold"],
                "val_metrics": gru_ew_eval["val_metrics"],
                "test_metrics": gru_ew_eval["test_metrics"],
                "test_terminal_metrics": gru_ew_eval["test_terminal_metrics"],
                "size": gru_ew_eval["model_size"],
            }
        )
        experiment_records.append({
            "Model Name": f"GRU Early Warning (H={horizon})",
            "Architecture": "GRU Classifier (h=32)",
            "Dataset": "NASA C-MAPSS",
            "Subset": "FD001",
            "Window Length": 30,
            "Task": f"Early Warning (H={horizon})",
            "Threshold": gru_ew_eval["optimal_threshold"],
            "Test Precision": gru_ew_eval["test_metrics"]["test_precision"],
            "Test Recall": gru_ew_eval["test_metrics"]["test_recall"],
            "Test F1": gru_ew_eval["test_metrics"]["test_f1"],
            "Test PR-AUC": gru_ew_eval["test_metrics"]["test_pr_auc"],
            "Test ROC-AUC": gru_ew_eval["test_metrics"]["test_roc_auc"],
            "Params": gru_ew_eval["model_size"]["total_parameters"],
            "Train Time (s)": gru_ew_train["train_duration_sec"],
        })
        print(f"  GRU  (H={horizon}) Test PR-AUC: {gru_ew_eval['test_metrics']['test_pr_auc']:.4f} | F1: {gru_ew_eval['test_metrics']['test_f1']:.4f}", flush=True)

        p_lstm, r_lstm, _ = precision_recall_curve(ew_loaders["test_data"][f"y_{target_name}"], lstm_ew_eval["test_probs"])
        p_gru, r_gru, _ = precision_recall_curve(ew_loaders["test_data"][f"y_{target_name}"], gru_ew_eval["test_probs"])
        ew_pr_curves_data[f"LSTM H={horizon}"] = (r_lstm, p_lstm, lstm_ew_eval["test_metrics"]["test_pr_auc"])
        ew_pr_curves_data[f"GRU H={horizon}"] = (r_gru, p_gru, gru_ew_eval["test_metrics"]["test_pr_auc"])

        if horizon == 30:
            last_ew_cm = np.array(lstm_ew_eval["test_metrics"]["test_confusion_matrix"])
            last_ew_title = f"NASA FD001 Early Warning Confusion Matrix (LSTM H=30, Th={lstm_ew_eval['optimal_threshold']:.2f})"

    # =========================================================================
    # PART 5: UCI AI4I 2020 TABULAR DEEP LEARNING (MLP)
    # =========================================================================
    print("\n" + "#" * 70, flush=True)
    print("PART 5: UCI AI4I 2020 DEEP TABULAR MLP CLASSIFIER", flush=True)
    print("#" * 70, flush=True)

    ai4i_loaders = get_ai4i_dataloaders(val_ratio=0.2, batch_size=128, random_state=42)
    ai4i_mlp = AI4IMLPClassifier(input_dim=ai4i_loaders["input_dim"], hidden_dims=(64, 32, 16), dropout=0.25)
    print(f"AI4I input dim: {ai4i_loaders['input_dim']} features | Pos weight: {ai4i_loaders['pos_weight']:.2f}", flush=True)

    ai4i_train_res = train_binary_classifier(
        ai4i_mlp, ai4i_loaders["train_loader"], ai4i_loaders["val_loader"],
        pos_weight=ai4i_loaders["pos_weight"], epochs=25, lr=2e-3, patience=6, device=device, verbose=True
    )
    ai4i_eval_res = evaluate_classification_model(
        ai4i_mlp, ai4i_loaders["val_loader"], ai4i_loaders["test_loader"],
        ai4i_loaders["y_val"], ai4i_loaders["y_test"], device=device
    )
    save_dl_model(
        ai4i_mlp, "ai4i_clf_deep_mlp", base_models_dir / "ai4i_mlp",
        {
            "config": ai4i_mlp.get_config(),
            "train": ai4i_train_res,
            "optimal_threshold": ai4i_eval_res["optimal_threshold"],
            "val_metrics": ai4i_eval_res["val_metrics"],
            "test_metrics": ai4i_eval_res["test_metrics"],
            "size": ai4i_eval_res["model_size"],
        }
    )
    experiment_records.append({
        "Model Name": "Deep Tabular MLP",
        "Architecture": "MLP (64-32-16 BatchNorm + Dropout)",
        "Dataset": "UCI AI4I 2020",
        "Subset": "Full",
        "Window Length": 1,
        "Task": "Machine Failure Classification",
        "Threshold": ai4i_eval_res["optimal_threshold"],
        "Test Precision": ai4i_eval_res["test_metrics"]["test_precision"],
        "Test Recall": ai4i_eval_res["test_metrics"]["test_recall"],
        "Test F1": ai4i_eval_res["test_metrics"]["test_f1"],
        "Test PR-AUC": ai4i_eval_res["test_metrics"]["test_pr_auc"],
        "Test ROC-AUC": ai4i_eval_res["test_metrics"]["test_roc_auc"],
        "Params": ai4i_eval_res["model_size"]["total_parameters"],
        "Train Time (s)": ai4i_train_res["train_duration_sec"],
    })
    print(f"--> AI4I MLP Test PR-AUC: {ai4i_eval_res['test_metrics']['test_pr_auc']:.4f} | F1: {ai4i_eval_res['test_metrics']['test_f1']:.4f} | Recall: {ai4i_eval_res['test_metrics']['test_recall']:.4f}", flush=True)

    # =========================================================================
    # PART 6: MULTI-CYCLE DEGRADATION TRAJECTORY TRACKING
    # =========================================================================
    print("\n" + "#" * 70, flush=True)
    print("PART 6: MULTI-CYCLE DEGRADATION TRAJECTORY TRACKING ON TEST FLEET", flush=True)
    print("#" * 70, flush=True)

    sample_engine_ids = [31, 49, 81]
    engine_trajectories = []

    test_engine_ids = fd001_w30["test_data"]["engine_ids"]
    test_end_cycles = fd001_w30["test_data"]["end_cycles"]
    test_true_rul = fd001_w30["test_data"]["y_rul"]
    test_lstm_pred = lstm_eval_res["y_test_all_pred"]
    test_gru_pred = gru_eval_res["y_test_all_pred"]

    for uid in sample_engine_ids:
        mask = (test_engine_ids == uid)
        if np.any(mask):
            engine_trajectories.append({
                "unit_id": uid,
                "cycles": test_end_cycles[mask],
                "true_rul": test_true_rul[mask],
                "lstm_rul": test_lstm_pred[mask],
                "gru_rul": test_gru_pred[mask],
            })
            print(f"Extracted trajectory for Engine #{uid}: {np.sum(mask)} sequence points.", flush=True)

    # =========================================================================
    # PART 7: GENERATING PUBLICATION-QUALITY FIGURES (01 TO 09)
    # =========================================================================
    print("\n" + "#" * 70, flush=True)
    print("PART 7: GENERATING PHASE 3 PUBLICATION-QUALITY VISUALIZATIONS", flush=True)
    print("#" * 70, flush=True)

    # Fig 01
    plot_loss_curves(
        {
            "LSTM (FD001)": lstm_train_res["history"],
            "GRU (FD001)": gru_train_res["history"],
            "CNN-LSTM (FD001)": cnn_train_res["history"],
        },
        fig_dirs, filename="01_loss_curves_rul_lstm_gru_cnnlstm.png"
    )
    print("  [✓] Generated Figure 01: 01_loss_curves_rul_lstm_gru_cnnlstm.png", flush=True)

    # Fig 02
    plot_rul_pred_vs_actual(
        {
            "LSTM Regressor": (lstm_eval_res["y_test_terminal_true"], lstm_eval_res["y_test_terminal_pred"]),
            "GRU Regressor": (gru_eval_res["y_test_terminal_true"], gru_eval_res["y_test_terminal_pred"]),
            "CNN-LSTM Regressor": (cnn_eval_res["y_test_terminal_true"], cnn_eval_res["y_test_terminal_pred"]),
        },
        fig_dirs, filename="02_rul_pred_vs_actual_deep_learning.png"
    )
    print("  [✓] Generated Figure 02: 02_rul_pred_vs_actual_deep_learning.png", flush=True)

    # Fig 03
    plot_rul_error_distribution(
        {
            "LSTM": (lstm_eval_res["y_test_terminal_true"], lstm_eval_res["y_test_terminal_pred"]),
            "GRU": (gru_eval_res["y_test_terminal_true"], gru_eval_res["y_test_terminal_pred"]),
            "CNN-LSTM": (cnn_eval_res["y_test_terminal_true"], cnn_eval_res["y_test_terminal_pred"]),
        },
        fig_dirs, filename="03_rul_error_dist_deep_learning.png"
    )
    print("  [✓] Generated Figure 03: 03_rul_error_dist_deep_learning.png", flush=True)

    # Fig 04
    plot_trajectory_tracking(engine_trajectories, fig_dirs, filename="04_rul_trajectory_tracking.png")
    print("  [✓] Generated Figure 04: 04_rul_trajectory_tracking.png", flush=True)

    # Fig 05
    plot_window_length_comparison(
        {
            "L=30 Cycles": {
                "rmse": lstm_eval_res["test_terminal_metrics"]["test_terminal_rmse"],
                "mae": lstm_eval_res["test_terminal_metrics"]["test_terminal_mae"],
                "nasa_score": lstm_eval_res["test_terminal_metrics"]["test_terminal_nasa_score"],
            },
            "L=50 Cycles": {
                "rmse": w50_eval_res["test_terminal_metrics"]["test_terminal_rmse"],
                "mae": w50_eval_res["test_terminal_metrics"]["test_terminal_mae"],
                "nasa_score": w50_eval_res["test_terminal_metrics"]["test_terminal_nasa_score"],
            },
            "L=60 Cycles": {
                "rmse": w60_eval_res["test_terminal_metrics"]["test_terminal_rmse"],
                "mae": w60_eval_res["test_terminal_metrics"]["test_terminal_mae"],
                "nasa_score": w60_eval_res["test_terminal_metrics"]["test_terminal_nasa_score"],
            },
        },
        fig_dirs, filename="05_window_length_comparison.png"
    )
    print("  [✓] Generated Figure 05: 05_window_length_comparison.png", flush=True)

    # Fig 06
    plot_early_warning_pr_curves(ew_pr_curves_data, fig_dirs, filename="06_early_warning_dl_pr_curves.png")
    print("  [✓] Generated Figure 06: 06_early_warning_dl_pr_curves.png", flush=True)

    # Fig 07
    if last_ew_cm is not None:
        plot_confusion_matrix(last_ew_cm, last_ew_title, fig_dirs, filename="07_early_warning_dl_cm.png")
        print("  [✓] Generated Figure 07: 07_early_warning_dl_cm.png", flush=True)

    # Fig 08
    ai4i_cm = np.array(ai4i_eval_res["test_metrics"]["test_confusion_matrix"])
    plot_ai4i_curves_and_cm(
        ai4i_loaders["y_test"], ai4i_eval_res["test_probs"], ai4i_cm,
        ai4i_eval_res["optimal_threshold"], fig_dirs, filename="08_ai4i_mlp_curves_cm.png"
    )
    print("  [✓] Generated Figure 08: 08_ai4i_mlp_curves_cm.png", flush=True)

    # Fig 09
    p2_xgb_rmse = 22.50
    p2_xgb_mae = 15.35
    p2_xgb_score = 9525.5

    p2_rf_rmse = 24.31
    p2_rf_mae = 17.65
    p2_rf_score = 22754.7

    p2_gb_rmse = 22.50
    p2_gb_mae = 16.09
    p2_gb_score = 3202.4

    p2_lr_rmse = 26.54
    p2_lr_mae = 19.57
    p2_lr_score = 11181.9

    plot_ml_vs_dl_comparison(
        {
            "Linear Reg (ML)": {"rmse": p2_lr_rmse, "mae": p2_lr_mae, "nasa_score": p2_lr_score, "paradigm": "ML"},
            "Random Forest (ML)": {"rmse": p2_rf_rmse, "mae": p2_rf_mae, "nasa_score": p2_rf_score, "paradigm": "ML"},
            "Gradient Boost (ML)": {"rmse": p2_gb_rmse, "mae": p2_gb_mae, "nasa_score": p2_gb_score, "paradigm": "ML"},
            "XGBoost (ML)": {"rmse": p2_xgb_rmse, "mae": p2_xgb_mae, "nasa_score": p2_xgb_score, "paradigm": "ML"},
            "LSTM (DL)": {
                "rmse": lstm_eval_res["test_terminal_metrics"]["test_terminal_rmse"],
                "mae": lstm_eval_res["test_terminal_metrics"]["test_terminal_mae"],
                "nasa_score": lstm_eval_res["test_terminal_metrics"]["test_terminal_nasa_score"],
                "paradigm": "DL",
            },
            "GRU (DL)": {
                "rmse": gru_eval_res["test_terminal_metrics"]["test_terminal_rmse"],
                "mae": gru_eval_res["test_terminal_metrics"]["test_terminal_mae"],
                "nasa_score": gru_eval_res["test_terminal_metrics"]["test_terminal_nasa_score"],
                "paradigm": "DL",
            },
            "CNN-LSTM (DL)": {
                "rmse": cnn_eval_res["test_terminal_metrics"]["test_terminal_rmse"],
                "mae": cnn_eval_res["test_terminal_metrics"]["test_terminal_mae"],
                "nasa_score": cnn_eval_res["test_terminal_metrics"]["test_terminal_nasa_score"],
                "paradigm": "DL",
            },
        },
        fig_dirs, filename="09_ml_vs_dl_comparison_bar.png"
    )
    print("  [✓] Generated Figure 09: 09_ml_vs_dl_comparison_bar.png", flush=True)

    # =========================================================================
    # PART 8: PERSIST COMPLETE EXPERIMENT LOG
    # =========================================================================
    exp_df = pd.DataFrame(experiment_records)
    exp_df.to_csv(exp_log_path, index=False)
    print(f"\n[✓] Saved complete Phase 3 experiment log to: {exp_log_path} ({len(exp_df)} runs logged)", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("PHASE 3 EXECUTION COMPLETED SUCCESSFULLY!", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    run_phase3_pipeline()
