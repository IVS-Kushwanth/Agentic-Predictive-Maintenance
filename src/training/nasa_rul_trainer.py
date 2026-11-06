"""NASA C-MAPSS RUL Regression Training and Evaluation Module.

Trains and evaluates baseline regressors:
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

Evaluation Protocol:
- Training strictly on training engines (80 engines in FD001)
- Model comparison and selection strictly on validation engines (20 engines in FD001)
- Primary benchmark evaluation: Terminal cycle per test engine against official RUL vector
- Supplementary evaluation: Full test trajectory cycles
"""

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd

from models.registry import get_regression_models
from evaluation.metrics import compute_regression_metrics, nasa_asymmetric_scoring_function


def load_nasa_tabular_data(
    subset: str = "FD001",
    processed_dir: Path = Path("data/processed/nasa")
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """Load train, val, and test tabular datasets along with feature names."""
    subset_dir = processed_dir / subset
    train_df = pd.read_csv(subset_dir / "train_tabular.csv")
    val_df = pd.read_csv(subset_dir / "val_tabular.csv")
    test_df = pd.read_csv(subset_dir / "test_tabular.csv")

    with open(subset_dir / "metadata.json", "r") as f:
        meta = json.load(f)
    
    feature_cols = meta["all_features"]
    return train_df, val_df, test_df, feature_cols


def train_and_evaluate_nasa_rul(
    subset: str = "FD001",
    processed_dir: Path = Path("data/processed/nasa"),
    models_dir: Path = Path("models/baseline"),
    random_state: int = 42
) -> Dict[str, Any]:
    """Execute complete RUL regression experiments for a NASA subset."""
    print(f"\n{'='*70}\nRUNNING NASA C-MAPSS RUL REGRESSION: {subset}\n{'='*70}")
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    train_df, val_df, test_df, feature_cols = load_nasa_tabular_data(subset, processed_dir)

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df["RUL"].to_numpy(dtype=np.float32)

    X_val = val_df[feature_cols].to_numpy(dtype=np.float32)
    y_val = val_df["RUL"].to_numpy(dtype=np.float32)

    # Prepare primary test benchmark: Terminal cycle per test engine
    test_terminal_df = test_df.groupby("unit_id").last().reset_index()
    X_test_terminal = test_terminal_df[feature_cols].to_numpy(dtype=np.float32)
    y_test_terminal = test_terminal_df["RUL"].to_numpy(dtype=np.float32)

    # Prepare supplementary test: All cycles
    X_test_all = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_test_all = test_df["RUL"].to_numpy(dtype=np.float32)

    print(f"Data shapes: Train X={X_train.shape}, Val X={X_val.shape}, Test Terminal X={X_test_terminal.shape}")

    models = get_regression_models(random_state=random_state)
    experiment_results = []
    saved_model_paths = {}
    val_predictions = {}
    test_predictions = {}

    best_model_name = None
    best_val_rmse = float("inf")

    for model_name, model in models.items():
        print(f"\n--> Training: {model_name}...")
        t0 = time.time()
        model.fit(X_train, y_train)
        duration = time.time() - t0

        # Predict on validation engines
        y_val_pred = model.predict(X_val)
        val_predictions[model_name] = y_val_pred
        val_metrics = compute_regression_metrics(y_val, y_val_pred, prefix="val")

        print(f"    Train duration: {duration:.2f}s")
        print(f"    Val RMSE: {val_metrics['val_rmse']:.2f} | Val MAE: {val_metrics['val_mae']:.2f} | Val Score: {val_metrics['val_nasa_score']:.1f}")

        # Check best model on validation
        if val_metrics["val_rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["val_rmse"]
            best_model_name = model_name

        # Save model
        clean_name = model_name.lower().replace(" ", "_")
        model_file = models_dir / f"nasa_rul_{clean_name}_{subset}.joblib"
        joblib.dump(model, model_file)
        saved_model_paths[model_name] = str(model_file)

        # Save model metadata
        meta_file = models_dir / f"nasa_rul_{clean_name}_{subset}_meta.json"
        with open(meta_file, "w") as f:
            json.dump({
                "model_name": model_name,
                "subset": subset,
                "train_samples": len(X_train),
                "feature_count": len(feature_cols),
                "features": feature_cols,
                "training_duration_seconds": round(duration, 3),
                "val_metrics": val_metrics,
            }, f, indent=2)

        # Log entry
        log_entry = {
            "model_name": model_name,
            "dataset": "NASA C-MAPSS",
            "subset": subset,
            "task": "RUL Regression",
            "target": "RUL",
            "feature_count": len(feature_cols),
            "training_duration_sec": round(duration, 2),
            **val_metrics,
        }
        experiment_results.append(log_entry)

    print(f"\n{'*'*60}\nBEST RUL MODEL ON VALIDATION: {best_model_name} (Val RMSE: {best_val_rmse:.2f})\n{'*'*60}")

    # Now evaluate on Test Fleet
    for entry in experiment_results:
        m_name = entry["model_name"]
        model = joblib.load(saved_model_paths[m_name])

        # Terminal benchmark
        y_pred_terminal = model.predict(X_test_terminal)
        terminal_metrics = compute_regression_metrics(y_test_terminal, y_pred_terminal, prefix="test_terminal")
        test_predictions[m_name] = {
            "terminal_pred": y_pred_terminal,
            "terminal_true": y_test_terminal,
        }

        # Supplementary all-cycle
        y_pred_all = model.predict(X_test_all)
        all_metrics = compute_regression_metrics(y_test_all, y_pred_all, prefix="test_all_cycles")

        entry.update(terminal_metrics)
        entry.update(all_metrics)

        print(f"\nTest Evaluation for {m_name}:")
        print(f"  [Primary Benchmark - Terminal Cycles] RMSE: {terminal_metrics['test_terminal_rmse']:.2f} | MAE: {terminal_metrics['test_terminal_mae']:.2f} | NASA Score: {terminal_metrics['test_terminal_nasa_score']:.1f}")
        print(f"  [Supplementary - All Cycles]         RMSE: {all_metrics['test_all_cycles_rmse']:.2f} | MAE: {all_metrics['test_all_cycles_mae']:.2f}")

    return {
        "subset": subset,
        "best_model_name": best_model_name,
        "results": experiment_results,
        "saved_model_paths": saved_model_paths,
        "val_predictions": val_predictions,
        "test_predictions": test_predictions,
        "y_val_true": y_val,
        "y_test_terminal_true": y_test_terminal,
        "feature_cols": feature_cols,
    }


if __name__ == "__main__":
    res = train_and_evaluate_nasa_rul(subset="FD001")
    df_res = pd.DataFrame(res["results"])
    print("\nSummary Table:")
    cols = ["model_name", "val_rmse", "val_mae", "val_nasa_score", "test_terminal_rmse", "test_terminal_mae", "test_terminal_nasa_score"]
    print(df_res[cols].to_string(index=False))
