"""NASA C-MAPSS Early-Warning Multi-Horizon Classification Training Module.

Trains and evaluates baseline classifiers for early-warning horizons:
- Horizon 10 (RUL <= 10 cycles, Critical Emergency)
- Horizon 20 (RUL <= 20 cycles, Maintenance Dispatch)
- Horizon 30 (RUL <= 30 cycles, Advance Scheduling)

Models:
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier

Protocol:
- Engine-level training and validation isolation
- Threshold selection strictly on validation data to optimize F1/Recall
- Evaluation metrics: Precision, Recall, F1, PR-AUC, ROC-AUC, Confusion Matrix
"""

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd

from models.registry import get_classification_models
from evaluation.metrics import compute_classification_metrics, find_optimal_threshold
from training.nasa_rul_trainer import load_nasa_tabular_data


def train_and_evaluate_nasa_early_warning(
    subset: str = "FD001",
    horizons: List[int] = [10, 20, 30],
    processed_dir: Path = Path("data/processed/nasa"),
    models_dir: Path = Path("models/baseline"),
    random_state: int = 42
) -> Dict[str, Any]:
    """Execute complete early-warning classification experiments for a NASA subset."""
    print(f"\n{'='*70}\nRUNNING NASA C-MAPSS EARLY-WARNING CLASSIFICATION: {subset}\n{'='*70}")
    models_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df, feature_cols = load_nasa_tabular_data(subset, processed_dir)

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    X_val = val_df[feature_cols].to_numpy(dtype=np.float32)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)

    all_horizon_results = []
    saved_model_paths = {}
    val_predictions = {}
    test_predictions = {}

    for h in horizons:
        target_col = f"early_warning_{h}"
        y_train = train_df[target_col].to_numpy(dtype=np.int32)
        y_val = val_df[target_col].to_numpy(dtype=np.int32)
        y_test = test_df[target_col].to_numpy(dtype=np.int32)

        n_pos = int(np.sum(y_train))
        n_neg = int(len(y_train) - n_pos)
        scale_pos_weight = float(n_neg / max(1, n_pos))
        pos_rate = float(n_pos / len(y_train))

        print(f"\n--- Horizon H={h} cycles (Positive rate in train: {pos_rate*100:.2f}%, scale_pos_weight: {scale_pos_weight:.2f}) ---")

        models = get_classification_models(
            class_weight="balanced",
            scale_pos_weight=scale_pos_weight,
            random_state=random_state
        )

        for model_name, model in models.items():
            print(f"  --> Training: {model_name} (H={h})...")
            t0 = time.time()
            model.fit(X_train, y_train)
            duration = time.time() - t0

            # Predict probabilities on validation
            if hasattr(model, "predict_proba"):
                y_val_prob = model.predict_proba(X_val)[:, 1]
                y_test_prob = model.predict_proba(X_test)[:, 1]
            else:
                decision = model.decision_function(X_val)
                y_val_prob = 1.0 / (1.0 + np.exp(-decision))
                decision_test = model.decision_function(X_test)
                y_test_prob = 1.0 / (1.0 + np.exp(-decision_test))

            # Select optimal decision threshold using validation data only
            opt_thresh, opt_val_f1 = find_optimal_threshold(y_val, y_val_prob)
            val_metrics = compute_classification_metrics(y_val, y_val_prob, threshold=opt_thresh, prefix="val")

            # Apply validation-selected threshold to test data
            test_metrics = compute_classification_metrics(y_test, y_test_prob, threshold=opt_thresh, prefix="test")

            print(f"      Val: Thresh={opt_thresh:.3f} | F1={val_metrics['val_f1']:.4f} | PR-AUC={val_metrics['val_pr_auc']:.4f} | Recall={val_metrics['val_recall']:.4f}")
            print(f"      Test: F1={test_metrics['test_f1']:.4f} | PR-AUC={test_metrics['test_pr_auc']:.4f} | Recall={test_metrics['test_recall']:.4f}")

            # Save model
            clean_name = model_name.lower().replace(" ", "_")
            model_key = f"nasa_ew_{h}_{clean_name}_{subset}"
            model_file = models_dir / f"{model_key}.joblib"
            joblib.dump(model, model_file)
            saved_model_paths[model_key] = str(model_file)

            meta_file = models_dir / f"{model_key}_meta.json"
            with open(meta_file, "w") as f:
                json.dump({
                    "model_name": model_name,
                    "subset": subset,
                    "horizon": h,
                    "target": target_col,
                    "optimal_threshold": opt_thresh,
                    "training_duration_seconds": round(duration, 3),
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                }, f, indent=2)

            log_entry = {
                "model_name": model_name,
                "dataset": "NASA C-MAPSS",
                "subset": subset,
                "task": f"Early Warning (H={h})",
                "target": target_col,
                "feature_count": len(feature_cols),
                "training_duration_sec": round(duration, 2),
                **val_metrics,
                **test_metrics,
            }
            all_horizon_results.append(log_entry)

            val_predictions[model_key] = {
                "prob": y_val_prob,
                "true": y_val,
                "threshold": opt_thresh,
            }
            test_predictions[model_key] = {
                "prob": y_test_prob,
                "true": y_test,
                "threshold": opt_thresh,
            }

    return {
        "subset": subset,
        "results": all_horizon_results,
        "saved_model_paths": saved_model_paths,
        "val_predictions": val_predictions,
        "test_predictions": test_predictions,
        "feature_cols": feature_cols,
    }


if __name__ == "__main__":
    res_ew = train_and_evaluate_nasa_early_warning(subset="FD001")
    df_ew = pd.DataFrame(res_ew["results"])
    cols = ["model_name", "target", "val_threshold", "val_f1", "val_pr_auc", "val_recall", "test_f1", "test_pr_auc", "test_recall"]
    print("\nEarly Warning Summary:")
    print(df_ew[cols].to_string(index=False))
