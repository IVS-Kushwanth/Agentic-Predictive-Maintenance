"""UCI AI4I 2020 Machine Failure Classification Training Module.

Trains and evaluates baseline classifiers for industrial machine failure:
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier

Protocol:
- Stratified train/validation splitting (80:20 from training set)
- Handling severe class imbalance (3.39% failure rate) via class weights
- Optimal threshold tuning strictly on validation set
- Primary evaluation: PR-AUC, Failure Recall, Precision, F1-Score
- Secondary failure-mode breakdown analysis (TWF, HDF, PWF, OSF, RNF) for evaluation only
"""

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from models.registry import get_classification_models
from evaluation.metrics import compute_classification_metrics, find_optimal_threshold


def load_ai4i_processed_data(
    processed_dir: Path = Path("data/processed/ai4i")
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load preprocessed AI4I train, test, and engineered datasets."""
    X_train_full = pd.read_csv(processed_dir / "X_train.csv")
    y_train_full = pd.read_csv(processed_dir / "y_train.csv")["Machine failure"]
    X_test = pd.read_csv(processed_dir / "X_test.csv")
    y_test = pd.read_csv(processed_dir / "y_test.csv")["Machine failure"]
    df_engineered = pd.read_csv(processed_dir / "ai4i_engineered.csv")

    return X_train_full, y_train_full, X_test, y_test, df_engineered


def train_and_evaluate_ai4i(
    processed_dir: Path = Path("data/processed/ai4i"),
    models_dir: Path = Path("models/baseline"),
    val_size: float = 0.20,
    random_state: int = 42
) -> Dict[str, Any]:
    """Execute complete training, validation, and testing for AI4I dataset."""
    print(f"\n{'='*70}\nRUNNING UCI AI4I 2020 MACHINE FAILURE CLASSIFICATION\n{'='*70}")
    models_dir.mkdir(parents=True, exist_ok=True)

    X_train_full, y_train_full, X_test, y_test, df_eng = load_ai4i_processed_data(processed_dir)
    feature_cols = list(X_train_full.columns)

    # 1. Stratified Validation Split from Training Data
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train_full
    )

    n_pos = int(y_tr.sum())
    n_neg = int(len(y_tr) - n_pos)
    scale_pos_weight = float(n_neg / max(1, n_pos))

    print(f"Dataset splits: Train={len(X_tr)} ({y_tr.mean()*100:.2f}% fail), Val={len(X_val)} ({y_val.mean()*100:.2f}% fail), Test={len(X_test)} ({y_test.mean()*100:.2f}% fail)")
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    models = get_classification_models(
        class_weight="balanced",
        scale_pos_weight=scale_pos_weight,
        random_state=random_state
    )

    results = []
    saved_model_paths = {}
    val_predictions = {}
    test_predictions = {}

    best_model_name = None
    best_val_pr_auc = -1.0

    X_tr_np = X_tr.to_numpy(dtype=np.float32)
    X_val_np = X_val.to_numpy(dtype=np.float32)
    X_test_np = X_test.to_numpy(dtype=np.float32)
    y_tr_np = y_tr.to_numpy(dtype=np.int32)
    y_val_np = y_val.to_numpy(dtype=np.int32)
    y_test_np = y_test.to_numpy(dtype=np.int32)

    for model_name, model in models.items():
        print(f"\n--> Training: {model_name}...")
        t0 = time.time()
        model.fit(X_tr_np, y_tr_np)
        duration = time.time() - t0

        if hasattr(model, "predict_proba"):
            y_val_prob = model.predict_proba(X_val_np)[:, 1]
            y_test_prob = model.predict_proba(X_test_np)[:, 1]
        else:
            dec_val = model.decision_function(X_val_np)
            y_val_prob = 1.0 / (1.0 + np.exp(-dec_val))
            dec_test = model.decision_function(X_test_np)
            y_test_prob = 1.0 / (1.0 + np.exp(-dec_test))

        # Optimize threshold on validation data only
        opt_thresh, opt_val_f1 = find_optimal_threshold(y_val_np, y_val_prob)
        val_metrics = compute_classification_metrics(y_val_np, y_val_prob, threshold=opt_thresh, prefix="val")

        print(f"    Train duration: {duration:.2f}s")
        print(f"    Val: Thresh={opt_thresh:.3f} | PR-AUC={val_metrics['val_pr_auc']:.4f} | F1={val_metrics['val_f1']:.4f} | Recall={val_metrics['val_recall']:.4f} | Prec={val_metrics['val_precision']:.4f}")

        if val_metrics["val_pr_auc"] > best_val_pr_auc:
            best_val_pr_auc = val_metrics["val_pr_auc"]
            best_model_name = model_name

        # Save model
        clean_name = model_name.lower().replace(" ", "_")
        model_file = models_dir / f"ai4i_clf_{clean_name}.joblib"
        joblib.dump(model, model_file)
        saved_model_paths[model_name] = str(model_file)

        # Apply validation-selected threshold to Test
        test_metrics = compute_classification_metrics(y_test_np, y_test_prob, threshold=opt_thresh, prefix="test")
        print(f"    Test: PR-AUC={test_metrics['test_pr_auc']:.4f} | F1={test_metrics['test_f1']:.4f} | Recall={test_metrics['test_recall']:.4f} | Prec={test_metrics['test_precision']:.4f}")

        # Save metadata
        meta_file = models_dir / f"ai4i_clf_{clean_name}_meta.json"
        with open(meta_file, "w") as f:
            json.dump({
                "model_name": model_name,
                "dataset": "UCI AI4I 2020",
                "features": feature_cols,
                "optimal_threshold": opt_thresh,
                "training_duration_seconds": round(duration, 3),
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
            }, f, indent=2)

        log_entry = {
            "model_name": model_name,
            "dataset": "UCI AI4I 2020",
            "subset": "Tabular",
            "task": "Machine Failure Classification",
            "target": "Machine failure",
            "feature_count": len(feature_cols),
            "training_duration_sec": round(duration, 2),
            **val_metrics,
            **test_metrics,
        }
        results.append(log_entry)

        val_predictions[model_name] = {
            "prob": y_val_prob,
            "true": y_val_np,
            "threshold": opt_thresh,
        }
        test_predictions[model_name] = {
            "prob": y_test_prob,
            "true": y_test_np,
            "threshold": opt_thresh,
        }

    print(f"\n{'*'*60}\nBEST AI4I MODEL ON VALIDATION: {best_model_name} (Val PR-AUC: {best_val_pr_auc:.4f})\n{'*'*60}")

    # 2. Secondary Failure-Mode Breakdown Analysis for the Best Model
    best_clean_name = best_model_name.lower().replace(" ", "_")
    best_model = joblib.load(saved_model_paths[best_model_name])
    y_test_pred = (test_predictions[best_model_name]["prob"] >= test_predictions[best_model_name]["threshold"]).astype(int)

    # In Phase 1, X_test was created from stratified split with random_state=42
    # Let's align with failure modes in the test indices
    from sklearn.model_selection import train_test_split as tts
    _, test_idx = tts(
        np.arange(len(df_eng)),
        test_size=0.20,
        random_state=42,
        stratify=df_eng["Machine failure"]
    )
    df_test_modes = df_eng.iloc[test_idx].reset_index(drop=True)

    modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    mode_analysis = {}
    for m in modes:
        mode_mask = df_test_modes[m] == 1
        n_mode_total = int(mode_mask.sum())
        if n_mode_total > 0:
            n_mode_detected = int(y_test_pred[mode_mask].sum())
            recall_mode = float(n_mode_detected / n_mode_total)
        else:
            n_mode_detected = 0
            recall_mode = 0.0
        mode_analysis[m] = {
            "total_occurrences_in_test": n_mode_total,
            "detected_by_best_model": n_mode_detected,
            "mode_recall": round(recall_mode, 4),
        }

    print("\nSecondary Failure-Mode Detection Rates (Test Fleet):")
    for m, info in mode_analysis.items():
        print(f"  Mode {m}: {info['detected_by_best_model']}/{info['total_occurrences_in_test']} detected ({info['mode_recall']*100:.1f}%)")

    return {
        "best_model_name": best_model_name,
        "results": results,
        "saved_model_paths": saved_model_paths,
        "val_predictions": val_predictions,
        "test_predictions": test_predictions,
        "feature_cols": feature_cols,
        "failure_mode_analysis": mode_analysis,
    }


if __name__ == "__main__":
    res_ai4i = train_and_evaluate_ai4i()
    df_ai4i = pd.DataFrame(res_ai4i["results"])
    cols = ["model_name", "val_threshold", "val_pr_auc", "val_f1", "val_recall", "test_pr_auc", "test_f1", "test_recall"]
    print("\nAI4I Summary:")
    print(df_ai4i[cols].to_string(index=False))
