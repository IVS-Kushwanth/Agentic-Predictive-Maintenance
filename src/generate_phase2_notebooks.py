"""Phase 2 Notebook Generation Script.

Constructs:
- notebooks/06_nasa_rul_baselines.ipynb
- notebooks/07_nasa_early_warning.ipynb
- notebooks/08_ai4i_failure_classification.ipynb
"""

import json
from pathlib import Path


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.15"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    }


def build_nb_06():
    cells = [
        md_cell("""# 06 - NASA C-MAPSS RUL Regression Baselines
### Phase 2: Machine Learning Baseline Development & Evaluation

This notebook evaluates baseline regression models for Remaining Useful Life (RUL) estimation on the **NASA C-MAPSS FD001** benchmark dataset:
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

Evaluations strictly adhere to:
1. Engine-level training/validation split (no window leakage).
2. Official NASA test terminal cycle benchmark (one prediction per engine at final observed cycle).
3. NASA Asymmetric Scoring Function & RMSE/MAE evaluation."""),
        code_cell("""import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src to sys.path
project_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.append(str(project_root / "src"))

from evaluation.metrics import compute_regression_metrics, nasa_asymmetric_scoring_function
from training.nasa_rul_trainer import load_nasa_tabular_data"""),
        md_cell("## 1. Load Processed Tabular Data & Features"),
        code_cell("""proc_dir = project_root / "data" / "processed" / "nasa"
train_df, val_df, test_df, feature_cols = load_nasa_tabular_data("FD001", processed_dir=proc_dir)

print(f"Train Shape: {train_df.shape} ({train_df['unit_id'].nunique()} engines)")
print(f"Val Shape:   {val_df.shape} ({val_df['unit_id'].nunique()} engines)")
print(f"Test Shape:  {test_df.shape} ({test_df['unit_id'].nunique()} engines)")
print(f"Total Features: {len(feature_cols)}")"""),
        md_cell("## 2. Load Saved Baseline Models & Evaluate on Validation Fleet"),
        code_cell("""models_dir = project_root / "models" / "baseline"
model_names = [
    "Linear Regression",
    "Random Forest Regressor",
    "Gradient Boosting Regressor",
    "XGBoost Regressor"
]

X_val = val_df[feature_cols].to_numpy(dtype=np.float32)
y_val = val_df["RUL"].to_numpy(dtype=np.float32)

val_summary = []
loaded_models = {}

for name in model_names:
    clean_name = name.lower().replace(" ", "_")
    model_path = models_dir / f"nasa_rul_{clean_name}_FD001.joblib"
    if model_path.exists():
        model = joblib.load(model_path)
        loaded_models[name] = model
        y_val_pred = model.predict(X_val)
        m = compute_regression_metrics(y_val, y_val_pred, prefix="val")
        val_summary.append({"Model": name, **m})

df_val_results = pd.DataFrame(val_summary)
display(df_val_results)"""),
        md_cell("## 3. Official NASA Test Terminal Benchmark Evaluation"),
        code_cell("""# Extract terminal cycle per test engine
test_terminal_df = test_df.groupby("unit_id").last().reset_index()
X_test_term = test_terminal_df[feature_cols].to_numpy(dtype=np.float32)
y_test_term = test_terminal_df["RUL"].to_numpy(dtype=np.float32)

term_summary = []
for name, model in loaded_models.items():
    y_test_pred = model.predict(X_test_term)
    m = compute_regression_metrics(y_test_term, y_test_pred, prefix="test_terminal")
    term_summary.append({"Model": name, **m})

df_term_results = pd.DataFrame(term_summary)
display(df_term_results)"""),
        md_cell("## 4. Predicted vs Actual RUL (Best Model)"),
        code_cell("""best_model_name = df_val_results.sort_values(by="val_rmse").iloc[0]["Model"]
print(f"Best Model Selected by Validation RMSE: {best_model_name}")

best_model = loaded_models[best_model_name]
y_pred_best = best_model.predict(X_test_term)

plt.figure(figsize=(7, 6))
plt.scatter(y_test_term, y_pred_best, alpha=0.75, color="#1f77b4", edgecolors="k", s=45)
min_v = min(y_test_term.min(), y_pred_best.min())
max_v = max(y_test_term.max(), y_pred_best.max())
plt.plot([min_v, max_v], [min_v, max_v], "r--", lw=2, label="Perfect Agreement (y = x)")
plt.title(f"Predicted vs Actual RUL on Terminal Test Fleet ({best_model_name})")
plt.xlabel("Actual Terminal RUL (Cycles)")
plt.ylabel("Predicted Terminal RUL (Cycles)")
plt.legend()
plt.tight_layout()
plt.show()"""),
    ]
    return make_notebook(cells)


def build_nb_07():
    cells = [
        md_cell("""# 07 - NASA C-MAPSS Early-Warning Multi-Horizon Classification
### Phase 2: Machine Learning Baseline Development & Evaluation

This notebook evaluates binary classification models across three critical early-warning horizons:
- **Horizon 10** ($RUL \le 10$ cycles): Emergency maintenance alert.
- **Horizon 20** ($RUL \le 20$ cycles): Tactical dispatch alert.
- **Horizon 30** ($RUL \le 30$ cycles): Advance supply-chain / part scheduling.

Decision thresholds are strictly tuned on validation data to optimize F1 / Recall under class imbalance."""),
        code_cell("""import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

project_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.append(str(project_root / "src"))

from evaluation.metrics import compute_classification_metrics, find_optimal_threshold
from training.nasa_rul_trainer import load_nasa_tabular_data"""),
        md_cell("## 1. Load Tabular Data & Class Distributions"),
        code_cell("""proc_dir = project_root / "data" / "processed" / "nasa"
train_df, val_df, test_df, feature_cols = load_nasa_tabular_data("FD001", processed_dir=proc_dir)

for h in [10, 20, 30]:
    rate = train_df[f"early_warning_{h}"].mean() * 100
    print(f"Horizon H={h} Positive Class Prevalence (Train): {rate:.2f}%")"""),
        md_cell("## 2. Load Models & Evaluate Test Performance Across Horizons"),
        code_cell("""models_dir = project_root / "models" / "baseline"
models = ["Logistic Regression", "Random Forest Classifier", "Gradient Boosting Classifier", "XGBoost Classifier"]

results = []
X_val = val_df[feature_cols].to_numpy(dtype=np.float32)
X_test = test_df[feature_cols].to_numpy(dtype=np.float32)

for h in [10, 20, 30]:
    y_val = val_df[f"early_warning_{h}"].to_numpy(dtype=np.int32)
    y_test = test_df[f"early_warning_{h}"].to_numpy(dtype=np.int32)
    
    for m in models:
        clean_m = m.lower().replace(" ", "_")
        path = models_dir / f"nasa_ew_{h}_{clean_m}_FD001.joblib"
        if path.exists():
            clf = joblib.load(path)
            y_val_prob = clf.predict_proba(X_val)[:, 1] if hasattr(clf, "predict_proba") else 1.0 / (1.0 + np.exp(-clf.decision_function(X_val)))
            y_test_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else 1.0 / (1.0 + np.exp(-clf.decision_function(X_test)))
            
            opt_th, _ = find_optimal_threshold(y_val, y_val_prob)
            val_m = compute_classification_metrics(y_val, y_val_prob, threshold=opt_th, prefix="val")
            test_m = compute_classification_metrics(y_test, y_test_prob, threshold=opt_th, prefix="test")
            
            results.append({
                "Horizon": f"H={h}",
                "Model": m,
                "Optimal Threshold": opt_th,
                "Val F1": val_m["val_f1"],
                "Val PR-AUC": val_m["val_pr_auc"],
                "Test F1": test_m["test_f1"],
                "Test PR-AUC": test_m["test_pr_auc"],
                "Test Recall": test_m["test_recall"],
                "Test Precision": test_m["test_precision"],
            })

df_ew_results = pd.DataFrame(results)
display(df_ew_results)"""),
    ]
    return make_notebook(cells)


def build_nb_08():
    cells = [
        md_cell("""# 08 - UCI AI4I 2020 Machine Failure Classification
### Phase 2: Machine Learning Baseline Development & Evaluation

This notebook evaluates machine failure classification on the tabular **UCI AI4I 2020** dataset:
- Severe class imbalance: ~3.39% failure rate.
- Input features strictly exclude all target leakage columns (`UDI`, `Product ID`, `Machine failure`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF`).
- Domain-informed physical features are incorporated.
- Decision threshold is tuned strictly on the validation subset.
- Secondary failure-mode breakdown (TWF, HDF, PWF, OSF, RNF) is performed for post-hoc diagnostic explanation."""),
        code_cell("""import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

project_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.append(str(project_root / "src"))

from evaluation.metrics import compute_classification_metrics, find_optimal_threshold
from training.ai4i_trainer import load_ai4i_processed_data"""),
        md_cell("## 1. Load Processed AI4I Data"),
        code_cell("""proc_dir = project_root / "data" / "processed" / "ai4i"
X_train_full, y_train_full, X_test, y_test, df_eng = load_ai4i_processed_data(proc_dir)

print(f"X_train Shape: {X_train_full.shape}, Failure Rate: {y_train_full.mean()*100:.2f}%")
print(f"X_test Shape:  {X_test.shape}, Failure Rate:  {y_test.mean()*100:.2f}%")
print(f"Features: {list(X_train_full.columns)}")"""),
        md_cell("## 2. Load Models & Evaluate on Test Set"),
        code_cell("""models_dir = project_root / "models" / "baseline"
models = ["Logistic Regression", "Random Forest Classifier", "Gradient Boosting Classifier", "XGBoost Classifier"]

summary = []
loaded_clfs = {}

for m in models:
    clean_m = m.lower().replace(" ", "_")
    p = models_dir / f"ai4i_clf_{clean_m}.joblib"
    meta_p = models_dir / f"ai4i_clf_{clean_m}_meta.json"
    if p.exists() and meta_p.exists():
        with open(meta_p, "r") as f:
            meta = json.load(f)
        clf = joblib.load(p)
        loaded_clfs[m] = (clf, meta["optimal_threshold"])
        summary.append({
            "Model": m,
            "Tuned Threshold": meta["optimal_threshold"],
            "Val PR-AUC": meta["val_metrics"]["val_pr_auc"],
            "Val F1": meta["val_metrics"]["val_f1"],
            "Test PR-AUC": meta["test_metrics"]["test_pr_auc"],
            "Test F1": meta["test_metrics"]["test_f1"],
            "Test Recall": meta["test_metrics"]["test_recall"],
            "Test Precision": meta["test_metrics"]["test_precision"],
            "Test ROC-AUC": meta["test_metrics"]["test_roc_auc"],
        })

df_ai4i_results = pd.DataFrame(summary)
display(df_ai4i_results)"""),
        md_cell("## 3. Secondary Failure-Mode Analysis"),
        code_cell("""# Check detection rates across specific failure modes
best_m = df_ai4i_results.sort_values(by="Val PR-AUC", ascending=False).iloc[0]["Model"]
clf, th = loaded_clfs[best_m]

X_test_np = X_test.to_numpy(dtype=np.float32)
y_prob = clf.predict_proba(X_test_np)[:, 1] if hasattr(clf, "predict_proba") else 1.0 / (1.0 + np.exp(-clf.decision_function(X_test_np)))
y_pred = (y_prob >= th).astype(int)

from sklearn.model_selection import train_test_split as tts
_, test_idx = tts(np.arange(len(df_eng)), test_size=0.20, random_state=42, stratify=df_eng["Machine failure"])
df_test_modes = df_eng.iloc[test_idx].reset_index(drop=True)

modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
mode_breakdown = []
for m in modes:
    mask = df_test_modes[m] == 1
    total = int(mask.sum())
    detected = int(y_pred[mask].sum()) if total > 0 else 0
    rec = detected / total if total > 0 else 0.0
    mode_breakdown.append({"Failure Mode": m, "Total in Test": total, "Detected": detected, "Recall (%)": round(rec*100, 1)})

df_mode = pd.DataFrame(mode_breakdown)
display(df_mode)"""),
    ]
    return make_notebook(cells)


def main():
    nb_dir = Path("notebooks")
    nb_dir.mkdir(parents=True, exist_ok=True)

    notebooks = {
        "06_nasa_rul_baselines.ipynb": build_nb_06(),
        "07_nasa_early_warning.ipynb": build_nb_07(),
        "08_ai4i_failure_classification.ipynb": build_nb_08(),
    }

    for name, nb in notebooks.items():
        file_path = nb_dir / name
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=2)
        print(f"Created Phase 2 notebook: {file_path}")


if __name__ == "__main__":
    main()
