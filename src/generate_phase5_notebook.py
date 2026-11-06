"""
Phase 5 Jupyter Notebook Generator.
Creates notebooks/14_explainable_ai.ipynb.
"""

import os
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


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
        "source": [line + "\n" for line in source.strip().split("\n")]
    }


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    }


def generate_phase5_notebook():
    cells = [
        md_cell("""# Phase 5: Explainable AI for Predictive Maintenance
## Interpreting Machine Degradation, Sensor Attribution, and Failure Mechanisms

**Objective:**
This notebook implements and demonstrates Explainable AI (XAI) across the models developed in Phases 2, 3, and 4:
1. **TreeSHAP for NASA C-MAPSS RUL Regression:** Identifying the physical and rolling sensor features driving remaining useful life estimates.
2. **TreeSHAP for UCI AI4I 2020 Machine Failure:** Explaining why a machine received a high failure-risk prediction and identifying influential operating parameters.
3. **Permutation Feature Importance vs. SHAP:** Comparing empirical performance degradation with cooperative game-theoretic Shapley values.
4. **Deep Learning Explainability (Integrated Gradients):** Spatiotemporal attribution across historical time steps and sensor channels for LSTM and GRU models.
5. **Standardized Decision-Support API:** Demonstrating `explain_prediction()`, `get_global_feature_importance()`, `get_temporal_explanation()`, and `get_sensor_contributions()` for future LangGraph agents."""),

        code_cell("""import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.explainability import (
    explain_prediction,
    get_global_feature_importance,
    get_temporal_explanation,
    get_sensor_contributions,
    TreeSHAPExplainer,
    PermutationExplainer,
    DeepTemporalExplainer
)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
print("Phase 5 Explainability package successfully loaded!")"""),

        md_cell("""## 1. NASA C-MAPSS: SHAP Analysis for RUL Regression
We load the pre-trained XGBoost Regressor and analyze global feature drivers along with local instance explanations."""),

        code_cell("""# Load NASA metadata and global SHAP importance
with open(os.path.join(ROOT_DIR, "models", "explainability", "nasa_shap_importance.json"), "r") as f:
    nasa_shap_list = json.load(f)

df_nasa_shap = pd.DataFrame(nasa_shap_list[:15])
print("Top 15 Global Feature Drivers for NASA RUL Regression:")
display(df_nasa_shap)

# Bar chart of top features
plt.figure(figsize=(10, 5))
plt.barh(df_nasa_shap["feature"][::-1], df_nasa_shap["mean_abs_shap"][::-1], color="#2980b9")
plt.title("NASA C-MAPSS FD001: Global SHAP Feature Importance (Mean |SHAP| Cycles)", fontweight="bold")
plt.xlabel("Mean |SHAP| Impact (Cycles)")
plt.show()"""),

        md_cell("""## 2. Individual Prediction Explanation: Waterfall Decomposition
We explain an individual engine cycle prediction using SHAP waterfall decomposition, showing how each sensor pushes the prediction away from the baseline fleet mean."""),

        code_cell("""val_tabular = pd.read_csv(os.path.join(ROOT_DIR, "data", "processed", "nasa", "FD001", "val_tabular.csv"))
features = [c for c in val_tabular.columns if c not in ['engine_id', 'cycle', 'RUL', 'RUL_clipped', 'ew_10', 'ew_20', 'ew_30']]

# Select a near-failure cycle
near_fail_sample = val_tabular[val_tabular["RUL"] < 25].iloc[0]
sample_x = near_fail_sample[features].values

explanation = explain_prediction(sample_x, model_id="xgboost", dataset="FD001", top_k=6)
print("=== Individual Prediction Explanation ===")
print(f"Predicted RUL: {explanation['prediction']['predicted_rul']} cycles")
print(f"Explanation Summary: {explanation['explanation_summary']}")
print("\nTop Contributing Features:")
for item in explanation["important_features"]:
    print(f"  - {item['feature']:30s}: {item['importance']:+7.2f} cycles (Direction: {item['contribution']})")"""),

        md_cell("""## 3. UCI AI4I 2020: Explaining Machine Failure Predictions
We analyze the most influential operating parameters for industrial machine breakdown risk."""),

        code_cell("""with open(os.path.join(ROOT_DIR, "models", "explainability", "ai4i_shap_importance.json"), "r") as f:
    ai4i_shap_list = json.load(f)

df_ai4i_shap = pd.DataFrame(ai4i_shap_list)
print("UCI AI4I 2020: Influential Operating Parameters for Machine Failure:")
display(df_ai4i_shap)

plt.figure(figsize=(9, 4.5))
plt.barh(df_ai4i_shap["feature"][::-1], df_ai4i_shap["mean_abs_shap"][::-1], color="#e74c3c")
plt.title("UCI AI4I: Global SHAP Feature Importance for Machine Failure", fontweight="bold")
plt.xlabel("Mean |SHAP| Impact")
plt.show()"""),

        code_cell("""# Explain an actual failure instance from AI4I test set
X_test_ai4i = pd.read_csv(os.path.join(ROOT_DIR, "data", "processed", "ai4i", "X_test.csv"))
y_test_ai4i = pd.read_csv(os.path.join(ROOT_DIR, "data", "processed", "ai4i", "y_test.csv")).values.ravel()

fail_idx = np.where(y_test_ai4i == 1)[0][0]
sample_fail = X_test_ai4i.iloc[fail_idx].values

exp_ai4i_fail = explain_prediction(sample_fail, model_id="xgboost", dataset="AI4I", top_k=5)
print("=== AI4I Breakdown Instance Explanation ===")
print(f"Predicted Failure Risk: {exp_ai4i_fail['prediction']['predicted_probability']*100:.1f}%")
print(f"Status: {exp_ai4i_fail['prediction']['predicted_label']}")
print(f"Summary: {exp_ai4i_fail['explanation_summary']}")"""),

        md_cell("""## 4. Permutation Feature Importance vs. SHAP Comparison
We compare permutation importance (empirical degradation upon shuffling) against SHAP game-theoretic attribution."""),

        code_cell("""with open(os.path.join(ROOT_DIR, "models", "explainability", "permutation_importance.json"), "r") as f:
    perm_data = json.load(f)

concordance = perm_data["concordance_metrics"]
print(f"NASA SHAP vs Permutation Spearman Rank Correlation: {concordance['nasa']['spearman_rank_correlation']:.4f}")
print(f"AI4I SHAP vs Permutation Spearman Rank Correlation: {concordance['ai4i']['spearman_rank_correlation']:.4f}")
print(f"Concordant Top Features in NASA (out of 15): {concordance['nasa']['concordant_top_features']}")"""),

        md_cell("""## 5. Deep Learning Explainability: Integrated Gradients (LSTM & GRU)
We investigate spatiotemporal attribution across the 30-cycle observation window, identifying how temporal decay influences degradation estimates."""),

        code_cell("""val_npz = np.load(os.path.join(ROOT_DIR, "data", "processed", "nasa", "FD001", "sequences_w30_val.npz"))
test_seq = val_npz["X"][0]  # (30, 153)

deep_exp = explain_prediction(test_seq, model_id="lstm", dataset="FD001", top_k=5)
print("=== Deep Learning LSTM Prediction Explanation ===")
print(f"Predicted RUL: {deep_exp['prediction']['predicted_rul']} cycles")
print(f"Summary: {deep_exp['explanation_summary']}")

temporal = deep_exp["temporal_attribution"]
print(f"Peak Influence Cycle: Cycle {temporal['peak_influence_cycle_in_window']} / 30")
print(f"Recency Weight Ratio (Last 10 vs First 10 cycles): {temporal['recent_vs_early_weight_ratio']}x")

# Plot historical time step attribution weights
plt.figure(figsize=(9, 4))
plt.plot(range(1, 31), np.array(temporal["time_step_weights"]) * 100, marker="o", color="#8e44ad", linewidth=2.0)
plt.fill_between(range(1, 31), 0, np.array(temporal["time_step_weights"]) * 100, color="#9b59b6", alpha=0.25)
plt.title("LSTM Temporal Attribution: Historical Time Step Influence Across 30-Cycle Window", fontweight="bold")
plt.xlabel("Cycle within Window (1 = Oldest, 30 = Most Recent)")
plt.ylabel("Attribution Share (%)")
plt.show()"""),

        md_cell("""## 6. Standardized Public API Functions for Downstream Agent Decision Support
Testing reusable helper functions designed for future LangGraph agents."""),

        code_cell("""# 1. get_global_feature_importance()
global_top = get_global_feature_importance(model_id="xgboost", dataset="FD001", top_k=5)
print("Global Top Features:", [item["feature"] for item in global_top])

# 2. get_temporal_explanation()
temporal_info = get_temporal_explanation(test_seq, model_id="lstm", dataset="FD001")
print(f"Temporal Peak Cycle: {temporal_info['peak_influence_cycle_in_window']}")

# 3. get_sensor_contributions()
sensor_contribs = get_sensor_contributions(sample_x, model_id="xgboost", dataset="FD001", top_k=3)
print("Top 3 Sensor Contributions:")
for item in sensor_contribs:
    print(f"  {item['feature']}: {item['importance']:+.2f} cycles")""")
    ]

    nb = make_notebook(cells)
    out_path = os.path.join(ROOT_DIR, "notebooks", "14_explainable_ai.ipynb")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Successfully generated notebook: {out_path}")


if __name__ == "__main__":
    generate_phase5_notebook()
