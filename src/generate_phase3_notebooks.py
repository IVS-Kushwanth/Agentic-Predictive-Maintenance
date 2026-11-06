"""Phase 3 Jupyter Notebook Generator Script.

Constructs:
- notebooks/09_lstm_rul_prediction.ipynb
- notebooks/10_gru_rul_prediction.ipynb
- notebooks/11_early_warning_classification.ipynb
- notebooks/12_deep_learning_comparison.ipynb
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


def build_nb_09():
    cells = [
        md_cell("""# 09 - LSTM Neural Networks for RUL Estimation
### Phase 3: Deep Learning Model Development, Training & Evaluation

This notebook investigates **Long Short-Term Memory (LSTM)** recurrent architectures for Remaining Useful Life (RUL) estimation on multivariate temporal sensor sequences from **NASA C-MAPSS FD001**.

#### Key Objectives:
1. Load 3D multivariate temporal sliding windows $(N, L, D)$ with zero leakage across engine boundaries.
2. Formulate stacked LSTM regression networks with Adam optimization and Huber loss.
3. Evaluate RUL predictions against the official NASA test fleet terminal cycle benchmark ($N=100$).
4. Analyze sensitivity to temporal window lengths ($L \\in \\{30, 50, 60\\}$).
5. Track continuous degradation trajectories across operational cycles."""),

        code_cell("""import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path("..").resolve() / "src"))

from deep_learning.dataset_loader import get_cmapss_dataloaders
from deep_learning.lstm_model import LSTMRULModel
from deep_learning.evaluate import evaluate_cmapss_rul

print("PyTorch version:", torch.__version__)
print("Device available:", "CUDA" if torch.cuda.is_available() else "CPU")"""),

        md_cell("""## 1. Load FD001 Sequence Data and Model Weights"""),

        code_cell("""# Load FD001 (w=30) sequences
loaders_w30 = get_cmapss_dataloaders("FD001", window_length=30, batch_size=256)
in_dim = loaders_w30["num_features"]

print(f"Sequence Length (L): {loaders_w30['sequence_length']}")
print(f"Number of Sensor/Operational Features: {in_dim}")
print(f"Train Sequences: {len(loaders_w30['train_data']['X'])}")
print(f"Validation Sequences: {len(loaders_w30['val_data']['X'])}")
print(f"Test Sequences (All Windows): {len(loaders_w30['test_data']['X'])}")
print(f"Official Test Engines (Terminal Cycles): {len(loaders_w30['terminal_test_indices'])}")"""),

        code_cell("""# Instantiate LSTM Architecture and Load Checkpoint
lstm_model = LSTMRULModel(input_dim=in_dim, hidden_dim=32, num_layers=1, fc_dim=32)
checkpoint_path = Path("../models/deep_learning/lstm/nasa_rul_lstm_FD001_w30.pt")
meta_path = Path("../models/deep_learning/lstm/nasa_rul_lstm_FD001_w30_meta.json")

lstm_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
lstm_model.eval()

with open(meta_path, "r") as f:
    meta = json.load(f)

print("LSTM Architecture:", meta["config"])
print(f"Total Parameters: {meta['size']['total_parameters']:,}")
print(f"Best Training Epoch: {meta['train']['best_epoch']}")"""),

        md_cell("""## 2. Benchmark Evaluation on Official NASA Test Fleet"""),

        code_cell("""eval_results = evaluate_cmapss_rul(
    lstm_model,
    loaders_w30["val_loader"],
    loaders_w30["test_loader"],
    loaders_w30["val_data"]["y_rul"],
    loaders_w30["test_data"]["y_rul"],
    loaders_w30["terminal_test_indices"],
    device="cpu"
)

term_metrics = eval_results["test_terminal_metrics"]
print("=" * 60)
print("NASA C-MAPSS FD001 LSTM TEST FLEET BENCHMARK (N=100)")
print("=" * 60)
print(f"  Test Terminal RMSE:       {term_metrics['test_terminal_rmse']:.2f} cycles")
print(f"  Test Terminal MAE:        {term_metrics['test_terminal_mae']:.2f} cycles")
print(f"  Test Terminal NASA Score: {term_metrics['test_terminal_nasa_score']:.1f}")
print(f"  Test Terminal R2 Score:   {term_metrics['test_terminal_r2']:.4f}")
print("=" * 60)"""),

        md_cell("""## 3. Predicted vs Actual RUL Scatter and Residual Analysis"""),

        code_cell("""y_true = eval_results["y_test_terminal_true"]
y_pred = eval_results["y_test_terminal_pred"]
errors = y_pred - y_true

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Scatter plot
max_val = max(float(np.max(y_true)), float(np.max(y_pred))) + 15
ax1.scatter(y_true, y_pred, color="#1f77b4", edgecolors="k", alpha=0.8, s=50)
ax1.plot([0, max_val], [0, max_val], "r--", linewidth=2, label="Identity line (ideal)")
ax1.set_title("LSTM RUL: Predicted vs Actual (Test Fleet N=100)", fontweight="bold")
ax1.set_xlabel("Official True RUL (Cycles)")
ax1.set_ylabel("Predicted RUL (Cycles)")
ax1.legend(loc="upper left")
ax1.grid(True, linestyle=":", alpha=0.6)

# Residual error distribution
ax2.hist(errors, bins=20, color="#1f77b4", edgecolor="black", alpha=0.7)
ax2.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero Error")
ax2.set_title(f"Residual Distribution (Mean: {errors.mean():.1f}, Std: {errors.std():.1f})", fontweight="bold")
ax2.set_xlabel("Error (Pred - True) in Cycles")
ax2.set_ylabel("Engine Count")
ax2.legend(loc="upper right")
ax2.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()"""),

        md_cell("""## 4. Multi-Cycle Degradation Trajectory Tracking"""),

        code_cell("""# Select test engines to track continuous degradation across cycles
test_uids = loaders_w30["test_data"]["engine_ids"]
test_cycles = loaders_w30["test_data"]["end_cycles"]
test_all_true = loaders_w30["test_data"]["y_rul"]
test_all_pred = eval_results["y_test_all_pred"]

sample_uids = [31, 49, 81]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)

for ax, uid in zip(axes, sample_uids):
    mask = (test_uids == uid)
    ax.plot(test_cycles[mask], test_all_true[mask], "k--", linewidth=2.2, label="True RUL")
    ax.plot(test_cycles[mask], test_all_pred[mask], color="#1f77b4", linewidth=2, label="LSTM Pred")
    ax.set_title(f"Test Engine #{uid} Degradation", fontweight="bold")
    ax.set_xlabel("Operating Cycle")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")

axes[0].set_ylabel("Remaining Useful Life (Cycles)")
plt.tight_layout()
plt.show()""")
    ]
    return make_notebook(cells)


def build_nb_10():
    cells = [
        md_cell("""# 10 - GRU and CNN-LSTM Architectures for RUL Estimation
### Phase 3: Deep Learning Model Development, Training & Evaluation

This notebook evaluates **Gated Recurrent Units (GRU)** and **Hybrid 1D Conv-LSTM (CNN-LSTM)** models for Remaining Useful Life (RUL) estimation on **NASA C-MAPSS**.

#### Key Objectives:
1. Compare GRU parameter efficiency, training speed, and accuracy against LSTM.
2. Evaluate 1D temporal convolution for localized sensor feature extraction coupled with recurrent dynamics.
3. Compare models on the official test fleet terminal cycle benchmark ($N=100$)."""),

        code_cell("""import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path("..").resolve() / "src"))

from deep_learning.dataset_loader import get_cmapss_dataloaders
from deep_learning.gru_model import GRURULModel
from deep_learning.cnn_lstm_model import CNNLSTMRULModel
from deep_learning.evaluate import evaluate_cmapss_rul"""),

        md_cell("""## 1. Load Data and Checkpoints for GRU and CNN-LSTM"""),

        code_cell("""loaders_w30 = get_cmapss_dataloaders("FD001", window_length=30, batch_size=256)
in_dim = loaders_w30["num_features"]

gru_model = GRURULModel(input_dim=in_dim, hidden_dim=32, num_layers=1, fc_dim=32)
gru_model.load_state_dict(torch.load("../models/deep_learning/gru/nasa_rul_gru_FD001_w30.pt", map_location="cpu", weights_only=True))
gru_model.eval()

cnn_lstm_model = CNNLSTMRULModel(input_dim=in_dim, conv_filters=32, kernel_size=3, lstm_hidden_dim=32, lstm_layers=1, fc_dim=32)
cnn_lstm_model.load_state_dict(torch.load("../models/deep_learning/cnn_lstm/nasa_rul_cnn_lstm_FD001_w30.pt", map_location="cpu", weights_only=True))
cnn_lstm_model.eval()

with open("../models/deep_learning/gru/nasa_rul_gru_FD001_w30_meta.json", "r") as f:
    gru_meta = json.load(f)
with open("../models/deep_learning/cnn_lstm/nasa_rul_cnn_lstm_FD001_w30_meta.json", "r") as f:
    cnn_meta = json.load(f)

print(f"GRU Parameters:      {gru_meta['size']['total_parameters']:,} ({gru_meta['size']['model_size_kb']} KB)")
print(f"CNN-LSTM Parameters: {cnn_meta['size']['total_parameters']:,} ({cnn_meta['size']['model_size_kb']} KB)")"""),

        md_cell("""## 2. Official Terminal Test Fleet Benchmark Comparison"""),

        code_cell("""gru_eval = evaluate_cmapss_rul(
    gru_model, loaders_w30["val_loader"], loaders_w30["test_loader"],
    loaders_w30["val_data"]["y_rul"], loaders_w30["test_data"]["y_rul"],
    loaders_w30["terminal_test_indices"], device="cpu"
)

cnn_eval = evaluate_cmapss_rul(
    cnn_lstm_model, loaders_w30["val_loader"], loaders_w30["test_loader"],
    loaders_w30["val_data"]["y_rul"], loaders_w30["test_data"]["y_rul"],
    loaders_w30["terminal_test_indices"], device="cpu"
)

summary_df = pd.DataFrame([
    {
        "Architecture": "GRU Regressor",
        "Parameters": gru_meta["size"]["total_parameters"],
        "Train Time (s)": gru_meta["train"]["train_duration_sec"],
        "Val RMSE": gru_eval["val_metrics"]["val_rmse"],
        "Test Terminal RMSE": gru_eval["test_terminal_metrics"]["test_terminal_rmse"],
        "Test Terminal MAE": gru_eval["test_terminal_metrics"]["test_terminal_mae"],
        "Test NASA Score": gru_eval["test_terminal_metrics"]["test_terminal_nasa_score"],
    },
    {
        "Architecture": "CNN-LSTM Regressor",
        "Parameters": cnn_meta["size"]["total_parameters"],
        "Train Time (s)": cnn_meta["train"]["train_duration_sec"],
        "Val RMSE": cnn_eval["val_metrics"]["val_rmse"],
        "Test Terminal RMSE": cnn_eval["test_terminal_metrics"]["test_terminal_rmse"],
        "Test Terminal MAE": cnn_eval["test_terminal_metrics"]["test_terminal_mae"],
        "Test NASA Score": cnn_eval["test_terminal_metrics"]["test_terminal_nasa_score"],
    }
])
display(summary_df)"""),

        md_cell("""## 3. Comparative Predicted vs Actual RUL"""),

        code_cell("""y_true = gru_eval["y_test_terminal_true"]
y_gru_pred = gru_eval["y_test_terminal_pred"]
y_cnn_pred = cnn_eval["y_test_terminal_pred"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

max_v = max(float(np.max(y_true)), float(np.max(y_gru_pred)), float(np.max(y_cnn_pred))) + 15
ax1.scatter(y_true, y_gru_pred, color="#2ca02c", edgecolors="k", alpha=0.8, s=50)
ax1.plot([0, max_v], [0, max_v], "r--", label="Ideal Identity")
ax1.set_title("GRU: Predicted vs Actual RUL", fontweight="bold")
ax1.set_xlabel("True RUL (Cycles)")
ax1.set_ylabel("Predicted RUL (Cycles)")
ax1.legend()
ax1.grid(True, linestyle=":", alpha=0.6)

ax2.scatter(y_true, y_cnn_pred, color="#d62728", edgecolors="k", alpha=0.8, s=50)
ax2.plot([0, max_v], [0, max_v], "r--", label="Ideal Identity")
ax2.set_title("CNN-LSTM: Predicted vs Actual RUL", fontweight="bold")
ax2.set_xlabel("True RUL (Cycles)")
ax2.legend()
ax2.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()""")
    ]
    return make_notebook(cells)


def build_nb_11():
    cells = [
        md_cell("""# 11 - Early Warning Classification & Imbalance Handling
### Phase 3: Deep Learning Model Development, Training & Evaluation

This notebook investigates **temporal early failure warning classification** across predictive horizons ($H \\in \\{10, 20, 30\\}$ operational cycles) on **NASA C-MAPSS FD001** and binary machine failure classification on **UCI AI4I 2020**.

#### Key Objectives:
1. Formulate recurrent binary failure classifiers using LSTM and GRU backbones.
2. Handle severe class imbalance using loss weighting (`pos_weight` in `BCEWithLogitsLoss`).
3. Tune optimal decision thresholds on validation sets to maximize $F_1$-score.
4. Benchmark Deep Tabular MLP against Phase 2 tree baselines on UCI AI4I 2020."""),

        code_cell("""import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

sys.path.insert(0, str(Path("..").resolve() / "src"))

from deep_learning.dataset_loader import get_cmapss_dataloaders, get_ai4i_dataloaders
from deep_learning.early_warning_model import TemporalEarlyWarningClassifier
from deep_learning.mlp_model import AI4IMLPClassifier
from deep_learning.evaluate import evaluate_classification_model"""),

        md_cell("""## 1. NASA C-MAPSS Early Warning Performance Across Horizons"""),

        code_cell("""exp_df = pd.read_csv("../reports/phase3/experiment_log.csv")
ew_df = exp_df[exp_df["Task"].str.startswith("Early Warning")].copy()
display(ew_df[["Model Name", "Task", "Threshold", "Test Precision", "Test Recall", "Test F1", "Test PR-AUC", "Test ROC-AUC"]])"""),

        md_cell("""## 2. Precision-Recall Curves Across Horizons (10, 20, 30 Cycles)"""),

        code_cell("""# Display saved PR curves figure
from IPython.display import Image
Image(filename="../reports/phase3/figures/06_early_warning_dl_pr_curves.png")"""),

        md_cell("""## 3. UCI AI4I 2020 Deep Tabular MLP Evaluation"""),

        code_cell("""ai4i_row = exp_df[exp_df["Dataset"] == "UCI AI4I 2020"].copy()
display(ai4i_row[["Model Name", "Architecture", "Threshold", "Test Precision", "Test Recall", "Test F1", "Test PR-AUC", "Test ROC-AUC"]])

Image(filename="../reports/phase3/figures/08_ai4i_mlp_curves_cm.png")""")
    ]
    return make_notebook(cells)


def build_nb_12():
    cells = [
        md_cell("""# 12 - Deep Learning vs Machine Learning Baseline Comparison
### Phase 3: Cross-Paradigm Synthesis and Research Analysis

This notebook presents the final **cross-paradigm benchmark comparison**:
- **Traditional Machine Learning (Phase 2):** Linear Regression, Random Forest, Gradient Boosting, XGBoost.
- **Deep Learning Architectures (Phase 3):** LSTM, GRU, CNN-LSTM, Deep Tabular MLP.

#### Key Research Findings:
1. **Stationary vs Non-Stationary Dynamics:** How recurrent gating compares to gradient-boosted decision trees on temporal sensor degradation.
2. **Asymmetric NASA Scoring Function:** Evaluating penalties on early vs late failure estimation.
3. **Data Efficiency & Computational Trade-Offs:** Parameter footprints, inference latency, and hardware compatibility."""),

        code_cell("""import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display

p2_log = pd.read_csv("../reports/phase2/experiment_log.csv")
p3_log = pd.read_csv("../reports/phase3/experiment_log.csv")

print(f"Phase 2 Logged Experiments: {len(p2_log)}")
print(f"Phase 3 Logged Experiments: {len(p3_log)}")"""),

        md_cell("""## 1. NASA C-MAPSS FD001 RUL Benchmark: ML Baselines vs Deep Learning"""),

        code_cell("""# Extract FD001 RUL models from Phase 2 and Phase 3
p2_rul = p2_log[(p2_log["Subset"] == "FD001") & (p2_log["Task"] == "RUL Regression")].copy()
p3_rul = p3_log[(p3_log["Subset"] == "FD001") & (p3_log["Task"] == "RUL Regression")].copy()

p2_summary = pd.DataFrame({
    "Paradigm": "Traditional ML",
    "Model Name": p2_rul["Model Name"],
    "Val RMSE": p2_rul["Val RMSE"],
    "Test Terminal RMSE": p2_rul["Test Terminal RMSE"],
    "Test Terminal MAE": p2_rul["Test Terminal MAE"],
    "Test NASA Score": p2_rul["Test Terminal NASA Score"],
})

p3_summary = pd.DataFrame({
    "Paradigm": "Deep Learning",
    "Model Name": p3_rul["Model Name"],
    "Val RMSE": p3_rul["Val RMSE"],
    "Test Terminal RMSE": p3_rul["Test Terminal RMSE"],
    "Test Terminal MAE": p3_rul["Test Terminal MAE"],
    "Test NASA Score": p3_rul["Test Terminal NASA Score"],
})

comparison_table = pd.concat([p2_summary, p3_summary], ignore_index=True)
display(comparison_table.sort_values(by="Test Terminal RMSE"))"""),

        md_cell("""## 2. Cross-Paradigm Comparison Figure"""),

        code_cell("""Image(filename="../reports/phase3/figures/09_ml_vs_dl_comparison_bar.png")"""),

        md_cell("""## 3. UCI AI4I 2020: Tree Baselines vs Deep MLP"""),

        code_cell("""p2_ai4i = p2_log[p2_log["Dataset"] == "UCI AI4I 2020"].copy()
p3_ai4i = p3_log[p3_log["Dataset"] == "UCI AI4I 2020"].copy()

ai4i_comp = pd.concat([
    pd.DataFrame({
        "Paradigm": "Traditional ML",
        "Model Name": p2_ai4i["Model Name"],
        "Precision": p2_ai4i["Test Precision"],
        "Recall": p2_ai4i["Test Recall"],
        "F1-Score": p2_ai4i["Test F1"],
        "PR-AUC": p2_ai4i["Test PR-AUC"],
        "ROC-AUC": p2_ai4i["Test ROC-AUC"],
    }),
    pd.DataFrame({
        "Paradigm": "Deep Learning",
        "Model Name": p3_ai4i["Model Name"],
        "Precision": p3_ai4i["Test Precision"],
        "Recall": p3_ai4i["Test Recall"],
        "F1-Score": p3_ai4i["Test F1"],
        "PR-AUC": p3_ai4i["Test PR-AUC"],
        "ROC-AUC": p3_ai4i["Test ROC-AUC"],
    })
], ignore_index=True)

display(ai4i_comp.sort_values(by="PR-AUC", ascending=False))""")
    ]
    return make_notebook(cells)


def generate_all_notebooks():
    nb_dir = Path("notebooks")
    nb_dir.mkdir(parents=True, exist_ok=True)

    notebooks = {
        "09_lstm_rul_prediction.ipynb": build_nb_09(),
        "10_gru_rul_prediction.ipynb": build_nb_10(),
        "11_early_warning_classification.ipynb": build_nb_11(),
        "12_deep_learning_comparison.ipynb": build_nb_12(),
    }

    for name, nb in notebooks.items():
        path = nb_dir / name
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print(f"[OK] Generated notebook: {path}")


if __name__ == "__main__":
    generate_all_notebooks()
