"""
Phase 4 Jupyter Notebook Generator.
Creates notebooks/13_multivariate_anomaly_detection.ipynb.
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


def generate_phase4_notebook():
    cells = [
        md_cell("""# Phase 4: Multivariate Sensor Anomaly Detection
## Agentic Predictive Maintenance Framework

**Objective:**
This notebook demonstrates the unsupervised and semi-supervised multivariate sensor anomaly detection module developed in Phase 4.
It explores:
1. **NASA C-MAPSS FD001:** Nominal baseline investigation, Multivariate LSTM Autoencoder reconstruction, Isolation Forest, One-Class SVM, and LOF detectors.
2. **Engine Lifespan Trajectories:** Progression of anomaly scores across engine operating lifespans compared against ground truth RUL degradation.
3. **Sensor-Level Abnormality Attribution:** Identification of top deviant sensor channels causing high reconstruction error.
4. **UCI AI4I 2020:** Unsupervised anomaly detection, threshold calibration, and post-hoc benchmark evaluation against machine breakdown events.
5. **Calibrated Severity Framework:** Multi-tier operational severity mapping (`Normal`, `Low anomaly`, `Medium anomaly`, `High anomaly`).
6. **Unified Inference API:** Demonstration of the `predict_anomaly(input_data)` endpoint."""),

        code_cell("""import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set path to root
ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.anomaly_detection import (
    predict_anomaly,
    LSTMAutoencoderDetector,
    IsolationForestDetector,
    OneClassSVMDetector,
    LOFDetector,
    DenseAutoencoderDetector,
    SeverityClassifier,
    SensorAttributor
)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
print("Phase 4 anomaly detection package successfully imported!")"""),

        md_cell("""## 1. NASA C-MAPSS: Dataset Loading & Nominal Baseline Investigation
We load the multivariate sliding window sequences ($w=30, D=153$) and inspect early-life sensor behaviour ($RUL \\ge 110$) as a reference for nominal operation."""),

        code_cell("""nasa_dir = os.path.join(ROOT_DIR, "data", "processed", "nasa", "FD001")
train_npz = np.load(os.path.join(nasa_dir, "sequences_w30_train.npz"))
val_npz = np.load(os.path.join(nasa_dir, "sequences_w30_val.npz"))

X_train = train_npz["X"]
y_train_rul = train_npz["y_rul"]
feature_names = list(train_npz["feature_names"])

X_val = val_npz["X"]
y_val_rul = val_npz["y_rul"]
val_engine_ids = val_npz["engine_ids"]
val_end_cycles = val_npz["end_cycles"]

print(f"Train sequences: {X_train.shape}")
print(f"Validation sequences: {X_val.shape}")
print(f"Sensor features: {len(feature_names)}")

# Investigate early-life nominal sequences
nominal_mask = y_train_rul >= 110
print(f"Nominal training sequences (RUL >= 110): {np.sum(nominal_mask)} / {len(y_train_rul)} ({100*np.mean(nominal_mask):.1f}%)")"""),

        md_cell("""## 2. Multivariate LSTM Autoencoder
We load the trained PyTorch LSTM Autoencoder, inspect its reconstruction loss, and compute anomaly scores across nominal, moderate, and severe degradation regimes."""),

        code_cell("""lstm_ae = LSTMAutoencoderDetector(model_id="lstm_autoencoder", dataset_name="FD001")
model_path = os.path.join(ROOT_DIR, "models", "anomaly_detection", "lstm_autoencoder", "lstm_autoencoder_FD001.pt")
lstm_ae.load(model_path)

print(f"Loaded {lstm_ae.model_id} from {model_path}")
print(f"Calibrated Thresholds:")
for k, v in lstm_ae.thresholds.items():
    print(f"  {k:8s}: {v:.6f}")

# Score validation set
val_scores = lstm_ae.score(X_val)

tier_nominal = val_scores[y_val_rul >= 110]
tier_moderate = val_scores[(y_val_rul >= 30) & (y_val_rul < 110)]
tier_severe = val_scores[y_val_rul < 30]

print(f"Mean Score - Nominal Life:   {np.mean(tier_nominal):.6f}")
print(f"Mean Score - Moderate Wear:  {np.mean(tier_moderate):.6f}")
print(f"Mean Score - Severe Wear:    {np.mean(tier_severe):.6f}")"""),

        code_cell("""# Plot score distributions across degradation regimes
plt.figure(figsize=(9, 4.5))
sns.kdeplot(tier_nominal, label=f"Nominal (RUL >= 110, N={len(tier_nominal)})", color="#2ecc71", fill=True, alpha=0.35)
sns.kdeplot(tier_moderate, label=f"Moderate Degradation (30 <= RUL < 110, N={len(tier_moderate)})", color="#f39c12", fill=True, alpha=0.35)
sns.kdeplot(tier_severe, label=f"Severe Degradation (RUL < 30, N={len(tier_severe)})", color="#e74c3c", fill=True, alpha=0.35)
plt.axvline(lstm_ae.thresholds["low"], color="#3498db", linestyle="--", label=f"Low Th: {lstm_ae.thresholds['low']:.3f}")
plt.axvline(lstm_ae.thresholds["medium"], color="#e67e22", linestyle="--", label=f"Medium Th: {lstm_ae.thresholds['medium']:.3f}")
plt.axvline(lstm_ae.thresholds["high"], color="#9b59b6", linestyle="--", label=f"High Th: {lstm_ae.thresholds['high']:.3f}")
plt.title("LSTM Autoencoder Reconstruction Score Distributions Across Degradation Stages", fontweight="bold")
plt.xlabel("Reconstruction MSE Anomaly Score")
plt.ylabel("Density")
plt.xlim(0, float(np.percentile(val_scores, 99.8)))
plt.legend()
plt.show()"""),

        md_cell("""## 3. Engine-Level Degradation Trajectory
We track the anomaly score over the entire operational lifecycle of a validation engine, observing how the score escalates as degradation advances and RUL approaches 0."""),

        code_cell("""target_engine = val_engine_ids[0]
mask = val_engine_ids == target_engine
cycles = val_end_cycles[mask]
scores = val_scores[mask]
ruls = y_val_rul[mask]

fig, ax1 = plt.subplots(figsize=(10, 4.5))
ax1.plot(cycles, scores, color="#2980b9", linewidth=2.0, label="Reconstruction Anomaly Score")
ax1.axhline(lstm_ae.thresholds["low"], color="#27ae60", linestyle=":", label="Low Anomaly Th")
ax1.axhline(lstm_ae.thresholds["medium"], color="#f39c12", linestyle="--", label="Medium Anomaly Th")
ax1.axhline(lstm_ae.thresholds["high"], color="#c0392b", linestyle="-.", label="High Anomaly Th")
ax1.set_xlabel("Operational Cycle")
ax1.set_ylabel("LSTM Reconstruction MSE", color="#2980b9")
ax1.legend(loc="upper left")

ax2 = ax1.twinx()
ax2.plot(cycles, ruls, color="#7f8c8d", linestyle="--", alpha=0.6, label="Ground Truth RUL")
ax2.set_ylabel("Remaining Useful Life (Cycles)", color="#7f8c8d")
ax2.grid(False)

plt.title(f"Engine #{target_engine}: Anomaly Score Trajectory vs Ground Truth Degradation", fontweight="bold")
plt.show()"""),

        md_cell("""## 4. Sensor-Level Abnormality Attribution
Which sensor channels are deviating most from nominal patterns during severe degradation?
We compute feature-wise reconstruction errors to rank top contributing sensors."""),

        code_cell("""attributor = SensorAttributor(feature_names)
feat_errors = lstm_ae.feature_errors(X_val)

# Severe wear attribution
severe_mask = y_val_rul < 30
severe_feat_errors = np.mean(feat_errors[severe_mask], axis=0)
attrib_res = attributor.attribute_reconstruction_error(severe_feat_errors, top_k=5)

df_attrib = pd.DataFrame(attrib_res["top_contributing_sensors"])
print("Top 5 Sensors Contributing to Severe Wear Reconstruction Error:")
display(df_attrib)

plt.figure(figsize=(8, 4))
plt.barh(df_attrib["sensor"][::-1], df_attrib["contribution_pct"][::-1], color="#34495e")
plt.title("Sensor-Level Attribution Breakdown (Severe Wear)", fontweight="bold")
plt.xlabel("Contribution Percentage (%)")
plt.show()"""),

        md_cell("""## 5. UCI AI4I 2020: Unsupervised Anomaly Detection & Post-Hoc Evaluation
We evaluate unsupervised anomaly detectors (Isolation Forest, One-Class SVM, LOF, Dense Autoencoder) against actual machine failures on the test set."""),

        code_cell("""ai4i_dir = os.path.join(ROOT_DIR, "data", "processed", "ai4i")
X_test_ai4i = pd.read_csv(os.path.join(ai4i_dir, "X_test.csv")).values
y_test_ai4i = pd.read_csv(os.path.join(ai4i_dir, "y_test.csv")).values.ravel()

# Load experiment metrics from phase 4
metrics_path = os.path.join(ROOT_DIR, "models", "anomaly_detection", "experiment_metrics.json")
with open(metrics_path, "r") as f:
    metrics = json.load(f)

ai4i_eval = metrics["ai4i_2020"]
df_ai4i = pd.DataFrame.from_dict(ai4i_eval, orient="index")[["precision", "recall", "f1_score", "pr_auc", "roc_auc"]]
print("AI4I 2020 Post-Hoc Benchmark Results:")
display(df_ai4i)"""),

        md_cell("""## 6. Unified Inference Interface: `predict_anomaly()`
We demonstrate the consistent API function `predict_anomaly(input_data)` for both temporal sequence windows and tabular records."""),

        code_cell("""# Example 1: Nominal sequence from NASA validation set
sample_nominal = X_val[0]
res_nominal = predict_anomaly(sample_nominal, model_id="lstm_autoencoder")
print("=== NASA Nominal Sequence Prediction ===")
print(f"Anomaly Score: {res_nominal['anomaly_score']}")
print(f"Status:        {'ANOMALOUS' if res_nominal['is_anomaly'] else 'NORMAL'}")
print(f"Severity Tier: {res_nominal['severity']}")
print(f"Model:         {res_nominal['model_identifier']}")

# Example 2: Severe wear sequence near engine failure
sample_severe = X_val[severe_mask][0]
res_severe = predict_anomaly(sample_severe, model_id="lstm_autoencoder")
print("\n=== NASA Severe Degradation Sequence Prediction ===")
print(f"Anomaly Score: {res_severe['anomaly_score']}")
print(f"Status:        {'ANOMALOUS' if res_severe['is_anomaly'] else 'NORMAL'}")
print(f"Severity Tier: {res_severe['severity']}")
print("Top Contributing Sensors:")
for s in res_severe["relevant_sensor_metadata"]["top_contributing_sensors"][:3]:
    print(f"  - {s['sensor']}: {s['contribution_pct']}% of error")

# Example 3: Tabular input for AI4I
sample_tabular = X_test_ai4i[0]
res_tabular = predict_anomaly(sample_tabular, model_id="isolation_forest", dataset="AI4I")
print("\n=== AI4I Tabular Record Prediction ===")
print(f"Anomaly Score: {res_tabular['anomaly_score']}")
print(f"Status:        {'ANOMALOUS' if res_tabular['is_anomaly'] else 'NORMAL'}")
print(f"Severity Tier: {res_tabular['severity']}")""")
    ]

    nb = make_notebook(cells)
    out_path = os.path.join(ROOT_DIR, "notebooks", "13_multivariate_anomaly_detection.ipynb")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Successfully generated notebook: {out_path}")


if __name__ == "__main__":
    generate_phase4_notebook()
