"""
Phase 4: Multivariate Sensor Anomaly Detection Execution Script.
Executes end-to-end model training, threshold calibration, degradation trajectory analysis,
sensor attribution, post-hoc benchmark evaluation, visualization generation, and artifact serialization.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix

# Ensure project root is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.anomaly_detection.traditional import IsolationForestDetector, OneClassSVMDetector, LOFDetector
from src.anomaly_detection.autoencoder import LSTMAutoencoderDetector, DenseAutoencoderDetector
from src.anomaly_detection.severity import SeverityClassifier
from src.anomaly_detection.attribution import SensorAttributor
from src.anomaly_detection.evaluation import evaluate_against_labels, compute_score_distribution

# Set publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.size": 10, "figure.autolayout": True})


def setup_directories():
    paths = [
        os.path.join(BASE_DIR, "models", "anomaly_detection", "isolation_forest"),
        os.path.join(BASE_DIR, "models", "anomaly_detection", "one_class_svm"),
        os.path.join(BASE_DIR, "models", "anomaly_detection", "lof"),
        os.path.join(BASE_DIR, "models", "anomaly_detection", "autoencoder"),
        os.path.join(BASE_DIR, "models", "anomaly_detection", "lstm_autoencoder"),
        os.path.join(BASE_DIR, "reports", "phase4", "figures"),
        os.path.join(BASE_DIR, "reports", "figures", "phase4"),
    ]
    for p in paths:
        os.makedirs(p, exist_ok=True)


def run_phase4():
    print("=" * 80)
    print("STARTING PHASE 4: MULTIVARIATE SENSOR ANOMALY DETECTION")
    print("=" * 80)
    setup_directories()
    t_start = time.time()

    metrics_summary = {
        "nasa_c_mapss": {},
        "ai4i_2020": {},
        "thresholds": {},
        "execution_time_seconds": 0.0
    }

    # =========================================================================
    # PART 1: NASA C-MAPSS ANOMALY DETECTION (FD001)
    # =========================================================================
    print("\n--- 1. Loading NASA C-MAPSS FD001 Sequence Datasets ---")
    nasa_dir = os.path.join(BASE_DIR, "data", "processed", "nasa", "FD001")
    train_npz = np.load(os.path.join(nasa_dir, "sequences_w30_train.npz"))
    val_npz = np.load(os.path.join(nasa_dir, "sequences_w30_val.npz"))
    test_npz = np.load(os.path.join(nasa_dir, "sequences_w30_test.npz"))

    X_train_seq = train_npz["X"]  # (14020, 30, 153)
    y_train_rul = train_npz["y_rul"]
    train_engine_ids = train_npz["engine_ids"]
    train_end_cycles = train_npz["end_cycles"]
    feature_names = list(train_npz["feature_names"])

    X_val_seq = val_npz["X"]  # (3711, 30, 153)
    y_val_rul = val_npz["y_rul"]
    val_engine_ids = val_npz["engine_ids"]
    val_end_cycles = val_npz["end_cycles"]

    X_test_seq = test_npz["X"]  # (10196, 30, 153)
    test_engine_ids = test_npz["engine_ids"]
    test_end_cycles = test_npz["end_cycles"]

    print(f"NASA Train Sequences: {X_train_seq.shape}")
    print(f"NASA Val Sequences:   {X_val_seq.shape}")
    print(f"NASA Test Sequences:  {X_test_seq.shape}")
    print(f"Feature count:        {len(feature_names)}")

    # Investigate early-life sensor behaviour as reference
    print("\n--- 1.1 Investigating Early-Life Nominal Sensor Sequences ---")
    # Engines operate nominally in early life; define nominal reference as sequences where RUL >= 110
    nominal_train_mask = y_train_rul >= 110
    nominal_val_mask = y_val_rul >= 110

    X_train_nominal = X_train_seq[nominal_train_mask]
    X_val_nominal = X_val_seq[nominal_val_mask]

    print(f"Nominal training sequences (RUL >= 110):   {len(X_train_nominal)} / {len(X_train_seq)} ({100*len(X_train_nominal)/len(X_train_seq):.1f}%)")
    print(f"Nominal validation sequences (RUL >= 110): {len(X_val_nominal)} / {len(X_val_seq)} ({100*len(X_val_nominal)/len(X_val_seq):.1f}%)")

    # Check nominal sequence variance to confirm stability
    nominal_mean = np.mean(X_train_nominal, axis=(0, 1))
    nominal_std = np.std(X_train_nominal, axis=(0, 1))
    print(f"Nominal sensor features mean range: [{nominal_mean.min():.4f}, {nominal_mean.max():.4f}]")
    print(f"Nominal sensor features std range:  [{nominal_std.min():.4f}, {nominal_std.max():.4f}]")

    # Subsample nominal train for traditional models to ensure rapid CPU fitting
    np.random.seed(42)
    sample_idx = np.random.choice(len(X_train_nominal), size=min(4000, len(X_train_nominal)), replace=False)
    X_train_trad_fit = X_train_nominal[sample_idx]

    # --- 1.2 Training NASA Anomaly Detectors ---
    print("\n--- 1.2 Training NASA Anomaly Detectors ---")

    # A. Multivariate LSTM Autoencoder (PyTorch)
    print("Training Multivariate LSTM Autoencoder (20 epochs)...")
    lstm_ae = LSTMAutoencoderDetector(
        model_id="lstm_autoencoder",
        dataset_name="FD001",
        input_dim=len(feature_names),
        hidden_dim=64,
        latent_dim=32,
        lr=1e-3,
        batch_size=128,
        epochs=20
    )
    lstm_ae.fit(X_train_nominal, feature_names=feature_names)
    print(f"  Final Epoch Loss: {lstm_ae.training_history[-1]:.6f}")

    # B. Isolation Forest
    print("Training Isolation Forest on nominal temporal features...")
    iforest_nasa = IsolationForestDetector(
        model_id="isolation_forest",
        dataset_name="FD001",
        n_estimators=100,
        contamination=0.05,
        random_state=42
    )
    iforest_nasa.fit(X_train_trad_fit, feature_names=feature_names)

    # C. One-Class SVM
    print("Training One-Class SVM on nominal temporal features...")
    ocsvm_nasa = OneClassSVMDetector(
        model_id="one_class_svm",
        dataset_name="FD001",
        kernel="rbf",
        gamma="scale",
        nu=0.05
    )
    ocsvm_nasa.fit(X_train_trad_fit, feature_names=feature_names)

    # D. Local Outlier Factor (Novelty Mode)
    print("Fitting Local Outlier Factor (novelty=True)...")
    lof_nasa = LOFDetector(
        model_id="lof",
        dataset_name="FD001",
        n_neighbors=20,
        contamination=0.05
    )
    lof_nasa.fit(X_train_trad_fit, feature_names=feature_names)

    # Attach nominal metadata for statistical attribution
    for m in [iforest_nasa, ocsvm_nasa, lof_nasa]:
        m.metadata["nominal_mean"] = nominal_mean.tolist()
        m.metadata["nominal_std"] = nominal_std.tolist()

    # --- 1.3 Threshold Calibration on Validation Nominal Data ---
    print("\n--- 1.3 Calibrating Severity Thresholds on Validation Nominal Sequences ---")
    nasa_detectors = {
        "lstm_autoencoder": lstm_ae,
        "isolation_forest": iforest_nasa,
        "one_class_svm": ocsvm_nasa,
        "lof": lof_nasa
    }

    calibrated_nasa_thresholds = {}
    for name, detector in nasa_detectors.items():
        # Score nominal validation sequences
        nom_val_scores = detector.score(X_val_nominal)
        th_dict = detector.calibrate_thresholds(nom_val_scores, percentiles=(90.0, 97.5, 99.0))
        calibrated_nasa_thresholds[name] = th_dict
        print(f"  {name:18s} Thresholds -> Low (p90): {th_dict['low']:.4f} | Med (p97.5): {th_dict['medium']:.4f} | High (p99): {th_dict['high']:.4f}")

    metrics_summary["thresholds"]["FD001"] = calibrated_nasa_thresholds

    # --- 1.4 Evaluating Anomaly Trajectories & Degradation Alignment ---
    print("\n--- 1.4 Evaluating Engine Degradation and Anomaly Progression ---")
    # Compute scores across ALL validation sequences
    val_scores_lstm = lstm_ae.score(X_val_seq)
    val_scores_iforest = iforest_nasa.score(X_val_seq)
    val_scores_ocsvm = ocsvm_nasa.score(X_val_seq)
    val_scores_lof = lof_nasa.score(X_val_seq)

    # Correlation with engine degradation (as RUL decreases, anomaly score should increase)
    degradation_indicator = -y_val_rul
    corr_lstm = float(np.corrcoef(val_scores_lstm, degradation_indicator)[0, 1])
    corr_iforest = float(np.corrcoef(val_scores_iforest, degradation_indicator)[0, 1])
    corr_ocsvm = float(np.corrcoef(val_scores_ocsvm, degradation_indicator)[0, 1])
    corr_lof = float(np.corrcoef(val_scores_lof, degradation_indicator)[0, 1])

    print(f"  Correlation with Degradation (-RUL):")
    print(f"    LSTM Autoencoder: r = {corr_lstm:.4f}")
    print(f"    Isolation Forest: r = {corr_iforest:.4f}")
    print(f"    One-Class SVM:    r = {corr_ocsvm:.4f}")
    print(f"    LOF:              r = {corr_lof:.4f}")

    # Evaluate score distributions across degradation tiers:
    # 1. Nominal (RUL >= 110)
    # 2. Moderate Wear (30 <= RUL < 110)
    # 3. Severe Wear (RUL < 30)
    tier_nominal = val_scores_lstm[y_val_rul >= 110]
    tier_moderate = val_scores_lstm[(y_val_rul >= 30) & (y_val_rul < 110)]
    tier_severe = val_scores_lstm[y_val_rul < 30]

    nasa_dist_metrics = {
        "nominal_mean_score": round(float(np.mean(tier_nominal)), 6),
        "moderate_mean_score": round(float(np.mean(tier_moderate)), 6),
        "severe_mean_score": round(float(np.mean(tier_severe)), 6),
        "correlation_with_degradation": {
            "lstm_autoencoder": round(corr_lstm, 4),
            "isolation_forest": round(corr_iforest, 4),
            "one_class_svm": round(corr_ocsvm, 4),
            "lof": round(corr_lof, 4)
        }
    }
    metrics_summary["nasa_c_mapss"] = nasa_dist_metrics
    print(f"  LSTM Score Mean: Nominal={nasa_dist_metrics['nominal_mean_score']:.4f} | Moderate={nasa_dist_metrics['moderate_mean_score']:.4f} | Severe={nasa_dist_metrics['severe_mean_score']:.4f}")

    # --- 1.5 Sensor Attribution Analysis ---
    print("\n--- 1.5 Sensor-Level Abnormality Analysis ---")
    attributor = SensorAttributor(feature_names)
    feat_errors = lstm_ae.feature_errors(X_val_seq)

    # Average feature error in severe wear sequences
    severe_mask = y_val_rul < 30
    mean_severe_feat_errors = np.mean(feat_errors[severe_mask], axis=0)
    severe_attrib = attributor.attribute_reconstruction_error(mean_severe_feat_errors, top_k=5)
    print("  Top 5 Sensors Contributing to Severe Wear Degradation:")
    for item in severe_attrib["top_contributing_sensors"]:
        print(f"    Sensor: {item['sensor']:15s} Error: {item['error']:.6f} ({item['contribution_pct']:.1f}%)")

    # =========================================================================
    # PART 2: UCI AI4I 2020 ANOMALY DETECTION
    # =========================================================================
    print("\n" + "=" * 80)
    print("--- 2. Loading UCI AI4I 2020 Dataset ---")
    ai4i_dir = os.path.join(BASE_DIR, "data", "processed", "ai4i")
    X_train_ai4i = pd.read_csv(os.path.join(ai4i_dir, "X_train.csv")).values
    y_train_ai4i = pd.read_csv(os.path.join(ai4i_dir, "y_train.csv")).values.ravel()
    X_test_ai4i = pd.read_csv(os.path.join(ai4i_dir, "X_test.csv")).values
    y_test_ai4i = pd.read_csv(os.path.join(ai4i_dir, "y_test.csv")).values.ravel()

    with open(os.path.join(ai4i_dir, "metadata.json"), "r") as f:
        ai4i_meta = json.load(f)
    ai4i_feature_names = ai4i_meta.get("feature_columns", [f"feat_{i}" for i in range(X_train_ai4i.shape[1])])

    print(f"AI4I Train shape: {X_train_ai4i.shape} (Unsupervised fitting without failure labels)")
    print(f"AI4I Test shape:  {X_test_ai4i.shape} (Post-hoc evaluation on {np.sum(y_test_ai4i)} true failures)")

    # --- 2.1 Unsupervised Training on AI4I ---
    print("\n--- 2.1 Training Unsupervised AI4I Anomaly Detectors ---")

    # A. Isolation Forest
    print("Fitting Isolation Forest on AI4I...")
    iforest_ai4i = IsolationForestDetector(
        model_id="isolation_forest",
        dataset_name="AI4I",
        n_estimators=100,
        contamination=0.034,
        random_state=42
    )
    iforest_ai4i.fit(X_train_ai4i, feature_names=ai4i_feature_names)

    # B. One-Class SVM
    print("Fitting One-Class SVM on AI4I...")
    ocsvm_ai4i = OneClassSVMDetector(
        model_id="one_class_svm",
        dataset_name="AI4I",
        kernel="rbf",
        gamma="scale",
        nu=0.034
    )
    ocsvm_ai4i.fit(X_train_ai4i, feature_names=ai4i_feature_names)

    # C. Local Outlier Factor
    print("Fitting Local Outlier Factor on AI4I...")
    lof_ai4i = LOFDetector(
        model_id="lof",
        dataset_name="AI4I",
        n_neighbors=20,
        contamination=0.034
    )
    lof_ai4i.fit(X_train_ai4i, feature_names=ai4i_feature_names)

    # D. Dense Tabular Autoencoder
    print("Fitting Dense Autoencoder (25 epochs) on AI4I...")
    dense_ae = DenseAutoencoderDetector(
        model_id="dense_autoencoder",
        dataset_name="AI4I",
        input_dim=X_train_ai4i.shape[1],
        hidden_dim=32,
        latent_dim=8,
        lr=1e-3,
        batch_size=128,
        epochs=25
    )
    dense_ae.fit(X_train_ai4i, feature_names=ai4i_feature_names)
    print(f"  Final Epoch Loss: {dense_ae.training_history[-1]:.6f}")

    # Attach statistical metadata for AI4I
    ai4i_nominal_mean = np.mean(X_train_ai4i, axis=0)
    ai4i_nominal_std = np.std(X_train_ai4i, axis=0)
    for m in [iforest_ai4i, ocsvm_ai4i, lof_ai4i]:
        m.metadata["nominal_mean"] = ai4i_nominal_mean.tolist()
        m.metadata["nominal_std"] = ai4i_nominal_std.tolist()

    # --- 2.2 Calibrate AI4I Thresholds ---
    print("\n--- 2.2 Calibrating AI4I Severity Thresholds ---")
    ai4i_detectors = {
        "isolation_forest": iforest_ai4i,
        "one_class_svm": ocsvm_ai4i,
        "lof": lof_ai4i,
        "dense_autoencoder": dense_ae
    }

    calibrated_ai4i_thresholds = {}
    for name, detector in ai4i_detectors.items():
        train_scores = detector.score(X_train_ai4i)
        th_dict = detector.calibrate_thresholds(train_scores, percentiles=(90.0, 96.6, 99.0))
        calibrated_ai4i_thresholds[name] = th_dict
        print(f"  {name:18s} Thresholds -> Low (p90): {th_dict['low']:.4f} | Med (p96.6): {th_dict['medium']:.4f} | High (p99): {th_dict['high']:.4f}")

    metrics_summary["thresholds"]["AI4I"] = calibrated_ai4i_thresholds

    # --- 2.3 Post-Hoc Evaluation on AI4I Test Set ---
    print("\n--- 2.3 Post-Hoc Benchmark Evaluation on AI4I Test Set ---")
    ai4i_results = {}
    ai4i_test_scores = {}

    for name, detector in ai4i_detectors.items():
        test_scores = detector.score(X_test_ai4i)
        ai4i_test_scores[name] = test_scores
        th = detector.thresholds["medium"]
        eval_metrics = evaluate_against_labels(y_test_ai4i, test_scores, threshold=th)
        ai4i_results[name] = eval_metrics
        print(f"  {name:18s} -> Precision: {eval_metrics['precision']:.4f} | Recall: {eval_metrics['recall']:.4f} | F1: {eval_metrics['f1_score']:.4f} | PR-AUC: {eval_metrics['pr_auc']:.4f} | ROC-AUC: {eval_metrics['roc_auc']:.4f}")

    metrics_summary["ai4i_2020"] = ai4i_results

    # =========================================================================
    # PART 3: GENERATE ALL 8 PUBLICATION VISUALIZATIONS
    # =========================================================================
    print("\n" + "=" * 80)
    print("--- 3. Generating Phase 4 Publication Visualizations ---")
    fig_dirs = [
        os.path.join(BASE_DIR, "reports", "phase4", "figures"),
        os.path.join(BASE_DIR, "reports", "figures", "phase4")
    ]

    # Helper function to save figure to both locations
    def save_plot(fig, filename):
        for d in fig_dirs:
            fig.savefig(os.path.join(d, filename), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved figure: {filename}")

    # Figure 1: NASA Anomaly Score Distribution (Nominal vs Moderate vs Severe)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.kdeplot(tier_nominal, ax=ax, label=f"Nominal Life (RUL >= 110, N={len(tier_nominal)})", color="#2ecc71", fill=True, alpha=0.35)
    sns.kdeplot(tier_moderate, ax=ax, label=f"Moderate Degradation (30 <= RUL < 110, N={len(tier_moderate)})", color="#f39c12", fill=True, alpha=0.35)
    sns.kdeplot(tier_severe, ax=ax, label=f"Severe Degradation (RUL < 30, N={len(tier_severe)})", color="#e74c3c", fill=True, alpha=0.35)
    ax.axvline(lstm_ae.thresholds["low"], color="#3498db", linestyle="--", linewidth=1.5, label=f"Low Th (p90): {lstm_ae.thresholds['low']:.3f}")
    ax.axvline(lstm_ae.thresholds["medium"], color="#e67e22", linestyle="--", linewidth=1.5, label=f"Medium Th (p97.5): {lstm_ae.thresholds['medium']:.3f}")
    ax.axvline(lstm_ae.thresholds["high"], color="#9b59b6", linestyle="--", linewidth=1.5, label=f"High Th (p99): {lstm_ae.thresholds['high']:.3f}")
    ax.set_title("NASA C-MAPSS FD001: LSTM Autoencoder Score Distribution Across Degradation Horizons", fontsize=12, fontweight="bold")
    ax.set_xlabel("Reconstruction MSE Anomaly Score")
    ax.set_ylabel("Density")
    ax.set_xlim(0, float(np.percentile(val_scores_lstm, 99.8)))
    ax.legend(loc="upper right", frameon=True)
    save_plot(fig, "01_nasa_anomaly_score_distribution.png")

    # Figure 2: NASA Engine Degradation Trajectories
    # Select 3 representative validation engines: e.g. engines with IDs in val_engine_ids
    unique_val_engines = np.unique(val_engine_ids)
    selected_engines = unique_val_engines[:3]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for idx, eng_id in enumerate(selected_engines):
        eng_mask = val_engine_ids == eng_id
        cycles = val_end_cycles[eng_mask]
        scores = val_scores_lstm[eng_mask]
        rul_vals = y_val_rul[eng_mask]

        ax = axes[idx]
        ax.plot(cycles, scores, color="#2980b9", linewidth=2.0, label="Anomaly Score")
        ax.axhline(lstm_ae.thresholds["low"], color="#27ae60", linestyle=":", label="Low Anomaly Th")
        ax.axhline(lstm_ae.thresholds["medium"], color="#f39c12", linestyle="--", label="Medium Anomaly Th")
        ax.axhline(lstm_ae.thresholds["high"], color="#c0392b", linestyle="-.", label="High Anomaly Th")

        # Overlay RUL on secondary axis
        ax_sec = ax.twinx()
        ax_sec.plot(cycles, rul_vals, color="#7f8c8d", linestyle="--", alpha=0.5, label="True RUL")
        if idx == 2:
            ax_sec.set_ylabel("Remaining Useful Life (Cycles)", color="#7f8c8d")
        ax_sec.grid(False)

        ax.set_title(f"Validation Engine #{eng_id} Lifecycle", fontsize=11, fontweight="bold")
        ax.set_xlabel("Operational Cycle")
        if idx == 0:
            ax.set_ylabel("LSTM Reconstruction MSE")
            ax.legend(loc="upper left", fontsize=8)

    plt.suptitle("Engine-Level Anomaly Score Trajectories Across Operating Lifespan", fontsize=13, fontweight="bold", y=1.03)
    save_plot(fig, "02_nasa_engine_trajectories.png")

    # Figure 3: NASA Sensor Attribution Breakdown
    fig, ax = plt.subplots(figsize=(10, 5))
    top_sensors = severe_attrib["top_contributing_sensors"]
    sensor_names = [s["sensor"] for s in top_sensors]
    contrib_pcts = [s["contribution_pct"] for s in top_sensors]
    errors = [s["error"] for s in top_sensors]

    bars = ax.barh(sensor_names[::-1], contrib_pcts[::-1], color="#34495e", edgecolor="#2c3e50", height=0.6)
    for bar, err in zip(bars, errors[::-1]):
        w = bar.get_width()
        ax.text(w + 0.8, bar.get_y() + bar.get_height()/2.0, f"{w:.1f}% (MSE: {err:.4f})", va="center", fontsize=9, fontweight="bold")

    ax.set_title("Top Sensor Channels Contributing to Anomaly Reconstruction Error (Severe Degradation)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Attribution Contribution Percentage (%)")
    ax.set_xlim(0, max(contrib_pcts) * 1.3)
    save_plot(fig, "03_nasa_sensor_attribution_breakdown.png")

    # Figure 4: NASA LSTM Reconstruction Error Convergence & Profile
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(range(1, len(lstm_ae.training_history) + 1), lstm_ae.training_history, marker="o", color="#8e44ad", linewidth=2.0)
    ax1.set_title("LSTM Autoencoder Training Loss Convergence", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Mean Squared Reconstruction Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Actual vs Reconstructed sample curve for top sensor (first sensor in top_sensors)
    top_sensor_idx = feature_names.index(top_sensors[0]["sensor"]) if top_sensors[0]["sensor"] in feature_names else 0
    recon_sample = lstm_ae.reconstruct(X_val_seq[severe_mask][:1])  # (1, 30, D)
    actual_sample = X_val_seq[severe_mask][0]

    ax2.plot(range(1, 31), actual_sample[:, top_sensor_idx], label="Actual Observed Sensor Sequence", color="#c0392b", linewidth=2.0)
    ax2.plot(range(1, 31), recon_sample[0, :, top_sensor_idx], label="Autoencoder Reconstructed Nominal", color="#2980b9", linestyle="--", linewidth=2.0)
    ax2.fill_between(range(1, 31), actual_sample[:, top_sensor_idx], recon_sample[0, :, top_sensor_idx], color="#e74c3c", alpha=0.2, label="Reconstruction Residual Error")
    ax2.set_title(f"Reconstruction Overlay: {top_sensors[0]['sensor']} (Severe Wear Window)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Cycle in Window (1 to 30)")
    ax2.set_ylabel("Normalized Sensor Measurement")
    ax2.legend(loc="best", fontsize=9)
    save_plot(fig, "04_nasa_lstm_reconstruction_error.png")

    # Figure 5: AI4I Score Distributions (Normal vs Failure)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    colors = {"Normal": "#27ae60", "Failure": "#c0392b"}

    for idx, (name, detector) in enumerate(ai4i_detectors.items()):
        scores = ai4i_test_scores[name]
        ax = axes[idx]
        norm_scores = scores[y_test_ai4i == 0]
        fail_scores = scores[y_test_ai4i == 1]

        sns.kdeplot(norm_scores, ax=ax, label=f"Normal (N={len(norm_scores)})", color=colors["Normal"], fill=True, alpha=0.3)
        sns.kdeplot(fail_scores, ax=ax, label=f"Machine Failure (N={len(fail_scores)})", color=colors["Failure"], fill=True, alpha=0.4)
        ax.axvline(detector.thresholds["medium"], color="#d35400", linestyle="--", linewidth=1.5, label=f"Calibrated Th: {detector.thresholds['medium']:.3f}")

        ax.set_title(f"{name.replace('_', ' ').title()}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("Density")
        ax.legend(loc="upper right", fontsize=8)

    plt.suptitle("UCI AI4I 2020: Anomaly Score Distributions for Normal vs True Machine Failure", fontsize=13, fontweight="bold")
    save_plot(fig, "05_ai4i_score_distributions_comparison.png")

    # Figure 6: AI4I Precision-Recall and ROC Curves
    fig, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(14, 5.5))
    plot_palette = ["#2980b9", "#27ae60", "#e67e22", "#8e44ad"]

    for idx, (name, detector) in enumerate(ai4i_detectors.items()):
        scores = ai4i_test_scores[name]
        precision, recall, _ = precision_recall_curve(y_test_ai4i, scores)
        fpr, tpr, _ = roc_curve(y_test_ai4i, scores)
        pr_auc_val = auc(recall, precision)
        roc_auc_val = auc(fpr, tpr)
        col = plot_palette[idx % len(plot_palette)]

        ax_pr.plot(recall, precision, label=f"{name} (PR-AUC = {pr_auc_val:.3f})", color=col, linewidth=2.0)
        ax_roc.plot(fpr, tpr, label=f"{name} (ROC-AUC = {roc_auc_val:.3f})", color=col, linewidth=2.0)

    # Baseline chance levels
    prevalence = np.mean(y_test_ai4i)
    ax_pr.axhline(prevalence, color="gray", linestyle=":", label=f"Random Chance ({prevalence*100:.1f}%)")
    ax_pr.set_title("Post-Hoc Precision-Recall Benchmark on Machine Failure", fontsize=11, fontweight="bold")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_ylim(0, 1.02)
    ax_pr.legend(loc="upper right", fontsize=9)

    ax_roc.plot([0, 1], [0, 1], color="gray", linestyle=":", label="Random Chance (0.50)")
    ax_roc.set_title("Post-Hoc ROC Benchmark on Machine Failure", fontsize=11, fontweight="bold")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right", fontsize=9)
    save_plot(fig, "06_ai4i_pr_roc_curves.png")

    # Figure 7: AI4I Confusion Matrices
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for idx, (name, detector) in enumerate(ai4i_detectors.items()):
        cm_data = ai4i_results[name]["confusion_matrix"]
        cm_arr = np.array([[cm_data["tn"], cm_data["fp"]], [cm_data["fn"], cm_data["tp"]]])
        ax = axes[idx]
        sns.heatmap(cm_arr, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Failure"])
        ax.set_title(f"{name}\n(F1={ai4i_results[name]['f1_score']:.3f})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Unsupervised Detection")
        if idx == 0:
            ax.set_ylabel("Actual Failure Label")

    plt.suptitle("AI4I 2020: Post-Hoc Confusion Matrices at Calibrated Thresholds", fontsize=12, fontweight="bold", y=1.05)
    save_plot(fig, "07_ai4i_confusion_matrices.png")

    # Figure 8: Severity Tier Breakdown across Degradation Phases
    fig, (ax_nasa, ax_ai4i) = plt.subplots(1, 2, figsize=(14, 5))
    classifier_nasa = SeverityClassifier.from_thresholds(lstm_ae.thresholds)

    # Classify NASA validation into severity tiers
    nasa_sev_nom = classifier_nasa.classify_batch(tier_nominal)
    nasa_sev_mod = classifier_nasa.classify_batch(tier_moderate)
    nasa_sev_sev = classifier_nasa.classify_batch(tier_severe)

    def get_proportions(sev_list):
        total = len(sev_list)
        return [100.0 * sev_list.count(lvl) / total for lvl in SeverityClassifier.SEVERITY_LEVELS]

    prop_nom = get_proportions(nasa_sev_nom)
    prop_mod = get_proportions(nasa_sev_mod)
    prop_sev = get_proportions(nasa_sev_sev)

    stages = ["Nominal\n(RUL >= 110)", "Moderate Wear\n(30 <= RUL < 110)", "Severe Wear\n(RUL < 30)"]
    sev_colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]

    bottom = np.zeros(3)
    for s_idx, lvl in enumerate(SeverityClassifier.SEVERITY_LEVELS):
        vals = [prop_nom[s_idx], prop_mod[s_idx], prop_sev[s_idx]]
        ax_nasa.bar(stages, vals, bottom=bottom, label=lvl, color=sev_colors[s_idx], edgecolor="white", width=0.55)
        bottom += np.array(vals)

    ax_nasa.set_title("NASA C-MAPSS: Severity Progression by Operating Phase", fontsize=11, fontweight="bold")
    ax_nasa.set_ylabel("Proportion of Operating Sequences (%)")
    ax_nasa.set_ylim(0, 100)
    ax_nasa.legend(loc="upper left", fontsize=9)

    # AI4I severity proportions: Normal machine state vs Machine Failure state
    classifier_ai4i = SeverityClassifier.from_thresholds(dense_ae.thresholds)
    scores_dense_ai4i = ai4i_test_scores["dense_autoencoder"]
    ai4i_sev_norm = classifier_ai4i.classify_batch(scores_dense_ai4i[y_test_ai4i == 0])
    ai4i_sev_fail = classifier_ai4i.classify_batch(scores_dense_ai4i[y_test_ai4i == 1])

    prop_ai4i_norm = get_proportions(ai4i_sev_norm)
    prop_ai4i_fail = get_proportions(ai4i_sev_fail)

    stages_ai4i = ["Normal Machines\n(N=1932)", "Failure Machines\n(N=68)"]
    bottom_ai4i = np.zeros(2)
    for s_idx, lvl in enumerate(SeverityClassifier.SEVERITY_LEVELS):
        vals = [prop_ai4i_norm[s_idx], prop_ai4i_fail[s_idx]]
        ax_ai4i.bar(stages_ai4i, vals, bottom=bottom_ai4i, label=lvl, color=sev_colors[s_idx], edgecolor="white", width=0.45)
        bottom_ai4i += np.array(vals)

    ax_ai4i.set_title("UCI AI4I 2020: Severity Breakdown by Operating State", fontsize=11, fontweight="bold")
    ax_ai4i.set_ylabel("Proportion of Samples (%)")
    ax_ai4i.set_ylim(0, 100)
    ax_ai4i.legend(loc="upper right", fontsize=9)

    save_plot(fig, "08_severity_tier_breakdown.png")

    # =========================================================================
    # PART 4: SERIALIZE MODEL ARTIFACTS
    # =========================================================================
    print("\n" + "=" * 80)
    print("--- 4. Serializing Anomaly Detection Model Artifacts ---")
    models_dir = os.path.join(BASE_DIR, "models", "anomaly_detection")

    # NASA models
    lstm_ae.save(os.path.join(models_dir, "lstm_autoencoder", "lstm_autoencoder_FD001.pt"))
    iforest_nasa.save(os.path.join(models_dir, "isolation_forest", "isolation_forest_FD001.joblib"))
    ocsvm_nasa.save(os.path.join(models_dir, "one_class_svm", "one_class_svm_FD001.joblib"))
    lof_nasa.save(os.path.join(models_dir, "lof", "lof_FD001.joblib"))

    # AI4I models
    dense_ae.save(os.path.join(models_dir, "autoencoder", "dense_autoencoder_AI4I.pt"))
    iforest_ai4i.save(os.path.join(models_dir, "isolation_forest", "isolation_forest_AI4I.joblib"))
    ocsvm_ai4i.save(os.path.join(models_dir, "one_class_svm", "one_class_svm_AI4I.joblib"))
    lof_ai4i.save(os.path.join(models_dir, "lof", "lof_AI4I.joblib"))

    # Master Thresholds JSON
    thresholds_file = os.path.join(models_dir, "thresholds.json")
    with open(thresholds_file, "w") as f:
        json.dump(metrics_summary["thresholds"], f, indent=4)
    print(f"  Saved master thresholds to: {thresholds_file}")

    # Metadata JSON
    metadata = {
        "phase": 4,
        "title": "Multivariate Sensor Anomaly Detection",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models_trained": {
            "FD001": ["lstm_autoencoder", "isolation_forest", "one_class_svm", "lof"],
            "AI4I": ["dense_autoencoder", "isolation_forest", "one_class_svm", "lof"]
        },
        "feature_dimensions": {
            "FD001": len(feature_names),
            "AI4I": len(ai4i_feature_names)
        },
        "severity_tiers": SeverityClassifier.SEVERITY_LEVELS
    }
    with open(os.path.join(models_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    # Master Experiment Metrics JSON
    metrics_summary["execution_time_seconds"] = round(time.time() - t_start, 2)
    with open(os.path.join(models_dir, "experiment_metrics.json"), "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"  Saved experiment metrics to: {os.path.join(models_dir, 'experiment_metrics.json')}")

    print("\n" + "=" * 80)
    print(f"PHASE 4 EXECUTION COMPLETE IN {metrics_summary['execution_time_seconds']} SECONDS")
    print("=" * 80)


if __name__ == "__main__":
    run_phase4()
