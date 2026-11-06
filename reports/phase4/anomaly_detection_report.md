# Phase 4 Technical Report: Multivariate Sensor Anomaly Detection

**Project:** Agentic Predictive Maintenance: An Explainable Multivariate Time-Series Framework for Early Machine Failure Detection, Remaining Useful Life Estimation and Maintenance Decision Support  
**Stage:** Phase 4 — Multivariate Sensor Anomaly Detection  
**Author:** Industrial Machine Learning & Anomaly Detection Specialist  
**Status:** Complete & Empirically Validated  

---

## 1. Executive Summary & Architectural Overview

Phase 4 introduces an unsupervised and semi-supervised **Multivariate Sensor Anomaly Detection Module** into the predictive maintenance framework. While Phase 2 (traditional ML) and Phase 3 (deep learning) focused on supervised Remaining Useful Life (RUL) regression and early-warning binary classification, Phase 4 addresses the critical industrial reality that catastrophic and subtle machine degradation often begins as abnormal sensor variations long before labeled failure data or supervised targets can be ascertained.

### Core Objectives Accomplished:
1. **NASA C-MAPSS (FD001):**
   - Investigated early-life operating behavior ($RUL \ge 110$) across training engines to establish a robust empirical nominal baseline ($N=5,239$ sequences).
   - Designed, trained, and validated a **Multivariate LSTM Autoencoder** in PyTorch to reconstruct 3D temporal sequence windows ($w=30, D=153$).
   - Trained three traditional unsupervised detectors: **Isolation Forest**, **One-Class SVM**, and **Local Outlier Factor (LOF)**.
   - Demonstrated continuous anomaly score escalation as engine degradation advances, yielding high negative correlation with Remaining Useful Life ($r = 0.7300$ for Isolation Forest, $r = 0.4865$ for LSTM AE).
   - Implemented sensor-level abnormality attribution, identifying the critical sensor channels (`sensor_7`, `sensor_12`, `sensor_11`, `sensor_4`) driving abnormal reconstruction errors.

2. **UCI AI4I 2020 Predictive Maintenance:**
   - Trained unsupervised anomaly detectors on 8,000 tabular training records strictly without utilizing `Machine failure` or failure mode labels during model fitting.
   - Conducted post-hoc benchmark evaluation against true failure events on the 2,000 test set instances.
   - Proved that unsupervised anomaly detection detects abnormal operating conditions with high ROC-AUC ($0.8824$ for Isolation Forest, $0.8230$ for One-Class SVM).
   - Explicitly delineated the operational difference between *unsupervised anomaly severity* and *supervised failure probability*.

3. **Multi-Tier Calibrated Severity Architecture:**
   - Established a quantile-based severity hierarchy: `Normal`, `Low anomaly`, `Medium anomaly`, and `High anomaly`.
   - Calibrated thresholds on validation nominal sequences (90.0th, 97.5th, and 99.0th percentiles).

4. **Unified Public API:**
   - Standardized the functional interface `predict_anomaly(input_data, model_id=..., dataset=...)` returning continuous anomaly scores, boolean flags, discrete severity tiers, model identifiers, and sensor attribution metadata.

---

## 2. Dataset Strategy & Experimental Integrity

### 2.1 NASA C-MAPSS Temporal Sequence Modeling
- **Input Data:** Sliding temporal windows of shape $(N, 30, 153)$ extracted with sequence length $w=30$.
- **Nominal Reference Investigation:** Rather than blindly assuming all early cycles are identical, we evaluated early-life cycles where $RUL \ge 110$ across all 80 training engines. This yielded $5,239$ training sequences and $1,511$ validation sequences. Feature statistics confirmed bounded, stable distributions (mean range: $[-0.7886, 0.9261]$, std range: $[0.0399, 0.4898]$).
- **Leakage Prevention:** Models were fitted exclusively on training engine data. Threshold calibration was conducted on validation engine nominal sequences. No future degradation cycles were utilized to compute past anomaly scores.

### 2.2 UCI AI4I 2020 Tabular Anomaly Modeling
- **Input Data:** 14 engineered operational condition features (`Air temperature`, `Process temperature`, `Rotational speed`, `Torque`, `Tool wear`, etc.).
- **Unsupervised Rule:** Training was performed purely on $X_{\text{train}}$ ($8,000$ rows) without target labels. Ground truth failure labels ($y_{\text{test}}$, 68 failure cases out of 2,000 test rows) were reserved strictly for post-hoc validation.

---

## 3. Algorithm Implementation Details

### 3.1 Multivariate LSTM Autoencoder (PyTorch)
- **Architecture:**
  - **Encoder:** $\text{Input}(B, 30, 153) \to \text{LSTM}(153 \to 64) \to \text{LSTM}(64 \to 32)$.
  - **Latent Bottleneck:** $z = h_{\text{last}} \in \mathbb{R}^{B \times 32}$.
  - **Sequence Expansion:** Latent vector $z$ is repeated across $L=30$ time steps $\to (B, 30, 32)$.
  - **Decoder:** $\text{LSTM}(32 \to 64) \to \text{Dense}(64 \to 153) \to \hat{X}(B, 30, 153)$.
- **Objective Function:** Mean Squared Reconstruction Error:
  $$\mathcal{L}(X, \hat{X}) = \frac{1}{B \cdot L \cdot D} \sum_{b=1}^B \sum_{t=1}^L \sum_{d=1}^D (x_{b,t,d} - \hat{x}_{b,t,d})^2$$
- **Training Convergence:** Trained for 20 epochs with Adam ($\text{lr}=10^{-3}$, batch size 128) on CPU, achieving final reconstruction loss of $0.028557$.

### 3.2 Dense Tabular Autoencoder (PyTorch)
- **Architecture:** $\text{Input}(14) \to \text{Dense}(32) \to \text{ReLU} \to \text{Dense}(8) \to \text{ReLU} \to \text{Dense}(32) \to \text{ReLU} \to \text{Dense}(14)$.
- **Convergence:** 25 epochs on AI4I training set, final reconstruction loss $0.002861$.

### 3.3 Traditional Anomaly Detectors
- **Isolation Forest:** 100 isolation trees, ensemble sub-sampling. Inverted decision function so higher score indicates greater anomaly.
- **One-Class SVM:** Radial Basis Function (RBF) kernel ($\nu=0.05 / 0.034$, $\gamma=\text{'scale'}$). Inverted signed hyperplane distance.
- **Local Outlier Factor (LOF):** $k=20$ nearest neighbors fitted with `novelty=True` for out-of-sample inference.

---

## 4. Empirical Evaluation & Degradation Tracking

### 4.1 NASA C-MAPSS Progression Analysis
Across the validation fleet ($3,711$ sequences), anomaly scores were partitioned into three operational regimes based on Remaining Useful Life:
1. **Nominal Operation ($RUL \ge 110$):** Mean score = $0.028887$
2. **Moderate Degradation ($30 \le RUL < 110$):** Mean score = $0.030439$
3. **Severe Degradation ($RUL < 30$):** Mean score = $0.075197$

> **Key Finding:** During the severe degradation phase preceding failure, the LSTM Autoencoder reconstruction score escalates by **$+160.3\%$** above the nominal baseline. This provides an effective early-warning anomaly indicator.

| Detector Model | Pearson $r$ with Degradation ($-RUL$) | Low Th (p90) | Med Th (p97.5) | High Th (p99) |
|---|---|---|---|---|
| **Isolation Forest** | **0.7300** | -0.0013 | 0.0176 | 0.0298 |
| **Local Outlier Factor** | 0.6201 | 1.5642 | 1.7166 | 1.7642 |
| **One-Class SVM** | 0.6105 | 0.3424 | 4.0233 | 6.0784 |
| **LSTM Autoencoder** | 0.4865 | 0.0370 | 0.0402 | 0.0415 |

### 4.2 Sensor-Level Abnormality Attribution
By isolating per-feature reconstruction error $\text{MSE}_d = \frac{1}{L} \sum_{t=1}^L (x_{t,d} - \hat{x}_{t,d})^2$ during severe wear ($RUL < 30$), the top 5 contributing sensor channels were identified:
1. **`sensor_7_baseline_dev`** (HPC outlet static pressure deviation): Reconstruction MSE = $0.2979$ ($2.6\%$ contribution)
2. **`sensor_12_baseline_dev`** (Fan speed ratio deviation): Reconstruction MSE = $0.2552$ ($2.2\%$ contribution)
3. **`sensor_11_baseline_dev`** (HPC outlet temperature deviation): Reconstruction MSE = $0.2419$ ($2.1\%$ contribution)
4. **`setting_2`** (Operational setting variation): Reconstruction MSE = $0.2352$ ($2.0\%$ contribution)
5. **`sensor_4_baseline_dev`** (LPT outlet temperature deviation): Reconstruction MSE = $0.2325$ ($2.0\%$ contribution)

This physical alignment matches the known thermodynamic degradation dynamics of turbofan high-pressure compressors and low-pressure turbines.

---

## 5. Artifact Verification & Traceability

All artifacts generated in Phase 4 have been verified on disk:
- **Models:** `models/anomaly_detection/{lstm_autoencoder, autoencoder, isolation_forest, one_class_svm, lof}/`
- **Thresholds & Metadata:** `models/anomaly_detection/thresholds.json`, `metadata.json`, `experiment_metrics.json`
- **Visualizations (8 Figures):** `reports/phase4/figures/` and `reports/figures/phase4/`
- **Jupyter Notebook:** `notebooks/13_multivariate_anomaly_detection.ipynb`
- **Inference Function:** `src/anomaly_detection/inference.py` (`predict_anomaly`)
