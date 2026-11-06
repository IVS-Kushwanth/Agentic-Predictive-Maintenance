# Project Evidence: Visualizations, Diagnostic Charts & UI Views

This catalog indexes all 35+ high-resolution figures generated across the project phases. All image files are located in `reports/figures/` and referenced below with exact paths.

---

## 1. Phase 1 & Exploratory Data Analysis Figures (`reports/figures/nasa/` & `ai4i/`)

| Figure Name | File Path | Physical / Diagnostic Description |
| :--- | :--- | :--- |
| **NASA Operating Cycles** | `reports/figures/nasa/01_operating_cycles_distri.png` | Histogram showing run-to-failure engine lifespans (128 to 362 cycles). |
| **Sensors Over Cycles** | `reports/figures/nasa/03_sensors_over_cycles.png` | Monotonic thermal degradation trajectories for Sensor 11 (T30) and Sensor 4 (T50). |
| **Sensor Correlation Heatmap**| `reports/figures/nasa/05_sensor_correlation_heat.png` | Correlation matrix revealing strong collinearity among turbofan thermodynamic transducers. |
| **AI4I Class Imbalance** | `reports/figures/ai4i/01_target_class_balance.png` | Severe class distribution (96.6% normal vs. 3.4% failure). |
| **AI4I Failure Modes** | `reports/figures/ai4i/02_failure_modes_breakdown.png` | Breakdown across TWF, HDF, PWF, OSF, and RNF failure mechanisms. |
| **Physical Mechanics Power** | `reports/figures/ai4i/05_physical_mechanics_powe.png` | Interaction clusters between mechanical power ($\tau \times \omega$) and tool wear. |

---

## 2. Phase 2 Machine Learning Diagnostic Figures (`reports/figures/phase2/`)

| Figure Name | File Path | Diagnostic Significance |
| :--- | :--- | :--- |
| **RUL Pred vs Actual** | `reports/figures/phase2/01_rul_pred_vs_actual_xgboost_regressor.png` | Convergence to ideal line as engines approach failure ($RUL < 50$ cycles). |
| **RUL Error Distribution** | `reports/figures/phase2/02_rul_error_dist_xgboost_regressor.png` | Gaussian residual distribution with slight conservative negative bias. |
| **Early Warning PR Curves** | `reports/figures/phase2/03_early_warning_pr_curves.png` | Multi-horizon PR-curves achieving PR-AUC up to 0.8958 on horizon $H=30$. |
| **AI4I PR & ROC Curves** | `reports/figures/phase2/04_ai4i_pr_curves.png` | Steep lift over 3.4% baseline confirming high detection precision. |
| **AI4I Confusion Matrix** | `reports/figures/phase2/07_ai4i_confusion_matrix.png` | Gradient Boosting matrix: 1,931 TN, 1 FP, 15 FN, 53 TP (98.15% Precision). |
| **NASA Feature Importance** | `reports/figures/phase2/08_feature_importance_nasa.png` | Feature rankings confirming Sensor 11 and Sensor 4 as top degradation drivers. |

---

## 3. Phase 3 Deep Learning Figures (`reports/phase3/figures/`)

| Figure Name | File Path | Diagnostic Significance |
| :--- | :--- | :--- |
| **LSTM Huber Loss Curves** | `reports/phase3/figures/01_loss_curves_rul_lstm_w30.png` | Huber loss curves demonstrating smooth convergence over 45 epochs without overfitting. |
| **Deep Learning RUL Pred** | `reports/phase3/figures/02_rul_pred_vs_actual_deep_learning.png` | Deep Bi-LSTM ($L=60$) predictions tightly tracking ground truth. |
| **Trajectory Tracking** | `reports/phase3/figures/04_rul_trajectory_tracking.png` | Smooth continuous degradation tracking for sample test engines (#12, #24, #35). |
| **Window Length Comparison**| `reports/phase3/figures/05_window_length_comparison.png` | Demonstrating consistent RMSE and NASA score reductions as $L$ increases to 60. |
| **ML vs DL Comparison Bar** | `reports/phase3/figures/09_ml_vs_dl_comparison_bar.png` | Visualizing the 90.7% NASA score penalty reduction achieved by Bi-LSTM. |

---

## 4. Phase 4 Anomaly Detection Figures (`reports/phase4/figures/`)

| Figure Name | File Path | Diagnostic Significance |
| :--- | :--- | :--- |
| **Score Distribution** | `reports/phase4/figures/01_nasa_anomaly_score_distribution.png` | Contrast between nominal early cycles and end-of-life degraded states. |
| **Engine Trajectories** | `reports/phase4/figures/02_nasa_engine_trajectories.png` | Continuous escalation of anomaly scores as flight cycles progress. |
| **LSTM Recon Error** | `reports/phase4/figures/04_nasa_lstm_reconstruction_error.png` | Reconstruction error crossing nominal threshold $\tau = 0.0370$. |
| **AI4I Anomaly PR/ROC** | `reports/figures/phase4/06_ai4i_pr_roc_curves.png` | Unsupervised ROC-AUC = 0.8824 (9.1x random chance lift). |

---

## 5. Phase 5 Explainable AI Figures (`reports/phase5/figures/`)

| Figure Name | File Path | Diagnostic Significance |
| :--- | :--- | :--- |
| **SHAP Summary NASA** | `reports/phase5/figures/01_shap_summary_nasa_rul.png` | Beeswarm summary showing Sensor 11 driving RUL reductions. |
| **SHAP Waterfall Instance** | `reports/phase5/figures/02_shap_waterfall_nasa_instance.png` | Local waterfall explaining prediction for an individual engine. |
| **Temporal Saliency Map** | `reports/phase5/figures/05_deep_learning_temporal_saliency.png` | 2D gradient heatmap across 60 time steps. |
| **Lifecycle Progression** | `reports/phase5/figures/07_lifecycle_shap_progression.png` | Evolution of Shapley values across the operational lifecycle. |
