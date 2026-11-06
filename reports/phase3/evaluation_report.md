# Phase 3 Evaluation Report: Deep Learning Benchmarks & Asymmetric Scoring

**Project:** Agentic Predictive Maintenance: An Explainable Multivariate Time-Series Framework for Early Machine Failure Detection, Remaining Useful Life Estimation and Maintenance Decision Support  
**Stage:** Phase 3 — Deep Learning Model Evaluation & Degradation Tracking  
**Author:** Antigravity Deep Learning Engineering & Machine Learning Research Team  

---

## 1. Evaluation Protocol and Ground Truth Verification

### 1.1 Strict Zero-Leakage Guarantee
All Phase 3 deep learning models were trained and evaluated following rigorous scientific standards:
1. **Engine-Disjoint Splitting:** Scalers, features, and model parameters were fitted purely on training engines (80 engines in FD001). Validation engines (20 engines) served strictly for learning rate reduction and early stopping.
2. **Official NASA Test Fleet Terminal Benchmark ($N=100$ Engines):** For each test engine, the model processes its historical operational sequence up to the final observed cycle $C_{\text{end}}$. Exactly one prediction $\hat{y}_{\text{terminal}}$ is generated per engine and compared directly against the official NASA ground-truth RUL vector ($RUL\_FD001.txt$).
3. **Supplementary All-Window Evaluation:** A secondary analysis computes metrics across every sliding window in the test set ($10,196$ windows for $w=30$), validating continuous model stability.

### 1.2 NASA Asymmetric Scoring Function
In industrial predictive maintenance, **late predictions (overestimating RUL)** are significantly more hazardous than **early predictions (underestimating RUL)** because overestimation risks running the engine past catastrophic failure. The official PHM 2008 asymmetric penalty function reflects this domain reality:

$$d_i = \hat{y}_i - y_i$$

$$s_i = \begin{cases} \exp\left(-\frac{d_i}{13}\right) - 1 & \text{if } d_i < 0 \quad (\text{Early Prediction / Conservative}) \\ \exp\left(\frac{d_i}{10}\right) - 1 & \text{if } d_i \ge 0 \quad (\text{Late Prediction / Hazardous}) \end{cases}$$

$$\text{NASA Score} = \sum_{i=1}^{N} s_i$$

---

## 2. NASA C-MAPSS FD001 RUL Benchmark Results

The table below summarizes performance on the primary **FD001 terminal test fleet benchmark ($N=100$ engines)**:

| Model Architecture | Sequence Window ($L$) | Validation RMSE | Test Terminal RMSE | Test Terminal MAE | Test Terminal NASA Score | Test All Windows RMSE | Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LSTM Regressor** | **30** | 33.61 | **21.98** | **16.15** | **1,464.35** | 37.40 | 25,025 |
| **GRU Regressor** | **30** | 34.97 | **22.21** | **16.43** | **1,769.12** | 38.31 | 19,041 |
| **CNN-LSTM Regressor** | **30** | 32.02 | 26.99 | 19.98 | 4,301.67 | 38.88 | 27,425 |
| **LSTM Regressor** | **50** | 32.92 | **21.27** | **16.10** | **1,021.57** | 35.85 | 25,025 |
| **LSTM Regressor** | **60** | 32.55 | **20.02** | **14.83** | **889.38** | 33.69 | 25,025 |

### Key Observations:
1. **Dramatic Penalty Reduction:** While traditional Phase 2 ML baselines achieved competitive RMSE (~22.50 cycles), their symmetric loss functions produced occasional late predictions, inflating NASA scores (XGBoost scored $9,525.5$; Random Forest scored $22,754.7$). In contrast, **LSTM ($L=30$) achieved a score of $1,464.35$** and **GRU achieved $1,769.12$** — an **$84.6\%$ penalty reduction**, demonstrating superior conservatism on deteriorating engines.
2. **Temporal Window Impact:** Expanding the sequence window from $L=30$ to $L=60$ cycles steadily drove Test Terminal RMSE down from $21.98$ to **$20.02$ cycles** and the NASA Score from $1,464.35$ down to **$889.38$**. Longer temporal baselines allow recurrent gating units to filter transient sensor noise and observe the full onset of wear.
3. **GRU Efficiency:** With $23.9\%$ fewer parameters than LSTM, GRU attained nearly identical test accuracy ($22.21$ vs $21.98$ RMSE) and superior training convergence stability.

---

## 3. Extension to Subsets FD002, FD003, and FD004

| Subset | Regimes | Fault Modes | Best Model | Test Terminal RMSE | Test Terminal MAE | NASA Score |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FD001** | 1 | HPC Degradation | LSTM ($w=60$) | **20.02** | **14.83** | **889.38** |
| **FD002** | 6 | HPC Degradation | LSTM ($w=30$) | **29.59** | **21.11** | **17,311.44** |
| **FD003** | 1 | HPC + Fan Degradation | LSTM ($w=30$) | **20.86** | **15.43** | **1,655.53** |
| **FD004** | 6 | HPC + Fan Degradation | LSTM ($w=30$) | **36.26** | **27.62** | **128,754.75** |

### Insights:
- **Operating Conditions Complexity:** FD002 and FD004 contain 6 alternating flight conditions (altitude, Mach number, throttle resolver angle). Operating point shifts introduce large step-changes in sensor measurements, increasing test RMSE to $29.59$ (FD002) and $36.26$ (FD004).
- **Dual Fault Modes:** FD003 features both High-Pressure Compressor (HPC) and Fan degradation under single operating conditions. Despite the dual failure mechanism, LSTM achieved an outstanding **$20.86$ cycles RMSE** and a **$1,655.53$ NASA score**, confirming that temporal sequence models adapt seamlessly to multiple physical failure modes when operating conditions remain stationary.

---

## 4. Temporal Early-Warning Failure Classification

Early warning classifiers predict the probability that an engine will reach end-of-life within a specific horizon $H \in \{10, 20, 30\}$ cycles. Optimal decision thresholds were tuned on validation predictions to maximize $F_1$-score.

| Horizon ($H$) | Backbone Architecture | Optimal Threshold | Precision | Recall | $F_1$-Score | PR-AUC | ROC-AUC |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$H=10$ Cycles** | LSTM ($h=32$) | 0.9682 | 72.22% | 76.47% | **0.7429** | **0.8214** | **0.9996** |
| **$H=10$ Cycles** | GRU ($h=32$) | 0.9245 | 72.22% | 76.47% | **0.7429** | **0.8551** | **0.9996** |
| **$H=20$ Cycles** | LSTM ($h=32$) | 0.9593 | 81.00% | 65.85% | **0.7265** | **0.8564** | **0.9976** |
| **$H=20$ Cycles** | GRU ($h=32$) | 0.9362 | 83.33% | 73.17% | **0.7792** | **0.8320** | **0.9975** |
| **$H=30$ Cycles** | LSTM ($h=32$) | 0.8638 | 90.20% | 66.57% | **0.7660** | **0.9025** | **0.9958** |
| **$H=30$ Cycles** | GRU ($h=32$) | 0.8702 | 87.82% | 71.69% | **0.7894** | **0.9079** | **0.9962** |

### Key Diagnostic Findings:
- As the horizon expands from $10$ to $30$ cycles, the positive failure class proportion increases, enabling both LSTM and GRU classifiers to reach **PR-AUC exceeding $0.90$** and **ROC-AUC exceeding $0.995$**.
- At $H=30$ cycles, GRU achieved an $F_1$-score of **$0.7894$** ($87.82\%$ precision, $71.69\%$ recall), providing a robust multi-week scheduling window for maintenance depots.

---

## 5. UCI AI4I 2020: Deep Tabular MLP Evaluation

| Model | Paradigm | Optimal Thresh | Precision | Recall | $F_1$-Score | PR-AUC | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Deep Tabular MLP** | Neural Net | 0.8891 | 58.97% | 67.65% | 0.6301 | 0.6437 | 0.9658 |
| **Gradient Boosting (P2)** | Tree Ensemble | 0.4578 | 98.15% | 77.94% | **0.8689** | **0.8374** | **0.9781** |
| **XGBoost (P2)** | Tree Ensemble | 0.5210 | 97.87% | 67.65% | 0.8000 | 0.8245 | 0.9632 |

### Research Insight:
On small-to-medium tabular datasets with extreme imbalance ($3.4\%$ positive failure rate) and heterogeneous feature semantics, tree-based ensembles (Gradient Boosting, XGBoost) consistently outperform feed-forward neural networks due to greedy axis-aligned partitioning and absence of inductive bias requirements.

---

## 6. Generated Visualizations

The following publication-grade figures were generated and verified at 300 DPI:
1. `01_loss_curves_rul_lstm_gru_cnnlstm.png`: Training and validation loss trajectories across epochs.
2. `02_rul_pred_vs_actual_deep_learning.png`: Predicted vs actual RUL on official test fleet ($N=100$).
3. `03_rul_error_dist_deep_learning.png`: Test terminal residual error distributions.
4. `04_rul_trajectory_tracking.png`: Continuous multi-cycle degradation tracking for Engines #31, #49, and #81.
5. `05_window_length_comparison.png`: RMSE, MAE, and NASA Score scaling across window lengths $L \in \{30, 50, 60\}$.
6. `06_early_warning_dl_pr_curves.png`: Precision-Recall curves across early warning horizons ($10, 20, 30$ cycles).
7. `07_early_warning_dl_cm.png`: Confusion matrix at $H=30$ horizon.
8. `08_ai4i_mlp_curves_cm.png`: Precision-Recall, ROC, and confusion matrix for Deep Tabular MLP.
9. `09_ml_vs_dl_comparison_bar.png`: Direct cross-paradigm comparison of Traditional ML vs Deep Learning.
