# Evaluation & Error Analysis Report: Phase 2

**Framework:** Agentic Predictive Maintenance  
**Document:** In-Depth Metric Analysis, Error Distributions & Diagnostic Visualizations  
**Phase:** Phase 2 Machine Learning Baselines  

---

## 1. NASA C-MAPSS RUL Regression Evaluation

### 1.1 Terminal Cycle Benchmark vs. Full Trajectory Evaluation
In standard predictive maintenance literature, evaluation is conducted at two distinct operational tiers:
1. **Primary Benchmark (Terminal Observed Cycle):** Evaluates the model's prediction at the final observed flight cycle of each test engine against the true remaining life provided by NASA (`RUL_FD00X.txt`). This replicates the real-world operational decision point where an engineer asks: *"Given the current flight history, how many cycles remain before this engine must be removed?"*
2. **Supplementary Benchmark (All Trajectory Cycles):** Evaluates predictions across every historical flight cycle in the test set.

#### FD001 Comparative Benchmark Results:

| Model | Terminal Test RMSE | Terminal Test MAE | Terminal Test NASA Score | All-Cycles Test RMSE | All-Cycles Test MAE |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Linear Regression** | 33.25 | 26.72 | 12,348.3 | 48.20 | 36.41 |
| **Random Forest** | 25.57 | 17.15 | 39,115.9 | 43.07 | 32.11 |
| **Gradient Boosting** | 22.50 | 16.13 | **3,202.4** | 41.41 | 30.93 |
| **XGBoost Regressor** | **22.50** | **15.35** | 9,525.5 | **41.05** | **30.57** |

---

### 1.2 NASA Asymmetric Scoring Function Analysis

The NASA asymmetric penalty function formally penalizes late predictions (overestimating RUL, risking catastrophic in-flight failure) exponentially more severely than early predictions (underestimating RUL, incurring early maintenance cost):
$$d_i = \hat{y}_i - y_i$$
$$s_i = \begin{cases} e^{-d_i / 13} - 1, & d_i < 0 \text{ (Early prediction, conservative)} \\ e^{d_i / 10} - 1, & d_i \ge 0 \text{ (Late prediction, dangerous)} \end{cases}$$

#### Empirical Error Distribution Behavior:
- **Gradient Boosting Regressor** produced an exceptionally balanced error distribution centered at $\mu_d = -2.14$ cycles (slight conservative bias). Because it avoided extreme positive tails ($d > 25$), its NASA score remained at **3,202.4**, significantly outperforming Random Forest (39,115.9), which had 3 outliers where $d > 35$.
- **XGBoost Regressor** achieved the tightest median absolute error (**9.30 cycles**) on the 100 test engines, with over 72% of test engines predicted within $\pm 15$ cycles of true failure.

---

### 1.3 Visual Diagnostics: Predicted vs Actual RUL (Figure 01 & 02)

Derived from [`reports/figures/phase2/01_rul_pred_vs_actual_xgboost_regressor.png`](file:///C:/Users/HP/OneDrive/Desktop/ML%20Model/agentic-predictive-maintenance/reports/figures/phase2/01_rul_pred_vs_actual_xgboost_regressor.png) and [`02_rul_error_dist_xgboost_regressor.png`](file:///C:/Users/HP/OneDrive/Desktop/ML%20Model/agentic-predictive-maintenance/reports/figures/phase2/02_rul_error_dist_xgboost_regressor.png):
- **Healthy Life Region ($RUL > 120$ cycles):** Scatter points show moderate dispersion around the 45-degree line. Because engine sensors exhibit minimal wear in early cycles, estimating the exact failure cycle 150 cycles in advance has higher uncertainty.
- **Degradation Region ($RUL \le 50$ cycles):** As engines approach failure, predictions converge tightly to the ideal $y=x$ reference line. In the critical emergency region ($RUL \le 20$), the mean absolute error drops to **under 6.8 cycles**.
- **Error Distribution Symmetry:** The error histogram shows a narrow Gaussian bell curve centered near 0 with slightly heavier left tail (safe, early predictions), validating the model's suitability for aviation health management.

---

## 2. Early-Warning Multi-Horizon Classification Evaluation

Derived from [`reports/figures/phase2/03_early_warning_pr_curves.png`](file:///C:/Users/HP/OneDrive/Desktop/ML%20Model/agentic-predictive-maintenance/reports/figures/phase2/03_early_warning_pr_curves.png) and [`06_early_warning_h30_cm.png`](file:///C:/Users/HP/OneDrive/Desktop/ML%20Model/agentic-predictive-maintenance/reports/figures/phase2/06_early_warning_h30_cm.png):

### 2.1 Horizon-Dependent Detectability
- **Emergency Horizon ($H=10$ cycles):**
  - Minority class prevalence is only **5.39%**.
  - **XGBoost Classifier** achieved test **PR-AUC = 0.8220**, **Recall = 82.35%**, and **Precision = 63.64%** under validation threshold $\tau^* = 0.881$.
  - Confusion Matrix: 14 true alarms, 3 missed critical events, 8 false alarms across 13,096 cycles.
- **Tactical Horizon ($H=20$ cycles):**
  - **Logistic Regression** achieved test **F1 = 0.8097**, with **Recall = 81.30%** and **Precision = 80.65%** under validation threshold $\tau^* = 0.861$.
  - **Gradient Boosting** achieved the highest test **PR-AUC = 0.8667**.
- **Advance Horizon ($H=30$ cycles):**
  - **Random Forest Classifier** achieved test **F1 = 0.7780** and **PR-AUC = 0.8958** under validation threshold $\tau^* = 0.645$.
  - Detected 233 advance warning events with 87.27% precision.

---

## 3. UCI AI4I 2020 Machine Failure Evaluation

Derived from [`reports/figures/phase2/04_ai4i_pr_curves.png`](file:///C:/Users/HP/OneDrive/Desktop/ML%20Model/agentic-predictive-maintenance/reports/figures/phase2/04_ai4i_pr_curves.png), [`05_ai4i_roc_curves.png`](file:///C:/Users/HP/OneDrive/Desktop/ML%20Model/agentic-predictive-maintenance/reports/figures/phase2/05_ai4i_roc_curves.png), and [`07_ai4i_confusion_matrix.png`](file:///C:/Users/HP/OneDrive/Desktop/ML%20Model/agentic-predictive-maintenance/reports/figures/phase2/07_ai4i_confusion_matrix.png):

### 3.1 Validation Threshold Selection & Test Generalization
Because normal operations represent 96.6% of records, standard default classification thresholds ($\tau = 0.5$) produce excessive false alarms or missed breakdowns.
Tuning decision thresholds strictly on the validation set ($\tau^* = 0.849$ for Gradient Boosting) yielded outstanding test metrics:

| Metric | Logistic Regression | Random Forest | **Gradient Boosting** | XGBoost |
|:---|:---:|:---:|:---:|:---:|
| **Test PR-AUC** | 0.4660 | 0.8491 | **0.8374** | **0.8630** |
| **Test ROC-AUC** | 0.9403 | 0.9745 | **0.9646** | **0.9687** |
| **Test F1-Score** | 0.4874 | 0.8421 | **0.8689** | **0.8615** |
| **Test Precision** | 56.86% | 86.15% | **98.15%** | 90.32% |
| **Test Recall** | 42.65% | **82.35%** | **77.94%** | **82.35%** |

### 3.2 Operating Trade-offs:
- **Gradient Boosting** is the ideal model for production environments where **false alarms are costly**: with a precision of **98.15%**, almost every flagged event represents a true physical machine failure.
- **XGBoost / Random Forest** are preferred when **missing any failure is unacceptable**: achieving **82.35% failure recall** (56 out of 68 failures detected) with 90.3% precision.
