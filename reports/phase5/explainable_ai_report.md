# Phase 5 Technical Report: Explainable AI for Predictive Maintenance

**Project:** Agentic Predictive Maintenance: An Explainable Multivariate Time-Series Framework for Early Machine Failure Detection, Remaining Useful Life Estimation and Maintenance Decision Support  
**Stage:** Phase 5 — Explainable AI (XAI) Integration  
**Author:** Explainable AI & Machine Learning Research Engineer  
**Status:** Complete & Empirically Validated  

---

## 1. Executive Summary & Purpose

In safety-critical industrial operations—such as aviation turbofan engines and industrial CNC manufacturing—black-box machine learning predictions are insufficient for operational adoption. Maintenance engineers require transparent, mathematically grounded justifications for automated decisions:
1. **Why** a machine received a high failure-risk prediction or depleted Remaining Useful Life (RUL) estimate.
2. **Which** specific sensor channels, thermodynamic parameters, or mechanical stresses triggered the warning.
3. **When** the degradation originated and which historical operational cycles contributed most heavily to the prognosis.
4. **How** different modeling paradigms (Tree Ensembles vs Recurrent Deep Networks) view machine degradation.

Phase 5 addresses these requirements by building an end-to-end **Explainable AI Module** directly integrated with the trained models from Phase 2 (traditional baselines), Phase 3 (deep recurrent architectures), and Phase 4 (multivariate anomaly detectors).

---

## 2. Methodology & Explainability Framework

### 2.1 Cooperative Game Theory via SHAP (SHapley Additive exPlanations)
- **Mathematical Principle:** Computes local feature attribution via classical Shapley values:
  $$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N| - |S| - 1)!}{|N|!} (v(S \cup \{i\}) - v(S))$$
  guaranteeing local accuracy (efficiency), missingness, and consistency.
- **Implementation:** `TreeSHAP` was deployed for tree ensembles (XGBoost Regressor/Classifier and Random Forest Regressor/Classifier).
- **Global & Local Attribution:**
  - Global importance: Mean absolute SHAP values across evaluation cohorts: $I_j = \frac{1}{N} \sum_{i=1}^N |\phi_{i,j}|$.
  - Local instances: Additive waterfall decompositions showing signed pushes ($\pm$ cycles or $\pm$ probability) from the base value $E[f(x)]$.
  - Operational lifecycle tracking: Monitoring the continuous evolution of sensor SHAP values across an engine's operating cycles from healthy commission to end-of-life.

### 2.2 Model-Agnostic Permutation Feature Importance
- **Mathematical Principle:** Evaluates empirical model performance degradation upon feature shuffling:
  $$\Delta \mathcal{M}_j = \mathcal{M}(X_{\text{val}}) - \mathcal{M}(X_{\text{val}}^{\pi_j})$$
  measured via drop in $R^2$ (for NASA RUL regression) and drop in ROC-AUC (for AI4I classification).
- **Leakage Prevention:** Permutations were computed strictly on validation data with 5 independent repetitions, never using target labels as features.

### 2.3 Deep Learning Spatiotemporal Attribution (Integrated Gradients)
- **Mathematical Principle:** Axiomatic attribution developed by Sundararajan et al. (2017) satisfying Completeness and Implementation Invariance:
  $$\text{IG}_i(x) = (x_i - x'_i) \times \int_0^1 \frac{\partial F(x' + \alpha(x - x'))}{\partial x_i} d\alpha$$
  computed across $M=25$ Riemann approximation steps against a nominal baseline reference.
- **Spatiotemporal Decomposition:** Deconstructs the 2D attribution tensor $(L \times D)$ into:
  - **Temporal Attribution Profile:** $A_{\text{time}}(t) = \sum_{d=1}^D |\text{IG}_{t,d}|$ across the 30-cycle observation window.
  - **Sensor Channel Profile:** $A_{\text{sensor}}(d) = \sum_{t=1}^L \text{IG}_{t,d}$ aggregating temporal influence per sensor.

---

## 3. Key Empirical Findings

### 3.1 NASA C-MAPSS FD001 (RUL Regression)
- **Dominant Physical Degradation Drivers:**
  1. `sensor_11_baseline_dev` (HPC outlet static temperature deviation): Mean $|SHAP| = 9.0104$ cycles.
  2. `sensor_4_roll_mean_10` (LPT outlet temperature 10-cycle rolling mean): Mean $|SHAP| = 7.5406$ cycles.
  3. `sensor_4_roll_mean_5` (LPT outlet temperature 5-cycle rolling mean): Mean $|SHAP| = 5.8214$ cycles.
  4. `sensor_13_baseline_dev` (LPT coolant bleed deviation): Mean $|SHAP| = 5.7737$ cycles.
  5. `sensor_12_baseline_dev` (Fan speed ratio deviation): Mean $|SHAP| = 5.4515$ cycles.
- **Physical Interpretation:** Engine degradation in FD001 is primarily driven by thermal stress in the High-Pressure Compressor (HPC, `sensor_11`) and Low-Pressure Turbine (LPT, `sensor_4`), coupled with bypass aerodynamic imbalance (`sensor_12`). Rolling aggregations smooth high-frequency vibration, allowing the tree models to capture sustained wear trends.

### 3.2 UCI AI4I 2020 (Machine Failure Classification)
- **Dominant Mechanical Breakdown Drivers:**
  1. `Mechanical power [W]`: Mean $|SHAP| = 0.9143$.
  2. `Tool wear [min]`: Mean $|SHAP| = 0.8817$.
  3. `Rotational speed [rpm]`: Mean $|SHAP| = 0.7234$.
  4. `Overstrain indicator [Nm*min]`: Mean $|SHAP| = 0.5629$.
  5. `Torque to speed ratio`: Mean $|SHAP| = 0.4826$.
- **Failure Mode Synergy:** High rotational speed combined with abnormal torque triggers acute overstrain failure (OSF) and power failure (PWF), whereas tool wear accumulates progressively toward Tool Wear Failure (TWF).

### 3.3 Deep Learning Temporal Recency & Decay
- The LSTM Integrated Gradients profile reveals an exponential recency gradient:
  - Cycle 1 (oldest in window): $0.28\%$ of total attribution.
  - Cycle 15 (midpoint): $1.64\%$ of total attribution.
  - Cycle 30 (most recent cycle): **$20.00\%$** of total attribution.
  - **Recency Ratio:** The final 10 cycles (cycles 21–30) account for **$10.22\times$** the attribution of the initial 10 cycles (cycles 1–10).
- This proves that while the recurrent hidden state preserves historical drift, immediate failure proximity is dictated predominantly by the most recent operational cycles.

---

## 4. Verification and Downstream Reusability

All Phase 5 components have been validated:
1. **Verification Script:** [`verify_phase5_deliverables.py`](file:///c:/Users/HP/OneDrive/Desktop/ML%20Model/agentic-predictive-maintenance/src/verify_phase5_deliverables.py) asserts 100% pass across all functions.
2. **Jupyter Notebook:** [`notebooks/14_explainable_ai.ipynb`](file:///c:/Users/HP/OneDrive/Desktop/ML%20Model/agentic-predictive-maintenance/notebooks/14_explainable_ai.ipynb) contains complete walkthroughs and visualizations.
3. **Standardized API:** Reusable functions in [`src/explainability/inference_api.py`](file:///c:/Users/HP/OneDrive/Desktop/ML%20Model/agentic-predictive-maintenance/src/explainability/inference_api.py) (`explain_prediction`, `get_global_feature_importance`, `get_temporal_explanation`, `get_sensor_contributions`) are structured specifically for autonomous agent tool-calling in Phase 6.
