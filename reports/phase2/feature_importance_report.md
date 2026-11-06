# Feature Importance & Explainability Foundations: Phase 2

**Framework:** Agentic Predictive Maintenance  
**Document:** Gini/MDI Feature Importance & Domain Attribution  
**Phase:** Phase 2 Machine Learning Baselines  

---

## 1. Executive Summary

Phase 2 evaluated feature importances across gradient boosted trees and random forest ensembles for both the **NASA C-MAPSS** turbofan engine degradation models and the **UCI AI4I 2020** machine failure models. 

### Core Takeaways:
1. **NASA C-MAPSS Temporal Superiority:** Backward-looking rolling means and baseline deviations account for **over 85% of total feature importance** in RUL prediction, empirically demonstrating that temporal aggregation and reference-subtraction vastly outperform raw, instantaneous sensor snapshots.
2. **Dominant Degradation Sensors:** `sensor_4` (Low-Pressure Turbine Outlet Temperature $T_{50}$), `sensor_11` (HPC Static Pressure $P_{s30}$), and `sensor_15` (Bypass Ratio $BPR$) emerge as the primary thermodynamic degradation signals.
3. **Physics-Informed Engineering Validation (AI4I):** The domain-informed physical features engineered in Phase 1 (`Overstrain indicator [Nm*min]`, `Mechanical power [W]`, and `Heat dissipation index`) directly represent **over 70.7% of total feature importance** in detecting industrial machine breakdowns, confirming that physics-guided feature engineering provides superior predictive power compared to raw sensor values alone.

---

## 2. NASA C-MAPSS RUL Regression Feature Importance

Derived from the winning **XGBoost Regressor** on subset `FD001`:

| Rank | Feature Name | Physical/Temporal Meaning | Importance Score | Cumulative Importance |
|:---:|:---|:---|:---:|:---:|
| 1 | `sensor_4_roll_mean_5` | LPT outlet temp ($T_{50}$), 5-cycle backward rolling mean | **0.2611** (26.1%) | 26.1% |
| 2 | `sensor_4_roll_mean_10` | LPT outlet temp ($T_{50}$), 10-cycle backward rolling mean | **0.2036** (20.4%) | 46.5% |
| 3 | `sensor_11_baseline_dev` | HPC static pressure ($P_{s30}$), deviation from initial healthy baseline | **0.0512** (5.1%) | 51.6% |
| 4 | `sensor_15_roll_mean_10` | Bypass ratio ($BPR$), 10-cycle backward rolling mean | **0.0505** (5.1%) | 56.7% |
| 5 | `sensor_4_baseline_dev` | LPT outlet temp ($T_{50}$), deviation from initial healthy baseline | **0.0292** (2.9%) | 59.6% |
| 6 | `sensor_12_baseline_dev` | Ratio of fuel flow to $P_{s30}$ ($\phi$), baseline deviation | **0.0274** (2.7%) | 62.3% |
| 7 | `sensor_9_baseline_dev` | Physical core speed ($N_c$), baseline deviation | **0.0251** (2.5%) | 64.8% |
| 8 | `sensor_21_roll_mean_10` | LPT coolant bleed ($\text{W}_{32}$), 10-cycle rolling mean | **0.0191** (1.9%) | 66.7% |
| 9 | `sensor_11_roll_mean_5` | HPC static pressure ($P_{s30}$), 5-cycle rolling mean | **0.0135** (1.4%) | 68.1% |
| 10 | `sensor_13_baseline_dev` | Corrected fan speed ($N_{f\text{R}}$), baseline deviation | **0.0128** (1.3%) | 69.4% |

### Domain Interpretation:
- **LPT Outlet Temperature (`sensor_4`):** Represents the thermal exhaust signature. As high-pressure compressor efficiency degrades, more fuel must be injected to maintain demanded thrust, directly increasing downstream turbine temperature.
- **HPC Static Pressure (`sensor_11`):** Direct indicator of compressor stage health. Loss of compression ratio causes anomalous pressure buildup at the compressor exit.
- **Why Temporal Means Dominate:** Raw sensor measurements are contaminated with high-frequency aerodynamic noise. Backward rolling means filter noise while preserving monotonic wear trajectories, allowing tree splits to identify true degradation milestones.

---

## 3. UCI AI4I 2020 Machine Failure Feature Importance

Derived from the winning **Gradient Boosting Classifier**:

| Rank | Feature Name | Origin / Physical Formula | Importance Score | Failure Mode Target |
|:---:|:---|:---|:---:|:---|
| 1 | `Overstrain indicator [Nm*min]` | **Engineered:** $\tau \times \text{Tool wear}$ | **0.2670** (26.7%) | Overstrain Failure (OSF) |
| 2 | `Mechanical power [W]` | **Engineered:** $\tau \times (2\pi \cdot \text{RPM} / 60)$ | **0.2656** (26.6%) | Power Failure (PWF) |
| 3 | `Heat dissipation index` | **Engineered:** $\Delta T \times \text{RPM}$ | **0.1743** (17.4%) | Heat Dissipation Failure (HDF) |
| 4 | `Rotational speed [rpm]` | Raw measurement | **0.1213** (12.1%) | Mechanical work / thermal |
| 5 | `Type_L` | Categorical indicator (Low quality tier) | **0.0377** (3.8%) | Fleet quality susceptibility |
| 6 | `Temp ratio` | **Engineered:** $T_{\text{process}} / T_{\text{air}}$ | **0.0329** (3.3%) | Thermal gradient |
| 7 | `Temp difference [K]` | **Engineered:** $T_{\text{process}} - T_{\text{air}}$ | **0.0294** (2.9%) | Heat Dissipation Failure (HDF) |
| 8 | `Tool wear [min]` | Raw measurement | **0.0227** (2.3%) | Tool Wear Failure (TWF) |
| 9 | `Torque [Nm]` | Raw measurement | **0.0185** (1.8%) | Mechanical load |
| 10 | `Process temperature [K]` | Raw measurement | **0.0110** (1.1%) | Machine thermodynamics |

### Analytical Validation:
1. **Engineered Physical Features Account for 70.7% of Predictive Power:**
   - `Overstrain indicator` ($26.7\%$) isolates machines where high mechanical resistance combines with heavy tool wear, explaining 100% of OSF events.
   - `Mechanical power` ($26.6\%$) pinpoints out-of-bounds power surges ($>9000\text{ W}$) and motor stalls ($<3500\text{ W}$), explaining 100% of PWF events.
   - `Heat dissipation index` ($17.4\%$) captures thermal convection breakdown ($\Delta T < 8.6\text{ K}$ and $\text{RPM} < 1380$), explaining 86.2% of HDF events.
2. **Readiness for Explainable AI (Phase 3/4):**
   These importance hierarchies provide the exact foundation for TreeSHAP and LIME explanations, ensuring that decision support alerts present verifiable physical rationales to human maintenance operators.
