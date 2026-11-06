# Data Dictionary: Agentic Predictive Maintenance Framework

## 1. Overview and Scope

This data dictionary formally defines all raw, cleaned, and engineered features across both primary and secondary datasets used in the **Agentic Predictive Maintenance Framework (Phase 1)**:
- **Primary Dataset**: NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset (subsets FD001, FD002, FD003, FD004)
- **Secondary Dataset**: UCI AI4I 2020 Predictive Maintenance Dataset

All units, data types, physical interpretations, and target indicators are documented below without fabricated interpretations.

---

## 2. Dataset A: NASA C-MAPSS Turbofan Engine Degradation

### 2.1 Raw Data Schema (26 Columns)

Each record represents a snapshot of turbofan engine operation during a single operational flight cycle. The files are whitespace-delimited without native header rows.

| Column Index | Field Name | Data Type | Physical / System Role | Description |
|:---|:---|:---|:---|:---|
| 1 | `unit_id` | `int64` | Identifier | Engine identifier within the fleet (1-indexed). |
| 2 | `cycle` | `int64` | Temporal Index | Operational cycle number (monotonically increasing $1, 2, \dots$). |
| 3 | `setting_1` | `float64` | Operational Setting | Altitude and flight condition operating setting 1. |
| 4 | `setting_2` | `float64` | Operational Setting | Mach number / throttle operating setting 2. |
| 5 | `setting_3` | `float64` | Operational Setting | Sea level / ambient operating setting 3 (constant at 100.0 in FD001). |
| 6 | `sensor_1` | `float64` | Sensor Measurement | Fan inlet temperature ($T_2$) [$^\circ\text{R}$] (Constant in FD001). |
| 7 | `sensor_2` | `float64` | Sensor Measurement | LPC outlet temperature ($T_{24}$) [$^\circ\text{R}$]. Sensitive to degradation. |
| 8 | `sensor_3` | `float64` | Sensor Measurement | HPC outlet temperature ($T_{30}$) [$^\circ\text{R}$]. Sensitive to degradation. |
| 9 | `sensor_4` | `float64` | Sensor Measurement | LPT outlet temperature ($T_{50}$) [$^\circ\text{R}$]. Strong upward degradation trend. |
| 10 | `sensor_5` | `float64` | Sensor Measurement | Fan inlet pressure ($P_2$) [$\text{psia}$] (Constant in FD001). |
| 11 | `sensor_6` | `float64` | Sensor Measurement | Bypass-duct pressure ($P_{15}$) [$\text{psia}$]. |
| 12 | `sensor_7` | `float64` | Sensor Measurement | HPC outlet pressure ($P_{30}$) [$\text{psia}$]. Decreases with HPC degradation. |
| 13 | `sensor_8` | `float64` | Sensor Measurement | Physical fan speed ($N_f$) [$\text{rpm}$]. |
| 14 | `sensor_9` | `float64` | Sensor Measurement | Physical core speed ($N_c$) [$\text{rpm}$]. Increases as HPC degrades. |
| 15 | `sensor_10` | `float64` | Sensor Measurement | Engine pressure ratio ($P_{r}$) (Constant in FD001). |
| 16 | `sensor_11` | `float64` | Sensor Measurement | HPC static pressure ($P_{s30}$) [$\text{psia}$]. Strong upward degradation trend. |
| 17 | `sensor_12` | `float64` | Sensor Measurement | Ratio of fuel flow to $P_{s30}$ ($\phi$) [$\text{pps/psi}$]. Strong degradation trend. |
| 18 | `sensor_13` | `float64` | Sensor Measurement | Corrected fan speed ($N_{f\text{R}}$) [$\text{rpm}$]. |
| 19 | `sensor_14` | `float64` | Sensor Measurement | Corrected core speed ($N_{c\text{R}}$) [$\text{rpm}$]. Upward drift with wear. |
| 20 | `sensor_15` | `float64` | Sensor Measurement | Bypass ratio ($BPR$). Increases with degradation. |
| 21 | `sensor_16` | `float64` | Sensor Measurement | Burner fuel-air ratio ($FAR$) (Constant in FD001). |
| 22 | `sensor_17` | `float64` | Sensor Measurement | Bleed enthalpy ($ht$). Sensitive to degradation. |
| 23 | `sensor_18` | `float64` | Sensor Measurement | Demanded fan speed (Constant in FD001). |
| 24 | `sensor_19` | `float64` | Sensor Measurement | Demanded corrected fan speed (Constant in FD001). |
| 25 | `sensor_20` | `float64` | Sensor Measurement | HPT coolant bleed ($\text{W}_{31}$) [$\text{lbm/s}$]. Decreases with wear. |
| 26 | `sensor_21` | `float64` | Sensor Measurement | LPT coolant bleed ($\text{W}_{32}$) [$\text{lbm/s}$]. Decreases with wear. |

### 2.2 Constant Sensor Audit (FD001)

Sensors showing empirical variance $\sigma = 0$ across all 20,631 training records are pruned during data cleaning to avoid numerical singularity and uninformative inputs:
- `setting_3`: Identical value (100.0) across all flights.
- `sensor_1`: Constant 518.67 $^\circ\text{R}$.
- `sensor_5`: Constant 14.62 psia.
- `sensor_10`: Constant 1.30.
- `sensor_16`: Constant 0.03.
- `sensor_18`: Constant 2388 rpm.
- `sensor_19`: Constant 100.00.

### 2.3 Generated Target Variables

| Target Name | Type | Range | Mathematical Definition | Purpose |
|:---|:---|:---|:---|:---|
| `RUL` | Continuous (`int64`) | $[0, 361]$ (train), $[7, 340]$ (test) | $RUL(i, t) = T_i^{\max} - t$ | Ground-truth Remaining Useful Life for regression models. |
| `early_warning_10` | Binary (`int32`) | $\{0, 1\}$ | $\mathbb{I}(RUL \le 10)$ | Critical immediate failure warning horizon (10 cycles). |
| `early_warning_20` | Binary (`int32`) | $\{0, 1\}$ | $\mathbb{I}(RUL \le 20)$ | Medium-term maintenance alert horizon (20 cycles). |
| `early_warning_30` | Binary (`int32`) | $\{0, 1\}$ | $\mathbb{I}(RUL \le 30)$ | Strategic advance planning horizon (30 cycles). |

### 2.4 Engineered Temporal Features

All temporal features are strictly backward-looking and computed independently within each engine trajectory (`unit_id`):

| Feature Pattern | Formula | Description |
|:---|:---|:---|
| `{sensor}_baseline_dev` | $x_t - \frac{1}{5}\sum_{k=1}^5 x_k$ | Deviation from the engine's initial healthy baseline (first 5 cycles). |
| `{sensor}_diff1` | $x_t - x_{t-1}$ | Backward lag-1 finite difference (instantaneous drift rate). |
| `{sensor}_roll_mean_{W}` | $\frac{1}{W}\sum_{k=0}^{W-1} x_{t-k}$ | Backward-looking rolling average over window $W \in \{5, 10, 20\}$ cycles. |
| `{sensor}_roll_std_{W}` | $\sqrt{\frac{1}{W-1}\sum (x_{t-k} - \mu)^2}$ | Backward-looking rolling volatility/noise over window $W \in \{5, 10, 20\}$ cycles. |

---

## 3. Dataset B: UCI AI4I 2020 Predictive Maintenance

### 3.1 Raw Feature Schema (14 Columns)

| Column Name | Type | Unit | Physical Role | Status in ML Modeling |
|:---|:---|:---|:---|:---|
| `UDI` | `int64` | None | Unique Row Identifier | **Excluded** (Arbitrary index). |
| `Product ID` | `string` | None | Variant ID (L/M/H + serial) | **Excluded** (Redundant with `Type`). |
| `Type` | `string` | Categorical | Product Quality Tier (L=50%, M=30%, H=20%) | **Model Input** (One-hot encoded). |
| `Air temperature [K]` | `float64` | Kelvin | Ambient Room Temperature | **Model Input** (Standardized). |
| `Process temperature [K]`| `float64` | Kelvin | Milling/Turning Process Temperature | **Model Input** (Standardized). |
| `Rotational speed [rpm]` | `float64` | RPM | Spindle Speed | **Model Input** (Standardized). |
| `Torque [Nm]` | `float64` | Newton-meters | Spindle Resistance Torque | **Model Input** (Standardized). |
| `Tool wear [min]` | `float64` | Minutes | Cumulative Tool Cutting Time | **Model Input** (Standardized). |
| `Machine failure` | `int64` | $\{0, 1\}$ | Primary Target Variable (0=Normal, 1=Fail) | **Target Variable** (3.39% positive). |
| `TWF` | `int64` | $\{0, 1\}$ | Tool Wear Failure mode | **Target Leakage** (Excluded from inputs). |
| `HDF` | `int64` | $\{0, 1\}$ | Heat Dissipation Failure mode | **Target Leakage** (Excluded from inputs). |
| `PWF` | `int64` | $\{0, 1\}$ | Power Failure mode | **Target Leakage** (Excluded from inputs). |
| `OSF` | `int64` | $\{0, 1\}$ | Overstrain Failure mode | **Target Leakage** (Excluded from inputs). |
| `RNF` | `int64` | $\{0, 1\}$ | Random Failure mode | **Target Leakage** (Excluded from inputs). |

> [!CAUTION]
> The failure mode flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) directly encode failure etiology and are downstream consequences of system breakdown. Including them in predictive input features constitutes severe target leakage. They are retained strictly for root-cause and explainability benchmarking.

### 3.2 Domain-Informed Physical Engineered Features

Based on machine tool mechanics and thermodynamics (Matzka, 2020):

| Feature Name | Mathematical Definition | Physical Justification |
|:---|:---|:---|
| `Mechanical power [W]` | $P = \tau \times \omega = \text{Torque} \times \left(\frac{2\pi \times \text{RPM}}{60}\right)$ | Captures electrical/mechanical work rate. Power failure occurs when $P < 3500\text{W}$ or $P > 9000\text{W}$. |
| `Temp difference [K]` | $\Delta T = T_{\text{process}} - T_{\text{air}}$ | Heat dissipation capacity. Heat dissipation failure occurs when $\Delta T < 8.6\text{K}$ and $\text{RPM} < 1380$. |
| `Temp ratio` | $T_{\text{process}} / T_{\text{air}}$ | Normalized thermal gradient. |
| `Torque to speed ratio` | $\tau / \text{RPM}$ | High mechanical impedance indicator. |
| `Overstrain indicator [Nm*min]` | $\tau \times \text{Tool wear}$ | Overstrain failure product. OSF occurs when $\tau \times \text{wear} > 11000\text{ Nm}\cdot\text{min}$ (Type L). |
| `Heat dissipation index` | $\Delta T \times \text{RPM}$ | Dynamic thermal convection proxy. |
| `Power wear index` | $P \times \text{Tool wear}$ | Cumulative thermodynamic tool stress. |
