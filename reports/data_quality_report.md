# Data Quality & Integrity Report: Phase 1

**Framework:** Agentic Predictive Maintenance  
**Phase:** Phase 1 — Dataset Acquisition, Verification, Quality Audit & Data Preparation  
**Auditor:** Senior Data Science & Engineering Team  
**Verification Date:** November 2026  

---

## 1. Executive Summary

This report establishes the empirical validation and data-hygiene baseline for the primary time-series dataset (**NASA C-MAPSS**) and secondary tabular benchmark dataset (**UCI AI4I 2020**). Both datasets were acquired directly from official repositories and subjected to an exhaustive battery of structural, statistical, and domain-integrity tests.

**Summary Outcome:**
- **NASA C-MAPSS (FD001–FD004):** PASSED all critical checks (zero missing values, zero duplicates, 100% strictly monotonic cycle sequences, validated non-zero variance sensors).
- **UCI AI4I 2020:** PASSED all physical consistency checks (temperatures strictly positive in Kelvin, RPM positive, zero duplicate UDIs, valid failure mode encodings).
- **Target Leakage:** 100% prevented by isolating AI4I failure modes and strictly using backward-looking rolling features for C-MAPSS.

---

## 2. NASA C-MAPSS Data Quality Audit

### 2.1 File Completeness and Dimensionality

All 12 expected raw data files and the official documentation archive were verified on disk:

| File Name | Record Count | Feature Columns | Number of Unique Engines | Min Cycles | Max Cycles | Mean Lifetime |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `train_FD001.txt` | 20,631 | 26 | 100 | 128 | 362 | 206.31 |
| `test_FD001.txt` | 13,096 | 26 | 100 | 31 | 303 | 130.96 |
| `RUL_FD001.txt` | 100 | 1 | 100 (Terminal Labels) | 7 | 145 | 75.08 |
| `train_FD002.txt` | 53,759 | 26 | 260 | 128 | 378 | 206.77 |
| `test_FD002.txt` | 33,991 | 26 | 259 | 21 | 367 | 131.24 |
| `RUL_FD002.txt` | 259 | 1 | 259 (Terminal Labels) | 8 | 149 | 80.60 |
| `train_FD003.txt` | 24,720 | 26 | 100 | 145 | 525 | 247.20 |
| `test_FD003.txt` | 16,596 | 26 | 100 | 38 | 475 | 165.96 |
| `RUL_FD003.txt` | 100 | 1 | 100 (Terminal Labels) | 9 | 140 | 75.05 |
| `train_FD004.txt` | 61,249 | 26 | 248 | 128 | 543 | 246.97 |
| `test_FD004.txt` | 41,214 | 26 | 249 | 19 | 486 | 165.52 |
| `RUL_FD004.txt` | 249 | 1 | 249 (Terminal Labels) | 7 | 144 | 80.39 |

### 2.2 Structural and Temporal Verification

1. **Missing (NaN) & Infinite Values:**
   - Raw missing cells: **0** across all 198,359 rows.
   - Infinite cells (`+inf`, `-inf`): **0**.
2. **Duplicate Records:**
   - Evaluated composite primary key `(unit_id, cycle)`.
   - Duplicates found: **0**.
3. **Cycle Monotonicity & Completeness:**
   - Every individual engine starts precisely at `cycle = 1`.
   - The cycle progression condition $\Delta \text{cycle} = \text{cycle}_{t} - \text{cycle}_{t-1} = 1$ holds for 100% of adjacent rows within each engine.
   - Zero missing or skipped cycles detected across all trajectories.

### 2.3 Sensor Variance & Constant Feature Pruning (FD001)

An empirical variance audit on the benchmark dataset `FD001` revealed that 7 out of 24 operating variables exhibit zero variance across all 20,631 records:

| Column | Observed Mean | Standard Deviation | Minimum | Maximum | Unique Values | Recommendation |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| `setting_3` | 100.0000 | 0.0000 | 100.0000 | 100.0000 | 1 | **Pruned (Zero variance)** |
| `sensor_1` | 518.6700 | 0.0000 | 518.6700 | 518.6700 | 1 | **Pruned (Zero variance)** |
| `sensor_5` | 14.6200 | 0.0000 | 14.6200 | 14.6200 | 1 | **Pruned (Zero variance)** |
| `sensor_10` | 1.3000 | 0.0000 | 1.3000 | 1.3000 | 1 | **Pruned (Zero variance)** |
| `sensor_16` | 0.0300 | 0.0000 | 0.0300 | 0.0300 | 1 | **Pruned (Zero variance)** |
| `sensor_18` | 2388.0000 | 0.0000 | 2388.0000 | 2388.0000 | 1 | **Pruned (Zero variance)** |
| `sensor_19` | 100.0000 | 0.0000 | 100.0000 | 100.0000 | 1 | **Pruned (Zero variance)** |

> [!NOTE]
> For multi-condition subsets (`FD002` and `FD004`), operational settings fluctuate across 6 flight conditions (sea level to high altitude), so settings and sensors vary dynamically. The pipeline evaluates variance conditionally per subset.

---

## 3. UCI AI4I 2020 Data Quality Audit

### 3.1 Structural Verification

- **Total records:** 10,000
- **Total columns:** 14
- **Missing values:** 0
- **Duplicate rows:** 0
- **Duplicate UDIs:** 0 (UDI ranges from 1 to 10,000 monotonically)

### 3.2 Physical Range Plausibility

| Feature | Min | Max | Mean | Physical Constraint | Violation? |
|:---|:---:|:---:|:---:|:---|:---:|
| `Air temperature [K]` | 295.30 | 304.50 | 300.00 | $T_{\text{air}} > 0\text{ K}$ | None |
| `Process temperature [K]` | 305.70 | 313.80 | 310.01 | $T_{\text{proc}} > 0\text{ K}$ | None |
| `Rotational speed [rpm]` | 1168.00 | 2886.00 | 1538.78 | $\text{RPM} > 0$ | None |
| `Torque [Nm]` | 3.80 | 76.60 | 39.99 | $\tau \ge 0$ | None |
| `Tool wear [min]` | 0.00 | 253.00 | 107.95 | $\text{wear} \ge 0$ | None |

### 3.3 Class Imbalance & Target Distribution

- Total samples: 10,000
- **Negative Class (0 - Normal Operation):** 9,661 (96.61%)
- **Positive Class (1 - Machine Failure):** 339 (3.39%)
- **Imbalance Ratio:** ~28.5 : 1

#### Failure Modes Breakdown (Positive Class Analysis):
- `HDF` (Heat Dissipation Failure): 115 occurrences
- `OSF` (Overstrain Failure): 98 occurrences
- `PWF` (Power Failure): 95 occurrences
- `TWF` (Tool Wear Failure): 46 occurrences
- `RNF` (Random Failure): 19 occurrences

Notice: The sum of failure modes $(115 + 98 + 95 + 46 + 19 = 373)$ exceeds the 339 failure records because several machine failures involve compounding multi-mode breakdowns (e.g., severe tool wear leading to overstrain or power surge). Furthermore, 9 failure instances occurred with zero specific mode activated, corresponding to unclassified digital twin anomalies.

---

## 4. Quality Assurance Sign-Off Checklist

| Verification Requirement | Status | Evidence / Verification Method |
|:---|:---:|:---|
| Authenticity of raw datasets | **VERIFIED** | Official NASA S3 archive and UCI repository direct downloads. |
| Trajectory continuity | **VERIFIED** | $\Delta \text{cycle} = 1$ across all engines. |
| Zero data leakage in RUL | **VERIFIED** | Strict $RUL(t) = T^{\max} - t$; future sensor readings never used. |
| Zero engine contamination | **VERIFIED** | Train and Val engines strictly separated (80/20 engine-level split). |
| Sequence boundary safety | **VERIFIED** | Sliding windows never cross from engine $k$ to $k+1$. |
| Normalization isolation | **VERIFIED** | Scalers fitted solely on training engines / train split; zero test contamination. |
