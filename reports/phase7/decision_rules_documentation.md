# Phase 7: Decision Rules and Maintenance Priority Documentation
## Agentic Predictive Maintenance Decision-Support Engine

### 1. Architectural Philosophy and Industrial Safety Principle
The recommendation engine serves strictly as a **decision-support advisory system**. Under no operational circumstances does the engine issue autonomous, closed-loop machine-control or equipment actuation commands. 

All derived maintenance action items require certified human engineering authorization (`human_approval_required = True` for all non-routine priorities). This design adheres to industrial safety standards (ISO 13374 Condition Monitoring and Diagnostics, ISO 13849 Safety-Related Control Systems, and FAA/EASA Part 145 Maintenance Organization Standards).

---

### 2. Multi-Tier Priority Classification Framework

The system defines 5 standardized, configurable operational priority tiers:

| Priority Tier | Urgency Horizon | Human Approval Required | Industrial Action Protocol |
| :--- | :--- | :---: | :--- |
| **Informational** | Routine / Scheduled | No | Normal operation within design envelope; standard monitoring interval |
| **Low priority** | Advisory / Next 25 Cycles (or Shift) | No | Minor telemetry drift; log trending; verify sensor calibration |
| **Medium priority** | Planned / Next Line Check (or 24h) | Yes | Targeted diagnostic; inspect specific subsystem; schedule preventive check |
| **High priority** | Immediate / Within 2–5 Cycles (or Shift End) | Yes | Pre-position replacement parts; throttle load; borescope/ultrasonic inspection |
| **Critical review required** | Urgent Immediate Intervention | Yes | Ground unit or stop machining; prevent catastrophic failure (AOG/tool burst) |

> **IMPORTANT NOTICE ON THRESHOLDS:**
> All numeric thresholds documented herein are **configurable experimental domain heuristics** derived from offline validation benchmarks. In commercial operations, thresholds must be calibrated against fleet operating regimes and certified by OEM reliability engineers. NASA Remaining Useful Life (RUL) and UCI AI4I Failure Probability are treated as strictly independent, non-interchangeable quantities.

---

### 3. NASA C-MAPSS Decision Rules

NASA C-MAPSS models predict **Remaining Useful Life (RUL in flight cycles)**, complemented by unsupervised multivariate sensor anomaly severity (LSTM Autoencoder reconstruction error) and early-warning horizon classifiers ($RUL \le 10, 20, 30$ cycles).

#### Configurable Threshold Parameters:
- `critical_rul_cycles = 20.0`
- `warning_rul_cycles = 50.0`
- `advisory_rul_cycles = 100.0`
- `early_warning_risk_high = 0.80` (80% probability of failure within 10 cycles)

#### Rule Evaluation Matrix:
1. **Critical Review Required:**
   - **Condition:** $\text{Predicted RUL} \le 20.0\text{ cycles}$ OR $\text{Early Warning Risk}_{10} \ge 80\%$
   - **Risk Status:** `imminent_exhaustion`
   - **Actions:** Schedule urgent borescope inspection; pre-allocate line-replaceable unit (LRU) turbofan engine; alert flight operations.
2. **High Priority:**
   - **Condition A:** $20.0 < \text{Predicted RUL} \le 50.0\text{ cycles}$ AND $\text{Severity} \in \{\text{Medium}, \text{High}\}$
   - **Condition B (Conflicting Evidence):** $\text{Predicted RUL} > 100.0\text{ cycles}$ AND $\text{Severity} == \text{"High anomaly"}$
   - **Risk Status:** `accelerated_degradation` (Condition A) or `conflicting_evidence` (Condition B)
   - **Actions:** Detailed thermal and pressure bleed audit; high-rate telemetry logging; hangar standby.
3. **Medium Priority:**
   - **Condition:** $50.0 < \text{Predicted RUL} \le 100.0\text{ cycles}$ with $\text{Severity} == \text{"Medium anomaly"}$, OR $\text{Predicted RUL} \le 50.0$ with $\text{Severity} == \text{"Normal"}$
   - **Risk Status:** `incipient_wear`
   - **Actions:** Planned line check inspection targeting the specific subsystem identified by SHAP/temporal attributions.
4. **Low Priority:**
   - **Condition:** $50.0 < \text{Predicted RUL} \le 100.0\text{ cycles}$ with $\text{Severity} \in \{\text{Normal}, \text{Low anomaly}\}$, OR $\text{Severity} == \text{"Low anomaly"}$
   - **Risk Status:** `incipient_wear` / `minor_drift`
   - **Actions:** Increase sampling frequency; log baseline deviations; no flight interruption required.
5. **Informational:**
   - **Condition:** $\text{Predicted RUL} > 100.0\text{ cycles}$ AND $\text{Severity} == \text{"Normal"}$
   - **Risk Status:** `nominal`
   - **Actions:** Standard maintenance schedule maintained.

#### Subsystem Mapping for NASA C-MAPSS:
Every recommendation identifies the exact physical subsystem driving degradation by inspecting SHAP/Integrated Gradients attributions:

| Feature / Sensor | Turbofan Subsystem | Standard Diagnostic Protocol | Reference Standard |
| :--- | :--- | :--- | :--- |
| `sensor_11` | High-Pressure Compressor (HPC) Outlet | Borescope inspection of HPC outlet; calibrate static thermocouple T30 | EASA / FAA Part 145 Engine Standards |
| `sensor_4` | Low-Pressure Turbine (LPT) Exhaust | Inspect LPT stator blades for thermal erosion; check thermocouple harness | OEM Turbofan Manual §72-50 |
| `sensor_12` | Fan & Bypass Duct Aerodynamics | Inspect bypass duct aerodynamic seals; check fan blade tip clearances | Engine Maintenance Manual (EMM) |
| `sensor_13` | Coolant Bleed & Variable Bleed Valves | Test high-pressure core bleed valve actuation; check seal integrity | CMP Diagnostic Bulletin §75-20 |
| `sensor_14` | High-Pressure Pressure Regulation | Calibrate core coolant differential pressure transducer | EMM Instrumentation §77-10 |
| `sensor_8` | Fan & Low-Pressure Rotor Bearings | Vibration spectral analysis on fan shaft; inspect bearing lubrication | Vibration Analysis Standard ISO 10816 |
| `sensor_9` | Core High-Pressure Rotor Shaft | Inspect High-Pressure core spool rotational balance; check mid-shaft bearings | Overhaul Manual §72-30 |

---

### 4. UCI AI4I 2020 Decision Rules

The UCI AI4I dataset models **discrete machine breakdown probability (0.0 to 1.0)** for CNC machining centers, reflecting mechanical, thermal, and tool wear stress.

#### Configurable Threshold Parameters:
- `critical_prob = 0.75` (75% breakdown probability)
- `warning_prob = 0.40` (40% breakdown probability)
- `advisory_prob = 0.20` (20% breakdown probability)

#### Rule Evaluation Matrix:
1. **Critical Review Required:**
   - **Condition:** $\text{Failure Risk} \ge 0.75$
   - **Risk Status:** `imminent_breakdown`
   - **Actions:** Immediate spindle stop at shift end; replace cutting insert; inspect tool holder runout.
2. **High Priority:**
   - **Condition A:** $0.40 \le \text{Failure Risk} < 0.75$ AND $\text{Severity} \in \{\text{Medium}, \text{High}\}$
   - **Condition B (Conflicting Evidence):** $\text{Failure Risk} < 0.20$ AND $\text{Severity} == \text{"High anomaly"}$
   - **Risk Status:** `elevated_risk` (Condition A) or `conflicting_evidence` (Condition B)
   - **Actions:** Throttle spindle speed/feed rate by 20%; inspect mechanical drive train; recalibrate torque sensors.
3. **Medium Priority:**
   - **Condition:** $0.40 \le \text{Failure Risk} < 0.75$ with normal anomaly status, OR $\text{Severity} == \text{"Medium anomaly"}$
   - **Risk Status:** `elevated_risk` / `process_drift`
   - **Actions:** Planned maintenance within 24 hours; flush coolant delivery lines; inspect tool wear progression.
4. **Low Priority:**
   - **Condition:** $0.20 \le \text{Failure Risk} < 0.40$
   - **Risk Status:** `monitoring_recommended`
   - **Actions:** Continue current batch run; log temperature differential; verify lubrication.
5. **Informational:**
   - **Condition:** $\text{Failure Risk} < 0.20$ AND $\text{Severity} == \text{"Normal"}$
   - **Risk Status:** `nominal`
   - **Actions:** Standard shift production; routine tool wear check at shift handover.

#### Subsystem Mapping for UCI AI4I:
| Engineered Feature | CNC Machine Tool Subsystem | Standard Diagnostic Protocol | Reference Standard |
| :--- | :--- | :--- | :--- |
| `Mechanical power` | Spindle Motor & Drive | Inspect electrical drive inverter, motor winding resistance, and belt tension | CNC Safety ISO 10791 |
| `Tool wear` | Cutting Tool Assembly | Replace insert; check tool holder runout tolerance ($< 0.005\text{ mm}$) | ISO 3685 Tool-Life Testing |
| `Rotational speed` | Spindle Bearings & Tachometer | High-speed acoustic emission test; verify spindle oil mist lubrication | ISO 1940-1 Balance Quality |
| `Overstrain indicator` | Workpiece Clamping & Feed Drive | Reduce axis feed rate by 15%; verify fixture hydraulic clamping pressure | Shop Safety Protocol §M-40 |
| `Torque` | Spindle Drive Gearbox | Inspect drive train gearbox backlash; recalibrate torque strain gauge | ISO 230-2 Axis Accuracy |
| `Torque to speed ratio` | Mechanical Drive Coupling | Inspect mechanical coupling for slippage or binding during load transitions | Preventive Maintenance SOP |
| `Heat dissipation index`| Coolant Delivery & Heat Exchanger | Flush coolant lines; verify nozzle pressure ($> 2.5\text{ bar}$); clean radiator | ISO 14955 Environmental CNC |

---

### 5. Conflicting Evidence Resolution Matrix

Industrial environments frequently exhibit discrepancies between supervised models and unsupervised detectors (e.g., a sensor develops electrical drift while the physical turbine blade remains intact). The engine resolves conflicting evidence according to the following matrix:

| Supervised Model Output | Unsupervised Anomaly Status | Resolved Priority | Assigned Risk Status | Reason & Required Action |
| :--- | :--- | :--- | :--- | :--- |
| High RUL ($> 100$ cyc) / Low Risk ($< 20\%$) | **High anomaly** | **High priority** | `conflicting_evidence` | Divergence detected. Human review required to differentiate sensor instrumentation failure from unmodeled machine stress. |
| Low RUL ($\le 20$ cyc) / High Risk ($\ge 75\%$) | **Normal** | **Critical review required** | `imminent_exhaustion` | Conservative safety protocol: Supervised failure signature takes immediate precedence. Urgent inspection required. |
| Moderate RUL ($50\text{--}100$ cyc) | **High anomaly** | **High priority** | `incipient_wear` | Anomaly accelerates priority assignment from Medium to High. |
| Missing RUL or Anomaly Score | Available metrics | Priority from available metrics | Graceful degradation | Missing fields are never fabricated. Evaluation proceeds with clear metadata documentation. |

---

### 6. Supporting Evidence Linkage

Every generated recommendation links each proposed action directly to quantitative model outputs:
1. **Supervised Prediction:** Exact predicted RUL in cycles or failure probability percentage.
2. **Unsupervised Anomaly:** Exact reconstruction error score and categorical severity.
3. **Attribution Drivers:** Top 3 physical features with normalized SHAP/Integrated Gradients importance values and attribution directions (`positive` = increasing degradation, `negative` = protective factor).
4. **Safety Disclaimer:** Explicit reminder of decision-support scope on every generated payload.
