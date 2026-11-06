# Comprehensive Dataset Analysis Report: Phase 1

**Framework:** Agentic Predictive Maintenance  
**Document:** Dataset Analysis & Exploratory Findings  
**Primary Dataset:** NASA C-MAPSS Turbofan Engine Fleet  
**Secondary Dataset:** UCI AI4I 2020 Predictive Maintenance  

---

## 1. NASA C-MAPSS Degradation Analysis

All observations below are derived strictly from empirical calculations and visualizations stored in `reports/figures/nasa/`.

### 1.1 Fleet Operating Cycle Distribution (Figures 01 & 02)
- In the benchmark training set (`FD001`), 100 engines were run to complete catastrophic failure.
- **Shortest engine lifetime:** Engine 39 failed at **128 cycles**.
- **Longest engine lifetime:** Engine 69 survived for **362 cycles**.
- **Fleet statistics:** Mean lifetime $\mu = 206.31$ cycles, median $= 199.0$ cycles, standard deviation $\sigma = 46.34$ cycles.
- **Analytical Insight:** Fleet lifetime exhibits a quasi-Gaussian distribution with moderate positive skewness. Initial wear and manufacturing variance cause a nearly 3-fold difference in operational lifespan between engines operating under identical nominal sea-level conditions.

### 1.2 Multivariate Sensor Degradation Dynamics (Figures 03 & 04)
Analyzing individual engine degradation trajectories (e.g., Engine 1, which survived 192 cycles) reveals clear physics-based thermodynamic shifts:
1. **Temperatures:**
   - `sensor_2` (LPC outlet temp $T_{24}$): Gradually rises from $\sim 642.0^\circ\text{R}$ to $> 643.5^\circ\text{R}$.
   - `sensor_3` (HPC outlet temp $T_{30}$): Rises from $\sim 1585^\circ\text{R}$ to $> 1600^\circ\text{R}$.
   - `sensor_4` (LPT outlet temp $T_{50}$): Displays the steepest monotonic increase, climbing from $\sim 1400^\circ\text{R}$ to $> 1430^\circ\text{R}$ prior to breakdown.
2. **Pressures & Flow Ratios:**
   - `sensor_7` (HPC outlet pressure $P_{30}$): Decreases monotonically from $\sim 554\text{ psia}$ to $< 551\text{ psia}$ as compressor internal leakage increases.
   - `sensor_11` (HPC static pressure $P_{s30}$): Rises from $47.2\text{ psia}$ to $> 48.2\text{ psia}$.
   - `sensor_12` (Ratio of fuel flow to $P_{s30}$): Drops from $\sim 522$ to $< 519$, signaling degraded combustion efficiency.
3. **Core Speeds:**
   - `sensor_9` and `sensor_14` (Physical and corrected core speeds $N_c$, $N_{c\text{R}}$): Drift upward as the engine core works harder to deliver demanded thrust despite diminished compressor efficiency.

### 1.3 Sensor Correlation Heatmap (Figure 05)
Pearson correlation analysis across active features and the target variable `RUL` demonstrates:
- **Strong Positive Correlation with RUL (Sensors that decline as failure nears):**
  - `sensor_7` ($r = +0.66$): HPC outlet pressure.
  - `sensor_12` ($r = +0.67$): Ratio of fuel flow to static pressure.
  - `sensor_20` ($r = +0.63$): HPT coolant bleed.
  - `sensor_21` ($r = +0.64$): LPT coolant bleed.
- **Strong Negative Correlation with RUL (Sensors that rise as failure nears):**
  - `sensor_4` ($r = -0.68$): LPT outlet temperature.
  - `sensor_11` ($r = -0.70$): HPC static pressure.
  - `sensor_15` ($r = -0.64$): Bypass ratio ($BPR$).
  - `sensor_2` ($r = -0.60$): LPC outlet temperature.

### 1.4 Operational Settings & Operating Regimes (Figure 06)
- In `FD001`, all engines operate under a single condition (Sea Level).
- `setting_1` has a mean of $0.0000$ with noise span $[-0.0087, +0.0087]$.
- `setting_2` has a mean of $0.0000$ with noise span $[-0.0006, +0.0006]$.
- `setting_3` is unvarying at $100.0$.
- In `FD002` and `FD004`, flight conditions span 6 distinct clusters corresponding to altitude, Mach number, and throttle combinations.

### 1.5 Cross-Engine Sensor Behavior & Fleet Generalizability (Figures 08 & 09)
Overlaying sensor trajectories across distinct engines (Engines 1, 5, 10, 20, 50):
- In the initial healthy phase ($t < 50$ cycles), sensor values across all engines occupy a tight, overlapping band.
- As operational wear accelerates, the variance between engine trajectories widens. However, all engines display the same terminal failure signature (e.g., `sensor_4` crossing $\sim 1425^\circ\text{R}$).
- This validates the hypothesis that degradation patterns are fleet-invariant, enabling transfer learning across engines.

### 1.6 Architectural Comparison across Subsets FD001–FD004 (Figure 10)

| Subset | Operating Conditions | Fault Modes Present | Training Records | Fleet Size (Train Engines) | Test Engines |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **FD001** | 1 (Sea Level) | 1 (High Pressure Compressor) | 20,631 | 100 | 100 |
| **FD002** | 6 (Complex Flights) | 1 (High Pressure Compressor) | 53,759 | 260 | 259 |
| **FD003** | 1 (Sea Level) | 2 (HPC + Fan Degradation) | 24,720 | 100 | 100 |
| **FD004** | 6 (Complex Flights) | 2 (HPC + Fan Degradation) | 61,249 | 248 | 249 |

### 1.7 Two-Phase Degradation Progression (Figures 12 & 13)
Normalizing engine lifetimes to percentage of life completed ($0\%$ to $100\%$):
- **Phase 1: Healthy Operational Steady State ($0\% - 65\%$ Life Completed):**
  Sensor measurements exhibit flat, stationary random-walk fluctuations around healthy baselines. Linear RUL degradation assumptions in this phase are physically inaccurate.
- **Phase 2: Accelerated Degradation ($65\% - 100\%$ Life Completed):**
  Defects propagate non-linearly. The rate of drift $dx/dt$ accelerates exponentially during the final 30–50 cycles before complete failure.
- **Health Regime Distribution Shift:**
  Dividing operations into **Healthy** ($RUL > 100$), **Degraded** ($30 < RUL \le 100$), and **Critical** ($RUL \le 30$) reveals statistically significant shifts in median sensor distributions, justifying the multi-horizon early-warning targets ($10, 20, 30$ cycles).

---

## 2. UCI AI4I 2020 Predictive Maintenance Analysis

Derived from `reports/figures/ai4i/`:

### 2.1 Class Imbalance & Product Tiers (Figure 01)
- 9,661 Normal instances (96.61%) vs. 339 Failure instances (3.39%).
- Product Tier Breakdown:
  - **Type L (Low Quality - 50% of fleet):** 5,000 units, failure rate $\approx 3.92\%$
  - **Type M (Medium Quality - 30% of fleet):** 2,997 units, failure rate $\approx 2.77\%$
  - **Type H (High Quality - 20% of fleet):** 2,003 units, failure rate $\approx 1.50\%$
- Insight: Failure susceptibility decreases monotonically as product quality tier increases ($L \to M \to H$).

### 2.2 Failure Modes Hierarchy (Figure 02)
1. **Heat Dissipation Failure (HDF) - 115 cases (33.9% of failures):** Occurs under inadequate convection cooling ($\Delta T < 8.6\text{ K}$ and $\text{RPM} < 1380$).
2. **Overstrain Failure (OSF) - 98 cases (28.9% of failures):** Occurs when the product of torque and tool wear exceeds critical thresholds.
3. **Power Failure (PWF) - 95 cases (28.0% of failures):** Driven by anomalous mechanical power ($P < 3500\text{ W}$ or $P > 9000\text{ W}$).
4. **Tool Wear Failure (TWF) - 46 cases (13.6% of failures):** Caused by cumulative cutting wear exceeding tool replacement limits ($200 - 240\text{ min}$).
5. **Random Failure (RNF) - 19 cases (5.6% of failures):** Independent stochastic breakdown with $0.1\%$ background probability.

### 2.3 Physical Engineering Interactions (Figures 03, 04, 05)
- **Torque vs. Rotational Speed:** Machines operate along an inverse hyperbolic torque-speed power curve. Mechanical power $P = \tau \times \frac{2\pi \cdot \text{RPM}}{60}$ separates normal milling operations from stall/overload conditions.
- **Temperature Difference $\Delta T$:** Process temperature minus Air temperature has a strong negative correlation with HDF failures.
- **Overstrain Indicator:** $\text{Torque} \times \text{Tool wear}$ clearly isolates OSF failure clusters from normal operations.
