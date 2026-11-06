# Temporal Analysis and Time-Series Modeling Foundations: Phase 1

**Framework:** Agentic Predictive Maintenance  
**Document:** Temporal Dynamics, Sequence Construction & Target Formulations  
**Primary Focus:** NASA C-MAPSS Multivariate Engine Trajectories  

---

## 1. Time-Series Properties of Degradation Trajectories

The NASA C-MAPSS dataset provides genuine multivariate time-series records generated through high-fidelity turbofan aero-thermodynamic simulation:
- **Sampling Frequency:** One aggregate snapshot per completed operational flight cycle.
- **Fleet Heterogeneity:** Each time series originates from a distinct engine exhibiting unknown initial manufacturing tolerances and minor initial wear.
- **Non-Stationarity:** The underlying stochastic process transitions from a wide-sense stationary healthy state into a non-stationary drift regime characterized by expanding variance and directional drift.

```
Cycle 1                                 Cycle ~120-150                       Cycle T_max
  │                                           │                                    │
  ▼                                           ▼                                    ▼
┌───────────────────────────────────────────┬──────────────────────────────────────┐
│  Phase 1: Stationary Healthy Regime       │  Phase 2: Accelerated Wear Regime    │
│  (Random noise around baseline)           │  (Exponential degradation drift)     │
└───────────────────────────────────────────┴──────────────────────────────────────┘
  RUL: High (> 100 cycles)                    RUL: Low (<= 30 cycles, Critical)
```

---

## 2. Backward-Looking Temporal Feature Engineering

To provide temporal context without target leakage, all engineered features are strictly backward-looking and isolated within individual engine trajectories.

### 2.1 Backward Rolling Statistics (Windows: 5, 10, 20 Cycles)
For a sensor measurement $x_t$ of engine $i$ at cycle $t$:
$$\mu_{t, W} = \frac{1}{\min(t, W)} \sum_{k=0}^{\min(t, W)-1} x_{t-k}$$
$$\sigma_{t, W} = \sqrt{\frac{1}{\min(t, W)-1} \sum_{k=0}^{\min(t, W)-1} (x_{t-k} - \mu_{t, W})^2}$$
- **Window $W = 5$:** Captures short-term sensor volatility and immediate operational fluctuations.
- **Window $W = 10$:** Smooths flight-to-flight high-frequency measurement noise.
- **Window $W = 20$:** Extracts medium-term degradation trajectories and baseline trend drift.
- **Initial Boundary Handling:** Using `min_periods = 1` avoids introducing `NaN` values during the initial cycles of an engine's flight life.

### 2.2 First-Order Backward Differences (Instantaneous Velocity)
$$\Delta x_t = x_t - x_{t-1}, \quad \text{with } \Delta x_1 = 0.0$$
Captures the instantaneous rate of degradation drift. During normal operation, $\mathbb{E}[\Delta x_t] \approx 0$. As component wear accelerates, $\Delta x_t$ exhibits sustained positive or negative drift.

### 2.3 Baseline Deviation (Cumulative Wear Indicator)
Each engine operates at a slightly different baseline due to manufacturing tolerances. To measure true structural wear, we calculate:
$$\mu_{\text{baseline}}(i) = \frac{1}{K}\sum_{k=1}^K x_{i, k}, \quad \text{where } K = 5$$
$$\text{dev}_t = x_{i, t} - \mu_{\text{baseline}}(i)$$
This feature normalizes individual engine tolerances, allowing deep learning and tree models to focus directly on component degradation.

---

## 3. Multivariate 3D Temporal Sequence Construction

Deep sequential architectures (LSTM, GRU, Temporal Convolutional Networks, Transformers) require three-dimensional tensor inputs:
$$\mathbf{X} \in \mathbb{R}^{N \times L \times D}$$
where:
- $N$: Total number of generated sliding windows across the fleet
- $L$: Sequence window length (history length in cycles)
- $D$: Number of input features per time step

### 3.1 Sliding Window Generation & Zero Cross-Engine Leakage
For an engine trajectory of length $T_i$ and window length $L$:
$$\text{Window}_s = [x_s, x_{s+1}, \dots, x_{s+L-1}], \quad \text{for } s \in \{1, 2, \dots, T_i - L + 1\}$$
- **Total Windows from Engine $i$:** $T_i - L + 1$ (for stride = 1).
- **Absolute Boundary Rule:** Sequences are generated strictly within engine $i$. When the window reaches $T_i$, window extraction terminates. No sequence ever spans between engine $i$ and engine $i+1$.

### 3.2 Sequence Dimensions Achieved (FD001 Benchmark)

| Window Length ($L$) | Feature Count ($D$) | Train Sequences ($N_{\text{train}}$) | Val Sequences ($N_{\text{val}}$) | Test Sequences ($N_{\text{test}}$) |
|:---:|:---:|:---:|:---:|:---:|
| **30 Cycles** | 153 | 14,020 | 3,711 | 10,196 |
| **50 Cycles** | 153 | 12,420 | 3,311 | 8,262 |
| **60 Cycles** | 153 | 11,620 | 3,111 | 7,363 |

> [!TIP]
> All sequence arrays are saved directly in compressed NumPy archive format (`.npz`) containing `X`, `y_rul`, `y_ew10`, `y_ew20`, `y_ew30`, `engine_ids`, `end_cycles`, and `feature_names`. This avoids gigabyte-scale CSV overhead and eliminates parsing latency during deep learning training.

---

## 4. RUL Target Formulations

### 4.1 Training Trajectories (Run-to-Failure)
$$RUL(i, t) = T_i^{\max} - t$$
- At initial cycle $t=1$: $RUL = T_i^{\max} - 1$
- At final cycle $t=T_i^{\max}$: $RUL = 0$

### 4.2 Test Trajectories Reconstruction
Test engines were intentionally truncated prior to failure at an unknown cycle $T_{\text{test}}^{\max}$. NASA provides an official terminal RUL vector $\mathbf{r}^* \in \mathbb{R}^{100}$ representing the remaining cycles after $T_{\text{test}}^{\max}$.
For any observed cycle $t \in [1, T_{\text{test}}^{\max}]$:
$$RUL_{\text{test}}(i, t) = r_i^* + (T_{\text{test}, i}^{\max} - t)$$
- At terminal cycle $t=T_{\text{test}, i}^{\max}$: $RUL_{\text{test}} = r_i^*$ (matching official ground truth).
- This reconstruction enables full sequence-level evaluation across the entire test trajectory.

### 4.3 Multi-Horizon Early-Warning Binary Classification
To support proactive decision support, three early-warning horizons were created:
$$y_{\text{EW}, H}(i, t) = \begin{cases} 1, & \text{if } RUL(i, t) \le H \\ 0, & \text{if } RUL(i, t) > H \end{cases}, \quad H \in \{10, 20, 30\}$$

#### Empirical Class Distribution in FD001 (Train Windows $L=30$):
- **Horizon 10:** Positive class frequency $= 5.7\%$ (Critical emergency warning)
- **Horizon 20:** Positive class frequency $= 11.4\%$ (Maintenance dispatch alert)
- **Horizon 30:** Positive class frequency $= 17.5\%$ (Supply chain / parts scheduling window)
- **Imbalance Handling:** For Phase 2 model training, we recommend focal loss or threshold tuning rather than oversampling raw time-series sequences.
