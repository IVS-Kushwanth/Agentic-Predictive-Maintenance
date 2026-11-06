# Mathematical Foundations & Theoretical Derivations
## Agentic Predictive Maintenance Framework

This document details the mathematical formulations, derivations, objective loss functions, and evaluation metrics implemented across the framework.

---

## 1. Regression Metrics & Asymmetric Loss Functions

### 1.1 Root Mean Squared Error (RMSE)
Measures the standard deviation of regression prediction residuals:
$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2}$$
- **Variables**: $N$ is total sample count; $y_i$ is ground truth Remaining Useful Life (RUL); $\hat{y}_i$ is predicted RUL.
- **Properties**: Penalizes large outlier errors quadratically; sensitive to extreme prediction errors.

### 1.2 Mean Absolute Error (MAE)
Measures the average magnitude of absolute residuals:
$$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |\hat{y}_i - y_i|$$
- **Properties**: Less sensitive to outliers than RMSE; provides an intuitive linear measure of average cycle error.

### 1.3 NASA PHM 2008 Asymmetric Scoring Function
In industrial predictive maintenance, an overestimate of RUL ($\hat{y} > y$, late prediction) is hazardous because the machine risks operating past catastrophic failure. Underestimating RUL ($\hat{y} < y$, early prediction) incurs premature maintenance cost but preserves operational safety.

For sample $i$ with residual $d_i = \hat{y}_i - y_i$:
$$s_i = \begin{cases} \exp\left(-\frac{d_i}{a_1}\right) - 1 & \text{if } d_i < 0 \quad (\text{Early Prediction / Conservative}) \\ \exp\left(\frac{d_i}{a_2}\right) - 1 & \text{if } d_i \ge 0 \quad (\text{Late Prediction / Hazardous}) \end{cases}$$
Where $a_1 = 13$ and $a_2 = 10$.
$$\text{NASA Score } S = \sum_{i=1}^{N} s_i$$
- **Asymmetric Penalty Ratio**: For an error of $|d_i| = 20$ cycles:
  - Early ($d_i = -20$): $s_i = \exp(20/13) - 1 \approx 3.65$
  - Late ($d_i = +20$): $s_i = \exp(20/10) - 1 \approx 6.39$ ($1.75\times$ penalty)
- For an error of $|d_i| = 40$ cycles:
  - Early ($d_i = -40$): $s_i = \exp(40/13) - 1 \approx 20.69$
  - Late ($d_i = +40$): $s_i = \exp(40/10) - 1 \approx 53.60$ ($2.59\times$ penalty)

### 1.4 Huber Loss (Smooth L1 Loss)
Combines the quadratic efficiency of MSE near zero with the linear robustness of MAE for large residuals:
$$\mathcal{L}_\delta(e) = \begin{cases} \frac{1}{2} e^2 & \text{if } |e| \le \delta \\ \delta \left(|e| - \frac{1}{2} \delta\right) & \text{otherwise} \end{cases} \quad \text{where } e = y - \hat{y}, \quad \delta = 1.0$$

---

## 2. Classification Metrics & Objective Functions

### 2.1 Binary Cross-Entropy (Log Loss)
For ground truth $y \in \{0, 1\}$ and predicted probability $p = \sigma(z) \in [0, 1]$:
$$\mathcal{L}_{\text{BCE}}(y, p) = -\left[ y \ln(p) + (1 - y) \ln(1 - p) \right]$$

### 2.2 Precision, Recall, and F1-Score
Given True Positives ($TP$), False Positives ($FP$), and False Negatives ($FN$):
$$\text{Precision } P = \frac{TP}{TP + FP}, \quad \text{Recall } R = \frac{TP}{TP + FN}$$
$$F_1 = 2 \cdot \frac{P \cdot R}{P + R} = \frac{2 TP}{2 TP + FP + FN}$$

### 2.3 Precision-Recall Area Under Curve (PR-AUC)
For imbalanced classification (3.4% failure prevalence in AI4I), PR-AUC evaluates performance across all decision thresholds $\tau \in [0, 1]$ without bias from True Negatives:
$$\text{PR-AUC} = \int_{0}^{1} P(R) \, dR \approx \sum_{k=1}^{M} (R_k - R_{k-1}) P_k$$

---

## 3. Unsupervised Anomaly Detection Mathematics

### 3.1 Isolation Forest Path Length Scoring
Let $h(x)$ be the path length of observation $x$ from the root to an isolating leaf in an isolation tree. The expected path length of an unsuccessful search in a Binary Search Tree (BST) of size $n$ is:
$$c(n) = 2 \left( \ln(n - 1) + \gamma \right) - \frac{2(n - 1)}{n}$$
where $\gamma \approx 0.5772156649$ (Euler-Mascheroni constant).

The anomaly score $s(x, n)$ over an ensemble of $T$ trees is defined as:
$$s(x, n) = 2^{-\frac{\mathbb{E}[h(x)]}{c(n)}} \quad \text{where } \mathbb{E}[h(x)] = \frac{1}{T} \sum_{t=1}^{T} h_t(x)$$
- When $\mathbb{E}[h(x)] \to 0 \implies s \to 1$ (Definite Anomaly).
- When $\mathbb{E}[h(x)] \to c(n) \implies s \to 0.5$ (Nominal Observation).
- When $\mathbb{E}[h(x)] \to n-1 \implies s \to 0$ (High-density Normal Cluster).

### 3.2 LSTM Autoencoder Reconstruction Loss
For sequence tensor $X \in \mathbb{R}^{B \times L \times D}$ and decoded reconstruction $\hat{X} \in \mathbb{R}^{B \times L \times D}$:
$$\mathcal{L}_{\text{recon}} = \frac{1}{B \cdot L \cdot D} \sum_{b=1}^{B} \sum_{t=1}^{L} \sum_{d=1}^{D} (x_{b, t, d} - \hat{x}_{b, t, d})^2$$

---

## 4. Explainable AI & Feature Attribution Mathematics

### 4.1 Cooperative Game Theory (Shapley Value Formulation)
Shapley values allocate fair marginal contributions across $M$ collaborating features in coalitional game theory:
$$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|! (|N| - |S| - 1)!}{|N|!} \left[ v(S \cup \{i\}) - v(S) \right]$$
Where:
- $N$ is the grand coalition of all available features.
- $S$ is a feature subset excluding feature $i$.
- $v(S) = \mathbb{E}[f(x) | x_S]$ is the conditional model expectation over subset $S$.

#### Axiomatic Guarantees:
1. **Efficiency**: $\sum_{i=1}^{|N|} \phi_i = f(x) - \mathbb{E}[f(x)]$ (exact additive attribution).
2. **Symmetry**: If $v(S \cup \{i\}) = v(S \cup \{j\})$ for all $S$, then $\phi_i = \phi_j$.
3. **Dummy / Null Player**: If $v(S \cup \{i\}) = v(S)$ for all $S$, then $\phi_i = 0$.
4. **Additivity**: For ensemble $f = \sum w_k f_k$, $\phi_i(f) = \sum w_k \phi_i(f_k)$.

### 4.2 Integrated Gradients
For deep sequence models, Integrated Gradients compute the path integral of the gradient along the straight line from a nominal baseline $x'$ to input $x$:
$$\text{IG}_i(x) = (x_i - x'_i) \times \int_{0}^{1} \frac{\partial F(x' + \alpha(x - x'))}{\partial x_i} \, d\alpha$$
Approximated via $M$-step Riemann summation:
$$\text{IG}_i^{\text{approx}}(x) = (x_i - x'_i) \times \frac{1}{M} \sum_{k=1}^{M} \frac{\partial F\left(x' + \frac{k}{M}(x - x')\right)}{\partial x_i}$$
