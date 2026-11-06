# Comprehensive Project Glossary
## Agentic Predictive Maintenance Framework

This glossary defines 30+ core terms, providing both a simplified explanation and an advanced technical definition.

---

### 1. Predictive Maintenance (PdM)
- **Simple Definition**: Servicing machines right before they break down based on sensor health data, avoiding unnecessary scheduled maintenance.
- **Technical Meaning**: Condition-based monitoring strategy that uses statistical learning and physics models to predict degradation trajectories and schedule optimal interventions.

### 2. Remaining Useful Life (RUL)
- **Simple Definition**: The number of operating hours or flight cycles a machine has left before it fails.
- **Technical Meaning**: The conditional random variable $\tau = T_{fail} - t$, representing remaining operating cycles from current time $t$ until physical functional failure.

### 3. Multivariate Time Series
- **Simple Definition**: Multiple sensor readings recorded over time from the same machine.
- **Technical Meaning**: A vector-valued stochastic process $X_t = [x_{1,t}, x_{2,t}, \dots, x_{D,t}]^T \in \mathbb{R}^D$ sampled at discrete temporal intervals.

### 4. Sliding Window
- **Simple Definition**: Looking at the most recent chunk of sensor history (e.g., past 30 cycles) to make a prediction.
- **Technical Meaning**: A temporal segmentation operator extracting continuous sub-matrices $W_t = [X_{t-L+1}, \dots, X_t] \in \mathbb{R}^{L \times D}$ of fixed length $L$.

### 5. Data Leakage
- **Simple Definition**: Cheating during model training by accidentally peeking at future test data or failure answers.
- **Technical Meaning**: The contamination of training data with information from validation/test distributions or downstream failure states, artificially inflating performance.

### 6. NASA Asymmetric Scoring Function
- **Simple Definition**: A penalty score that punishes overestimating engine life much harder than underestimating it.
- **Technical Meaning**: An asymmetric exponential loss function $S = \sum s_i$ penalizing positive residuals ($d_i > 0$) with $\exp(d_i/10)-1$ and negative residuals ($d_i < 0$) with $\exp(-d_i/13)-1$.

### 7. SHAP (SHapley Additive exPlanations)
- **Simple Definition**: A mathematical method that scores how much each sensor pushed the model toward a failure prediction.
- **Technical Meaning**: A game-theoretic local feature attribution method computing unique additive values satisfying efficiency, symmetry, and dummy player axioms.

### 8. TreeSHAP
- **Simple Definition**: A fast version of SHAP designed specifically for decision tree models like XGBoost.
- **Technical Meaning**: An exact polynomial-time algorithm for computing Shapley values in tree ensembles using conditional feature expectations in $\mathcal{O}(T L D^2)$ time.

### 9. Integrated Gradients
- **Simple Definition**: Tracing how a deep neural network's decision changes as sensors move from normal to broken.
- **Technical Meaning**: An axiomatic feature attribution method computing the path integral of model gradients along the straight line from a baseline reference to the input.

### 10. LangGraph
- **Simple Definition**: A software framework that connects multiple AI agents into a structured, step-by-step decision graph.
- **Technical Meaning**: A library for building stateful, multi-actor applications with LLMs and deterministic nodes using cyclical state machines.

### 11. StateGraph
- **Simple Definition**: The map that tells AI agents what order to work in and which rules to follow.
- **Technical Meaning**: A parameterized directed graph where nodes represent state-transforming agent functions and edges represent conditional routing predicates.

### 12. Human-in-the-Loop (HITL)
- **Simple Definition**: Requiring a human engineer to approve important decisions before an automated system acts.
- **Technical Meaning**: A system governance architecture where automated algorithmic recommendations require authorized human sign-off prior to state execution.

### 13. LSTM Autoencoder
- **Simple Definition**: A neural network that learns what normal machine sounds like and flags anything unusual.
- **Technical Meaning**: An unsupervised sequence-to-sequence recurrent network that compresses sequence windows into a latent bottleneck and reconstructs them; high reconstruction MSE indicates anomalies.

### 14. Isolation Forest
- **Simple Definition**: An anomaly detector that finds weird data points by seeing how easily they can be separated from normal points.
- **Technical Meaning**: An unsupervised tree ensemble that recursively partitions feature space; anomalous points exhibit significantly shorter average path lengths $\mathbb{E}[h(x)]$.

### 15. PR-AUC (Precision-Recall Area Under Curve)
- **Simple Definition**: The best accuracy metric for rare events like machine breakdowns.
- **Technical Meaning**: The integral of precision over recall across all decision thresholds; unlike ROC-AUC, it is robust against high true negative counts in imbalanced datasets.
