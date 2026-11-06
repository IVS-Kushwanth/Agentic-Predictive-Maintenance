# Comprehensive Literature Review & Related Work
## Agentic Predictive Maintenance Framework

This review establishes the theoretical and empirical context of the project, analyzing key academic contributions across predictive maintenance, remaining useful life estimation, explainable machine learning, and multi-agent systems.

---

## 1. Evolution of Industrial Prognostics & Maintenance Paradigms

Condition-Based Maintenance (CBM) and Prognostics and Health Management (PHM) have progressed through three distinct eras:
1. **First-Generation Physics-Based Modeling**: Grounded in continuum mechanics, fracture propagation laws (Paris-Erdogan law), and thermodynamic degradation kinetics. While physically rigorous, empirical calibration requires precise finite-element modeling of every individual component, rendering fleet-wide scaling infeasible.
2. **Second-Generation Statistical Reliability Models**: Utilized Weibull distribution analysis, Markov decision processes, and proportional hazard models. These techniques model average population survival rates but fail to account for dynamic operating stresses unique to an individual asset.
3. **Third-Generation Data-Driven Machine Learning**: Accelerated by high-frequency sensor telemetry, multivariate machine learning algorithms learn degradation signatures directly from operational history.

---

## 2. Benchmark Datasets in Literature

### 2.1 NASA C-MAPSS Turbofan Degradation Benchmark
Introduced by Saxena et al. (2008) at the NASA Ames Research Center, the Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) remains the canonical international benchmark for Remaining Useful Life (RUL) estimation. The simulation models a 90,000 lb thrust class commercial turbofan engine experiencing high-pressure compressor fouling and turbine blade erosion under sea-level and multi-regime flight envelopes.

Saxena et al. formulated the asymmetric scoring function:
$$s_i = \begin{cases} \exp(-d_i/13) - 1 & d_i < 0 \\ \exp(d_i/10) - 1 & d_i \ge 0 \end{cases}$$
which establishes that overestimating engine life introduces catastrophic operational risks compared to early maintenance intervention.

### 2.2 UCI AI4I 2020 Predictive Maintenance Benchmark
Introduced by Matzka (2020), the AI4I 2020 dataset models a milling machine digital twin. Matzka highlighted the severe challenge of class imbalance (~3.39% failure rate) and detailed five distinct physical breakdown mechanisms: Tool Wear Failure (TWF), Heat Dissipation Failure (HDF), Power Failure (PWF), Overstrain Failure (OSF), and Random Failure (RNF).

---

## 3. Machine Learning & Deep Learning in Time-Series Prognostics

- **Gradient Boosted Decision Trees**: Chen & Guestrin (2016) introduced XGBoost, demonstrating superior performance on tabular feature spaces through second-order Taylor expansion optimization. In predictive maintenance, tree ensembles excel when fed handcrafted rolling window statistics.
- **Recurrent Neural Networks (LSTM & GRU)**: Hochreiter & Schmidhuber (1997) and Cho et al. (2014) formulated gated recurrent units that maintain hidden states across long temporal sequences, overcoming vanishing gradient limitations. Zheng et al. (2017) applied multi-layer LSTMs to C-MAPSS, demonstrating that sequence architectures outperform static tabular regressors.
- **Unsupervised Anomaly Detection**: Liu, Ting, & Zhou (2008) introduced Isolation Forest, proving that anomalous data points can be isolated efficiently through recursive tree partitioning without density estimation. Baldi (2012) formalized deep autoencoders for nonlinear dimensionality reduction and reconstruction error scoring.

---

## 4. Explainable Artificial Intelligence (XAI) in Cyber-Physical Systems

High-consequence industrial domains demand verifiable explanations:
- **Game-Theoretic Attributions**: Lundberg & Lee (2017) established SHAP (SHapley Additive exPlanations), unifying classical cooperative game theory with local additive feature attribution. Lundberg et al. (2020) later developed TreeSHAP, an exact algorithm computing Shapley values in polynomial time for decision tree ensembles.
- **Deep Sequence Attributions**: Sundararajan, Taly, & Yan (2017) introduced Integrated Gradients, satisfying the axioms of Completeness and Implementation Invariance by integrating gradients along a straight line path from a neutral baseline.

---

## 5. Multi-Agent Systems & LangGraph in Industry 4.0

Traditional industrial architectures rely on monolithic scripts or rigid rule engines. Modern agentic architectures deploy collaborative multi-agent teams coordinated through stateful directed graphs:
- **LangGraph StateGraph**: Provides an asynchronous state-machine framework supporting cyclical reasoning, conditional routing, and deterministic human-in-the-loop checkpoints.
- **ISO 13381-1 Condition Monitoring Standard**: Provides the international ontological guideline for condition monitoring and diagnostics of machines, formalizing priority assignment and prescriptive maintenance scheduling.
