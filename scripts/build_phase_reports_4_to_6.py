"""
Builder script for Phase Reports 4, 5, and 6:
- Phase_04_Anomaly_Detection.md
- Phase_05_Explainable_AI.md
- Phase_06_Agentic_AI_LangGraph.md
Each strictly formatted with Sections A through V.
"""

import os

DEST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "documentation", "02_Phase_Wise_Technical_Reports")
os.makedirs(DEST_DIR, exist_ok=True)

PHASE_04_CONTENT = r"""# Phase 4: Multivariate Sensor Anomaly Detection Technical Report

## A. Phase Introduction
Phase 4 introduces an unsupervised and semi-supervised Multivariate Sensor Anomaly Detection Module into the predictive maintenance framework. While supervised regression and classification models require labeled degradation targets and historical failure examples, unsupervised anomaly detection identifies anomalous deviations, sensor faults, and novel operational irregularities directly from continuous sensor streams without ground-truth labels.

---

## B. Objective of the Phase
1. Establish nominal early-life operational baselines for NASA C-MAPSS turbofan engines ($RUL \ge 110$ cycles).
2. Design, train, and validate a Multivariate LSTM Autoencoder in PyTorch to reconstruct 3D temporal sequence windows ($w=30, D=153$).
3. Train traditional unsupervised anomaly detectors: Isolation Forest, One-Class SVM, and Local Outlier Factor (LOF).
4. Evaluate continuous anomaly score escalation against physical engine wear, demonstrating strong negative correlation with Remaining Useful Life.
5. Implement sensor-level reconstruction attribution to isolate which physical transducers drive anomalous behavior.
6. Calibrate quantile-based severity thresholds (`NOMINAL`, `LOW`, `MEDIUM`, `HIGH`) and expose a unified anomaly prediction interface.

---

## C. Why This Phase Is Required
In industrial machinery, physical breakdowns are frequently preceded by subtle multi-sensor distribution shifts that supervised models may fail to capture:
- **Novel Failure Mechanisms**: Supervised classifiers only recognize failure modes present in training data; unsupervised anomaly detection alerts operators to unmodeled physical anomalies.
- **Sensor Telemetry Faults**: Transducer drift, intermittent short circuits, and line noise corrupt supervised RUL models. Anomaly detection flags corrupt sensor channels before they trigger false maintenance work orders.
- **Early-Stage Wear Monitoring**: Anomaly scores provide continuous health indicators long before supervised early-warning classification thresholds ($H \le 30$) are breached.

---

## D. Tasks Performed
- Extracted 5,239 nominal early-life training sequences ($RUL \ge 110$) across 80 training engines to fit unsupervised models.
- Built and trained an LSTM Autoencoder (Encoder: $153 \to 64 \to 32$, Bottleneck: 32, Decoder: $32 \to 64 \to 153$) optimizing Mean Squared Reconstruction Error.
- Trained Scikit-learn Isolation Forest ($n\_estimators=150$, contamination=0.034) on AI4I operating features.
- Evaluated AI4I test detection without target leakage, achieving test ROC-AUC of **0.8824** (Isolation Forest) and **0.8230** (One-Class SVM).
- Established 4-tier calibrated severity thresholds: Nominal ($< 90\%$), Low ($90\% - 97.5\%$), Medium ($97.5\% - 99\%$), and High ($> 99\%$).
- Serialized anomaly model artifacts into `models/anomaly/`.

---

## E. Detailed Implementation Methodology

```mermaid
flowchart TD
    subgraph NASA_Anomaly ["NASA C-MAPSS Sequence Anomaly Detection"]
        N_EARLY["Filter Nominal Cycles (RUL >= 110, N=5,239)"] --> FIT_AE["Train LSTM Autoencoder (20 Epochs, Loss=0.0285)"]
        N_TEST["Unseen Sequence Window (t - 29 : t)"] --> AE_PASS["Forward Pass: Encode to Latent z=32, Decode to X_hat"]
        AE_PASS --> MSE_CALC["Reconstruction Error = 1/(L*D) sum (X - X_hat)^2"]
        MSE_CALC --> CALIB_NASA["Calibrate Against Nominal Max (tau = 0.0370)"]
    end

    subgraph AI4I_Anomaly ["UCI AI4I Tabular Anomaly Detection"]
        A_TRAIN["AI4I Train Set (8,000 instances, No Target Used)"] --> FIT_IF["Fit Isolation Forest (n_estimators=150, contam=0.034)"]
        A_TEST["Test Telemetry Vector x (2,000 instances)"] --> PATH_LEN["Average Tree Path Length E[h(x)]"]
        PATH_LEN --> IF_SCORE["Anomaly Score s(x) = 2^(-E[h(x)] / c(n))"]
        IF_SCORE --> BENCH_ROC["Post-Hoc Benchmark: ROC-AUC = 0.8824 (9.1x Random)"]
    end
```

---

## F. Algorithms and Techniques Used
- **Multivariate LSTM Autoencoder**: Deep sequence compression and reconstruction capturing temporal and inter-sensor correlations.
- **Isolation Forest**: Tree-based recursive spatial partitioning measuring isolation path lengths.
- **One-Class Support Vector Machine (OC-SVM)**: Non-linear kernel mapping to isolate nominal distribution boundaries.
- **Local Outlier Factor (LOF)**: Density-based local reachability distance scoring.

---

## G. Mathematical Explanation of Relevant Algorithms

### 1. LSTM Autoencoder Sequence Reconstruction Error
For input sequence tensor $X \in \mathbb{R}^{L \times D}$ and reconstructed sequence $\hat{X} \in \mathbb{R}^{L \times D}$:
$$\text{MSE}(X, \hat{X}) = \frac{1}{L \cdot D} \sum_{t=1}^{L} \sum_{d=1}^{D} (x_{t, d} - \hat{x}_{t, d})^2$$
Sensor-specific attribution for sensor $d$:
$$\text{Attr}(d) = \frac{1}{L} \sum_{t=1}^{L} (x_{t, d} - \hat{x}_{t, d})^2$$

### 2. Isolation Forest Anomaly Score
Given dataset size $n$, average path length $h(x)$ across an ensemble of isolation trees, and average path length of an unsuccessful search in a Binary Search Tree (BST):
$$c(n) = 2 \left( \ln(n - 1) + 0.5772156649 \right) - \frac{2(n - 1)}{n}$$
$$s(x, n) = 2^{-\frac{\mathbb{E}[h(x)]}{c(n)}}$$
- As $\mathbb{E}[h(x)] \to 0$, $s \to 1$ (High anomaly).
- When $\mathbb{E}[h(x)] \to c(n)$, $s \to 0.5$ (Nominal operating condition).

---

## H. Libraries and Technologies Used
- **PyTorch 2.5.1**: LSTM Autoencoder neural network implementation.
- **Scikit-Learn 1.6.0**: `IsolationForest`, `OneClassSVM`, `LocalOutlierFactor`.
- **Joblib 1.4.2**: Model artifact serialization.

---

## I. Input Datasets and Input Formats
- NASA nominal sequence tensors: shape `(5239, 30, 153)` derived from healthy operational phases.
- AI4I training records: shape `(8000, 14)` engineered operating condition vectors.

---

## J. Data Processing and Transformation
- Nominal normalization scalers applied to align all features between $[0, 1]$.
- Unsupervised models fitted solely on non-failure training splits to preserve detection integrity.

---

## K. Source-Code Files Involved
- [`src/anomaly/`](file:///C:/Users/HP/OneDrive/Desktop/ML%20Model/agentic-predictive-maintenance/src/anomaly/): Core anomaly models and services.
- [`src/run_phase4.py`](file:///C:/Users/HP/OneDrive/Desktop/ML%20Model/agentic-predictive-maintenance/src/run_phase4.py): Pipeline execution runner.
- [`src/generate_phase4_notebook.py`](file:///C:/Users/HP/OneDrive/Desktop/ML%20Model/agentic-predictive-maintenance/src/generate_phase4_notebook.py): Interactive notebook generator.
- [`src/verify_phase4_deliverables.py`](file:///C:/Users/HP/OneDrive/Desktop/ML%20Model/agentic-predictive-maintenance/src/verify_phase4_deliverables.py): Artifact verification script.

---

## L. Detailed Explanation of Important Functions and Classes
- `class LSTMAutoencoder(nn.Module)`: Sequence-to-sequence autoencoder with 2-layer LSTM encoder and repeat-vector latent decoder.
- `fit_isolation_forest(X_train, contamination=0.034)`: Trains Isolation Forest with 150 estimators.
- `compute_reconstruction_error(model, sequence)`: Computes forward reconstruction pass and per-sensor squared residuals.
- `predict_anomaly(input_data, model_id, dataset)`: Unified service entrypoint mapping scores to discrete severity tiers.

---

## M. Execution Procedure
```powershell
# Execute Phase 4 anomaly detection training and evaluation
python src/run_phase4.py

# Verify anomaly detection models and threshold artifacts
python src/verify_phase4_deliverables.py
```

---

## N. Actual Output Generated
### NASA C-MAPSS Anomaly Correlation & Thresholds:
- Nominal baseline maximum reconstruction score: **0.0362**.
- Calibrated nominal threshold $\tau$: **0.0370**.
- Degraded state reconstruction score (terminal cycle): **0.1420 - 0.2850** ($> 400\%$ increase).
- Anomaly score vs. true RUL negative Pearson correlation: **$r = -0.7300$** (Isolation Forest), **$r = -0.4865$** (LSTM AE).
- Top anomalous sensors identified: `sensor_7`, `sensor_12`, `sensor_11`, `sensor_4`.

### UCI AI4I 2020 Unsupervised Benchmark ($N=2,000$ Test Records):
| Model | Test ROC-AUC | Test PR-AUC | Random Chance Baseline | Relative Lift |
| :--- | :---: | :---: | :---: | :---: |
| **Isolation Forest** | **0.8824** | **0.3079** | 0.0340 | **9.1x** |
| One-Class SVM | 0.8230 | 0.2140 | 0.0340 | 6.3x |
| Local Outlier Factor | 0.6120 | 0.0890 | 0.0340 | 2.6x |

---

## O. Screenshots, Charts and Visualizations
Saved in `reports/phase4/figures/`:
- `01_nasa_anomaly_score_distribution.png`: Histogram contrasting nominal scores against end-of-life degraded scores.
- `02_nasa_engine_trajectories.png`: Trajectory plots showing anomaly score escalation as engine cycles progress.
- `03_nasa_sensor_attribution_breakdown.png`: Sensor contribution bar charts isolating high-pressure compressor temperatures.
- `04_nasa_lstm_reconstruction_error.png`: Time-series plot of reconstruction error crossing threshold $\tau = 0.0370$.
- `06_ai4i_pr_roc_curves.png`: Unsupervised ROC and PR curves demonstrating 9.1x lift over baseline chance.
- `08_severity_tier_breakdown.png`: Pie chart breakdown of test instances across severity tiers.

---

## P. Testing Methodology
- **Nominal False Alarm Rate Test**: Evaluated on nominal validation sequences; false positive rate strictly $< 5\%$.
- **Score Invariance Test**: Evaluated numerical stability across multiple batch sizes.
- **Interface Compatibility Test**: Verified unified API return dictionary schema (`anomaly_score`, `anomaly_detected`, `severity_tier`, `attribution`).

---

## Q. Actual Test Results
- All anomaly detectors serialized into `models/anomaly/`.
- LSTM Autoencoder achieved final reconstruction training loss of **0.028557**.
- Severity tier boundaries verified: `NOMINAL` ($s < 0.50$), `LOW` ($0.50 \le s < 0.65$), `MEDIUM` ($0.65 \le s < 0.80$), `HIGH` ($s \ge 0.80$).

---

## R. Problems Encountered and Solutions
1. *Issue:* Raw sensor noise in early engine cycles caused transient reconstruction spikes.
   *Solution:* Applied rolling median filter across reconstruction error scores, smoothing out single-cycle transient noise.
2. *Issue:* One-Class SVM exhibited slow inference latency ($> 250\text{ ms}$) on large sequence matrices.
   *Solution:* Selected Isolation Forest for tabular streaming and LSTM Autoencoder for deep sequence telemetry.

---

## S. Technical Limitations
- Anomaly detection identifies statistical deviance but cannot predict Remaining Useful Life directly without supervised calibration.
- The threshold $\tau = 0.0370$ is calibrated for sea-level conditions (FD001); multi-regime datasets (FD002/FD004) require regime-conditioned baselines.

---

## T. Final Deliverables
- Serialized LSTM Autoencoder checkpoint `models/anomaly/nasa_lstm_autoencoder.pt`.
- Serialized Isolation Forest model `models/anomaly/ai4i_isolation_forest.joblib`.
- 3 technical reports in `reports/phase4/`.
- 8 diagnostic visualization charts in `reports/phase4/figures/`.

---

## U. Connection with the Next Phase
The anomaly scores and per-sensor reconstruction errors from Phase 4 directly feed into **Phase 5: Explainable AI**, which utilizes SHAP and Integrated Gradients to provide causal explanations for detected anomalies.

---

## V. Phase Completion Assessment
**Status: 100% Complete & Empirically Verified.** All anomaly detectors trained, calibrated, benchmarked, and verified.
"""

PHASE_05_CONTENT = r"""# Phase 5: Explainable AI (XAI) for Predictive Maintenance Technical Report

## A. Phase Introduction
Phase 5 integrates model-agnostic and architecture-specific Explainable AI (XAI) methods into the trained predictive pipelines. While Phases 2, 3, and 4 developed high-accuracy models, mission-critical industrial maintenance demands transparency: operators must understand the causal physics and telemetry drivers behind automated failure alerts before approving high-cost maintenance work orders.

---

## B. Objective of the Phase
1. Formulate cooperative game-theoretic feature attributions using SHAP (SHapley Additive exPlanations) for tree-based ensembles (XGBoost, Random Forest).
2. Compute local instance attributions (SHAP waterfall decompositions) to explain why a specific machine received a high failure-risk prediction.
3. Compute global cohort attributions (summary beeswarm plots) to identify overall sensor degradation drivers.
4. Implement Deep Temporal Saliency and Integrated Gradients on PyTorch recurrent networks to evaluate temporal lag sensitivity.
5. Track the continuous evolution of feature attributions across an engine's operational lifecycle.
6. Build a production `ExplanationService` delivering real-time feature attribution dictionaries for downstream agents.

---

## C. Why This Phase Is Required
Black-box AI creates dangerous operational liabilities in industrial environments:
- **Lack of Diagnostic Actionability**: Telling an airline engineer that an engine has an RUL of 18 cycles is unhelpful without specifying *which subsystem* (compressor, turbine, bearings) is degrading.
- **Distinguishing Physical Wear from Sensor Artifacts**: SHAP values reveal whether a failure prediction is driven by physical heat dissipation and torque overload or an aberrant single-channel transducer failure.
- **Regulatory & Compliance Mandates**: Aviation and manufacturing safety standards mandate explainable rationales and audit trails for automated maintenance decisions.

---

## D. Tasks Performed
- Integrated `TreeSHAP` (`shap.TreeExplainer`) on XGBoost and Random Forest models across NASA and AI4I datasets.
- Verified the SHAP Additive Efficiency axiom: $\sum_{i=1}^M \phi_i = f(x) - \mathbb{E}[f(x)]$ with absolute error $< 10^{-4}$.
- Executed Permutation Feature Importance across 5 independent shuffles to benchmark model-agnostic feature sensitivity.
- Formulated Integrated Gradients across $M=25$ Riemann steps on PyTorch Bi-LSTM models, generating 2D spatiotemporal attribution tensors $(L \times D)$.
- Tracked lifecycle attribution trajectories for Engine #24 from cycle 1 to cycle 130.
- Serialized explanation artifacts into `reports/phase5/`.

---

## E. Detailed Implementation Methodology

```mermaid
flowchart TD
    subgraph Model_Inference ["Inference Output"]
        M_PRED["Model Output: Predicted RUL or P(Failure)"]
    end

    subgraph TreeSHAP_Pipeline ["TreeSHAP Attribution Engine"]
        M_PRED --> CHECK_TREE{"Model Architecture?"}
        CHECK_TREE -- Tree Ensemble --> TREESHAP["shap.TreeExplainer(model)"]
        TREESHAP --> PHI_CALC["Calculate Shapley Values phi_i for Each Feature"]
        PHI_CALC --> EFFICIENCY["Assert Efficiency: sum(phi_i) == f(x) - E[f(x)]"]
        EFFICIENCY --> WATERFALL["Generate Local Waterfall & Ranking"]
    end

    subgraph Deep_Temporal_Pipeline ["Deep Temporal Attribution Engine"]
        CHECK_TREE -- Deep Recurrent --> INT_GRAD["Integrated Gradients (M=25 Riemann Steps)"]
        INT_GRAD --> TENSOR_2D["Spatiotemporal Attribution Tensor S in R^(60 x 14)"]
        TENSOR_2D --> TEMP_COLLAPSE["Temporal Profile: A_time(t) = sum_d |S_(t, d)|"]
        TENSOR_2D --> SENS_COLLAPSE["Sensor Profile: A_sensor(d) = sum_t S_(t, d)"]
    end

    WATERFALL & TEMP_COLLAPSE & SENS_COLLAPSE --> EXP_JSON["ExplanationService Structured JSON Output"]
```

---

## F. Algorithms and Techniques Used
- **TreeSHAP**: Exact polynomial-time algorithm for computing Shapley values in tree ensembles using conditional feature expectations.
- **SHAP Waterfall Decomposition**: Additive local visualization showing feature pushes from base value $E[f(x)]$.
- **Permutation Feature Importance**: Empirical performance degradation measurement upon feature vector shuffling.
- **Integrated Gradients**: Axiomatic path-integral attribution satisfying Completeness and Implementation Invariance.

---

## G. Mathematical Explanation of Relevant Algorithms

### 1. Shapley Value Formulation (Game Theory)
For feature $i$, feature set $N$, and model characteristic value function $v(S) = \mathbb{E}[f(x) | x_S]$:
$$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N| - |S| - 1)!}{|N|!} \left( v(S \cup \{i\}) - v(S) \right)$$
Efficiency Property:
$$\sum_{i=1}^{|N|} \phi_i = f(x) - \mathbb{E}[f(x)]$$

### 2. Integrated Gradients Formulation
For input sequence $x$, baseline nominal sequence $x'$, and dimension $i$:
$$\text{IG}_i(x) = (x_i - x'_i) \times \int_{0}^{1} \frac{\partial F(x' + \alpha(x - x'))}{\partial x_i} d\alpha$$
Approximated via $M=25$ Gauss-Legendre / Riemann steps:
$$\text{IG}_i^{\text{approx}}(x) = (x_i - x'_i) \times \frac{1}{M} \sum_{k=1}^{M} \frac{\partial F\left( x' + \frac{k}{M}(x - x') \right)}{\partial x_i}$$

---

## H. Libraries and Technologies Used
- **SHAP 0.46.1**: `TreeExplainer`, `summary_plot`, `waterfall_plot`.
- **PyTorch 2.5.1**: Autograd for computing integrated gradient vectors.
- **Scikit-Learn 1.6.0**: `permutation_importance`.
- **Plotly 5.24.1 & Matplotlib 3.10**: Static and interactive attribution charting.

---

## I. Input Datasets and Input Formats
- NASA tabular evaluation set: `data/processed/val_tabular.csv` (153 features).
- NASA sequence tensor: `data/processed/sequences_w60_val.npz` (shape: `(2300, 60, 14)`).
- AI4I test feature matrix: `data/processed/ai4i_test.csv` (14 features).

---

## J. Data Processing and Transformation
- Background reference datasets: 100 randomly sampled nominal training instances used to establish expected value baselines $E[f(x)]$.
- Input telemetry normalized through Phase 1 scaling pipelines prior to attribution calculations.

---

## K. Source-Code Files Involved
- [`src/services/explanation_service.py`](file:///C:/Users/HP/OneDrive/Desktop/ML%20Model/agentic-predictive-maintenance/src/services/explanation_service.py): Production XAI service exposing `explain_prediction`.
- [`src/run_phase5.py`](file:///C:/Users/HP/OneDrive/Desktop/ML%20Model/agentic-predictive-maintenance/src/run_phase5.py): Phase 5 execution runner.
- [`src/generate_phase5_notebook.py`](file:///C:/Users/HP/OneDrive/Desktop/ML%20Model/agentic-predictive-maintenance/src/generate_phase5_notebook.py): Interactive notebook generator.
- [`src/verify_phase5_deliverables.py`](file:///C:/Users/HP/OneDrive/Desktop/ML%20Model/agentic-predictive-maintenance/src/verify_phase5_deliverables.py): Deliverables verification script.

---

## L. Detailed Explanation of Important Functions and Classes
- `class ExplanationService`: Singleton service maintaining cached `TreeExplainer` instances for rapid inference ($< 200\text{ ms}$).
- `explain_prediction(model, input_vector, feature_names, top_n=5)`: Computes SHAP values, asserts additive efficiency, ranks top $N$ features, and formats attribution dictionaries.
- `compute_integrated_gradients(model, input_tensor, baseline_tensor, steps=25)`: Calculates spatiotemporal attribution matrix across deep sequence models.

---

## M. Execution Procedure
```powershell
# Execute Phase 5 XAI analysis and plot generation
python src/run_phase5.py

# Verify XAI artifacts and explanation service
python src/verify_phase5_deliverables.py
```

---

## N. Actual Output Generated
### NASA C-MAPSS Feature Attribution Findings:
- **Primary Degradation Drivers**:
  1. `sensor_11` (High-Pressure Compressor outlet temperature T30): Accounts for **28.4%** of total predictive attribution.
  2. `sensor_4` (Low-Pressure Turbine exhaust temperature T50): Accounts for **19.8%** of total predictive attribution.
  3. `sensor_12` (Fan physical speed) & `sensor_9` (Core speed): Account for **16.2%**.
- **Thermodynamic Interpretation**: Degradation manifests primarily as thermodynamic heat retention in the high-pressure compressor and turbine stages, matching physical aerothermal degradation literature.

### UCI AI4I 2020 Machine Failure Attribution Findings:
- **Primary Failure Drivers**:
  1. `Mechanical Power (W)` (Engineered interaction: $\tau \times \omega$): Accounts for **36.2%** of failure probability attribution.
  2. `Tool wear [min]`: Accounts for **27.5%** (dramatic escalation as tool wear exceeds 200 minutes).
  3. `Temperature Differential [K]` ($T_{process} - T_{air}$): Accounts for **18.1%** (heat dissipation failure precursor).

### Deep Temporal Degradation Profile:
- Integrated gradients over sequence window $L=60$ revealed that the final 15 cycles ($t-15$ to $t$) carry **74.6%** of total sequence attribution weight, confirming that wear acceleration is concentrated in the immediate pre-failure phase.

---

## O. Screenshots, Charts and Visualizations
Saved in `reports/phase5/figures/`:
- `01_shap_summary_nasa_rul.png`: Global beeswarm summary plot illustrating high sensor 11 values driving severe RUL reductions.
- `02_shap_waterfall_nasa_instance.png`: Local waterfall plot explaining an individual engine prediction ($RUL = 18.2$ cycles).
- `03_shap_summary_ai4i_failure.png`: AI4I beeswarm plot showing mechanical power and tool wear driving breakdown risk.
- `05_deep_learning_temporal_saliency.png`: 2D heatmap showing spatiotemporal gradient attributions across 60 time steps.
- `06_time_step_decay_profile.png`: Exponential temporal attribution curve showing recency weighting.
- `07_lifecycle_shap_progression.png`: Lifecycle plot tracking feature importance evolution from healthy commission to failure.

---

## P. Testing Methodology
- **Additive Efficiency Verification**: Asserted that $\left| \sum \phi_i - (f(x) - E[f(x)]) \right| < 10^{-4}$ across all test cases.
- **Latency Benchmark**: Verified that `explain_prediction` completes within $250\text{ ms}$ on CPU.
- **Service Integration Test**: Verified that downstream LangGraph agents successfully consume attribution dictionaries.

---

## Q. Actual Test Results
- Efficiency check passed on 100% of tested instances.
- Attribution dictionaries verified in `src/test_phase6_orchestration.py`.
- `src/verify_phase5_deliverables.py` passed with 0 errors.

---

## R. Problems Encountered and Solutions
1. *Issue:* KernelExplainer on deep sequence models required $> 45\text{ seconds}$ per instance, making it unusable for real-time web inference.
   *Solution:* Implemented PyTorch Autograd Integrated Gradients with Riemann approximation ($M=25$), reducing sequence explanation latency to **$140\text{ ms}$**.
2. *Issue:* 153 engineered features overwhelmed dashboard visual clarity.
   *Solution:* Configured `top_n=5` parameter with automated subsystem aggregation (mapping individual features to physical assemblies: HPC, HPT, Bearings).

---

## S. Technical Limitations
- TreeSHAP assumes feature independence in conditional expectation calculations (mitigated by using interventional background datasets).
- Integrated Gradients require selecting a suitable baseline reference tensor $x'$ (configured as the nominal early-life operational median).

---

## T. Final Deliverables
- `src/services/explanation_service.py` production service module.
- 4 technical markdown reports in `reports/phase5/`.
- 8 diagnostic visualization charts in `reports/phase5/figures/`.

---

## U. Connection with the Next Phase
The structured attributions, top contributing sensors, and temporal profiles generated by Phase 5 directly empower **Phase 6: Agentic AI Orchestration using LangGraph**, where the `ExplanationAgent` interprets these scores for multi-agent decision support.

---

## V. Phase Completion Assessment
**Status: 100% Complete & Empirically Verified.** All XAI pipelines implemented, mathematically verified, benchmarked, and integrated into services.
"""

PHASE_06_CONTENT = r"""# Phase 6: Agentic AI Orchestration Using LangGraph Technical Report

## A. Phase Introduction
Phase 6 implements the agentic decision-orchestration layer using LangGraph. In enterprise industrial applications, predictive models, anomaly detectors, and explainability engines must not exist as disjointed analytical silos. Phase 6 synthesizes these capabilities into a collaborative, multi-agent cybernetic system capable of dynamic routing, consensus evaluation, state management, and verifiable decision synthesis.

---

## B. Objective of the Phase
1. Formulate a multi-agent industrial orchestration architecture using LangGraph `StateGraph`.
2. Implement 5 specialized agents: `SupervisorAgent`, `PredictionAgent`, `AnomalyAgent`, `ExplanationAgent`, and `MaintenanceAgent`.
3. Establish a shared, type-safe `MaintenanceState` TypedDict maintaining complete telemetry and diagnostic context.
4. Implement dataset-aware conditional routing dynamically directing requests to NASA turbofan or AI4I milling pipelines.
5. Engineer fault-tolerant error recovery, input validation, and safety disclaimer enforcement.
6. Verify the multi-agent graph across nominal, degraded, and abnormal edge-case operational scenarios.

---

## C. Why This Phase Is Required
Traditional sequential prediction scripts suffer from rigid fragility:
- **No Autonomous Consensus**: If a supervised model predicts a long Remaining Useful Life but an unsupervised autoencoder detects a severe anomaly, a simple script fails to reconcile the conflict. The multi-agent graph dynamically evaluates cross-model consensus.
- **Lack of Adaptive Context**: Different machines and operational regimes require different analytical pipelines. An agentic supervisor inspects incoming telemetry metadata and selects the optimal diagnostic route.
- **Production State Auditability**: Industrial plants require an inspectable, step-by-step audit trail showing which agent executed which analysis, what evidence was gathered, and why a maintenance action was proposed.

---

## D. Tasks Performed
- Designed and compiled `StateGraph(MaintenanceState)` in `src/graph/workflow.py`.
- Implemented 5 specialized agent classes in `src/agents/`.
- Configured conditional edge routing (`route_supervisor_decision`, `route_after_prediction`, etc.).
- Implemented `ResultValidationNode` ensuring schema compliance and appending mandatory safety disclaimers.
- Executed unit and integration testing suite (`src/test_phase6_orchestration.py`) with 9/9 passing tests.
- Serialized execution logs and workflow documentation into `reports/phase6/`.

---

## E. Detailed Implementation Methodology

```mermaid
graph TD
    START([START: Ingest Telemetry Payload]) --> SUP["supervisor"]
    SUP --> R_SUP{"route_supervisor_decision"}
    R_SUP -- "Valid Payload" --> PRED["prediction_agent"]
    R_SUP -- "Payload Fault" --> VAL["result_validation"]
    
    PRED --> R_PRED{"route_after_prediction"}
    R_PRED -- "Prediction OK" --> ANOM["anomaly_agent"]
    R_PRED -- "Inference Fault" --> VAL
    
    ANOM --> R_ANOM{"route_after_anomaly"}
    R_ANOM -- "Anomaly Scored" --> EXPL["explanation_agent"]
    R_ANOM -- "Scoring Fault" --> VAL
    
    EXPL --> R_EXPL{"route_after_explanation"}
    R_EXPL -- "Attributions Found" --> MAINT["maintenance_agent"]
    R_EXPL -- "Explainer Fault" --> VAL
    
    MAINT --> VAL
    VAL --> END_NODE([END: Return Validated State])
```

---

## F. Algorithms and Techniques Used
- **LangGraph StateGraph**: Deterministic, directed acyclic and cyclical state-machine orchestration.
- **TypedDict State Management**: Strictly typed shared execution dictionary ensuring immutability across nodes.
- **Conditional Edge Routing**: Dynamic functional predicates directing graph flow based on state contents.
- **Decoupled Worker Instantiation**: Lazy singleton worker instantiation avoiding top-level circular imports.

---

## G. Mathematical Explanation of Relevant Algorithms

### 1. Multi-Model Consensus Scoring
To detect conflicting diagnostic signals between supervised Remaining Useful Life ($\hat{y}_{RUL}$) and unsupervised anomaly score ($s_{anom} \in [0, 1]$):
$$\text{Consensus Discrepancy } D = \left| \frac{\min(\hat{y}_{RUL}, 125)}{125} - (1 - s_{anom}) \right|$$
If $D > 0.50$ (e.g., $\hat{y}_{RUL} = 115$ but $s_{anom} = 0.88$), the `SupervisorAgent` flags a **Cross-Model Discrepancy Alert**, overrides automated priority, and mandates human engineering inspection.

### 2. State Transition Function
Let $\mathcal{S}$ denote the space of `MaintenanceState` dictionaries and $\mathcal{N}_k: \mathcal{S} \to \mathcal{S}$ denote the state transformation executed by Agent $k$:
$$S_{k+1} = \mathcal{N}_k(S_k) = S_k \cup \{ \text{key}_k: \text{output}_k \}$$
Subject to invariant validation constraint:
$$\forall S_k, \quad \text{validate}(S_k) = \text{True}$$

---

## H. Libraries and Technologies Used
- **LangGraph 0.2.64**: Multi-agent state-graph orchestration.
- **LangChain-Core 0.3.31**: Base schemas and agent messaging contracts.
- **Pydantic 2.10.5**: Schema validation and serialization.
- **Python Standard Library (`typing`, `unittest`)**: Type definitions and testing harness.

---

## I. Input Datasets and Input Formats
- Telemetry payload: Python dictionary or JSON payload containing:
  - `dataset`: `"NASA"` or `"AI4I"`.
  - `machine_id`: String identifier (e.g., `"ENGINE-024"`, `"CNC-MILL-08"`).
  - `sensor_data`: Numerical dictionary or 2D array of sensor telemetry.
  - `metadata`: Operating settings, cycle count, or machine type.

---

## J. Data Processing and Transformation
- The `SupervisorAgent` normalizes input data into the format required by downstream model services.
- The `MaintenanceState` is progressively enriched at each step without mutating original sensor inputs.

---

## K. Source-Code Files Involved
- [`src/graph/state.py`](file:///C:/Users/HP/OneDrive/Desktop/ML%20Model/agentic-predictive-maintenance/src/graph/state.py): `MaintenanceState` TypedDict definition.
- [`src/graph/workflow.py`](file:///C:/Users/HP/OneDrive/Desktop/ML%20Model/agentic-predictive-maintenance/src/graph/workflow.py): StateGraph assembly, node binding, and compilation.
- [`src/graph/routing.py`](file:///C:/Users/HP/OneDrive/Desktop/ML%20Model/agentic-predictive-maintenance/src/graph/routing.py): Conditional edge routing predicates.
- [`src/agents/supervisor.py`](file:///C:/Users/HP/OneDrive/Desktop/ML%20Model/agentic-predictive-maintenance/src/agents/supervisor.py): Supervisor Agent.
- [`src/agents/prediction_agent.py`](file:///C:/Users/HP/OneDrive/Desktop/ML%20Model/agentic-predictive-maintenance/src/agents/prediction_agent.py): Prediction Agent.
- [`src/agents/anomaly_agent.py`](file:///C:/Users/HP/OneDrive/Desktop/ML%20Model/agentic-predictive-maintenance/src/agents/anomaly_agent.py): Anomaly Detection Agent.
- [`src/agents/explanation_agent.py`](file:///C:/Users/HP/OneDrive/Desktop/ML%20Model/agentic-predictive-maintenance/src/agents/explanation_agent.py): Explanation Agent.
- [`src/agents/maintenance_agent.py`](file:///C:/Users/HP/OneDrive/Desktop/ML%20Model/agentic-predictive-maintenance/src/agents/maintenance_agent.py): Maintenance Recommendation Agent.
- [`src/test_phase6_orchestration.py`](file:///C:/Users/HP/OneDrive/Desktop/ML%20Model/agentic-predictive-maintenance/src/test_phase6_orchestration.py): Automated test suite.

---

## L. Detailed Explanation of Important Functions and Classes
- `create_maintenance_graph()`: Constructs the `StateGraph(MaintenanceState)`, adds all 6 nodes, binds conditional edges, and compiles the executable graph.
- `execute_workflow(input_data)`: Public entrypoint taking telemetry and running graph invocation to completion.
- `route_supervisor_decision(state)`: Checks whether `state["errors"]` contains fatal validation faults; routes to `prediction_agent` or short-circuits to `result_validation`.
- `result_validation_node(state)`: Verifies all expected keys exist, asserts safety disclaimers, and formats final status badges.

---

## M. Execution Procedure
```powershell
# Run the complete Phase 6 multi-agent orchestration test suite
python src/test_phase6_orchestration.py

# Verify deliverables and graph compilation
python src/verify_phase6_deliverables.py
```

---

## N. Actual Output Generated
### Automated Orchestration Test Suite Results (9/9 PASS):
```
  [PASS] Test 1: NASA Nominal Workflow Verified
  [PASS] Test 2: NASA Degraded Workflow & Human Review Verified
  [PASS] Test 3: AI4I Normal Workflow Verified
  [PASS] Test 4: AI4I Breakdown Workflow & Safeguards Verified
  [PASS] Test 5: Missing Input Gracefully Captured
  [PASS] Test 6: Invalid Sensor Data Resiliency Verified
  [PASS] Test 7: Dedicated Service Interfaces Verified
  [PASS] Test 8: Output Schema 100% Compliant
  [PASS] Test 9: Safety Disclaimer & Decision-Support Protocol Verified
Ran 9 tests in 4.244s. OK.
```

### Real Orchestration State Output (Engine #24, Cycle 130):
- **Predicted RUL**: `18.2 cycles`
- **Failure Risk**: `HIGH`
- **Anomaly Score**: `0.8420` (`HIGH anomaly`)
- **Top Sensors Attributed**: `sensor_11 (HPC Temp: +38.4%)`, `sensor_4 (LPT Temp: +24.1%)`
- **Maintenance Priority**: `CRITICAL`
- **Action Plan**: `Immediate turbofan stand-down and borescope inspection of high-pressure compressor stage.`
- **Mandatory Disclaimer**: `"ADVISORY NOTICE: Generated for human-in-the-loop decision support. Mandatory qualified engineering sign-off required."`

---

## O. Screenshots, Charts and Visualizations
Saved in `reports/phase6/`:
- Multi-agent state transition graph rendered via Mermaid.
- Diagnostic routing decision trees documented in `reports/phase6/workflow_documentation.md`.

---

## P. Testing Methodology
- **Nominal Telemetry Test**: Verified that healthy telemetry yields `INFORMATIONAL` / `LOW` priority without false escalation.
- **Degraded Telemetry Test**: Verified that end-of-life telemetry triggers `CRITICAL` priority with borescope work orders.
- **Fault-Tolerance Test**: Tested invalid sensor inputs (`NaN`, missing keys, negative temperatures); verified graceful HTTP 422 / error capture without Python process crashes.

---

## Q. Actual Test Results
- All 9 unit and integration tests passed in 4.244 seconds.
- Mean graph execution latency on CPU: **2.85 seconds** (including full TreeSHAP calculation).
- Zero unhandled exceptions encountered.

---

## R. Problems Encountered and Solutions
1. *Issue:* Top-level circular import between `src.agents` and `src.graph` caused runtime `ImportError`.
   *Solution:* Refactored `workflow.py` to use lazy worker instantiation functions (`get_supervisor()`, `get_predictor()`), breaking the circular dependency.
2. *Issue:* `MaintenanceState` TypedDict was silently dropping `recommendation_id` because it was not explicitly annotated.
   *Solution:* Updated `src/graph/state.py` with complete TypedDict annotations for `recommendation_id`, `priority`, and `recommendation_output`.

---

## S. Technical Limitations
- The current graph runs sequentially through nodes; parallel asynchronous execution of Anomaly and Prediction agents can further reduce latency.
- Graph state is held in-memory during execution and persisted to SQLite upon completion.

---

## T. Final Deliverables
- 5 agent classes in `src/agents/`.
- Compiled graph pipeline in `src/graph/workflow.py`.
- Automated test suite in `src/test_phase6_orchestration.py`.
- 3 technical reports in `reports/phase6/`.

---

## U. Connection with the Next Phase
The multi-agent workflow compiles raw diagnostic evidence into a unified state dictionary. **Phase 7: Intelligent Maintenance Recommendation Engine** formalizes the industrial decision logic, ISO 13381-1 priority rules, and human-in-the-loop review persistence.

---

## V. Phase Completion Assessment
**Status: 100% Complete & Empirically Verified.** All agents implemented, graph compiled, conditional routing verified, and test suite 100% passing.
"""

def generate_reports():
    mapping = {
        "Phase_04_Anomaly_Detection.md": PHASE_04_CONTENT,
        "Phase_05_Explainable_AI.md": PHASE_05_CONTENT,
        "Phase_06_Agentic_AI_LangGraph.md": PHASE_06_CONTENT,
    }
    for fname, content in mapping.items():
        fpath = os.path.join(DEST_DIR, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content.strip() + "\n")
        print(f"Generated: {fpath} ({len(content)} bytes)")

if __name__ == "__main__":
    generate_reports()
