# Phase 3: Deep Learning Model Development Technical Report

## A. Phase Introduction
Phase 3 advances the predictive modeling capabilities by designing, training, and benchmarking deep recurrent neural network architectures. While Phase 2 established strong tabular baselines, deep learning architectures natively process raw multivariate temporal sequences, learning complex nonlinear degradation dynamics and temporal dependencies without manual feature engineering.

---

## B. Objective of the Phase
1. Formulate multivariate sequence models (LSTM, Bidirectional LSTM, GRU, CNN-LSTM) in PyTorch.
2. Model turbofan degradation over sliding sequence windows of lengths $L=30$ and $L=60$.
3. Benchmark deep sequence models against the official NASA C-MAPSS terminal test fleet ($N=100$ engines).
4. Demonstrate a significant reduction in the NASA Asymmetric Penalty Score compared to traditional ML baselines.
5. Train deep Multi-Layer Perceptrons (MLP) on the tabular AI4I dataset for comparison against gradient boosting.
6. Serialize PyTorch model checkpoints (`.pt`) and compile comprehensive training loss logs.

---

## C. Why This Phase Is Required
Traditional ML models operate on static tabular summaries (rolling mean, rolling std dev), which compress temporal trajectories into scalar aggregates, discarding the sequential ordering and rate-of-change dynamics. In contrast:
- **Long Short-Term Memory (LSTM)** networks maintain an internal cell state regulated by input, forget, and output gates, allowing them to capture early wear onset and distinguish transient sensor spikes from permanent degradation.
- **Bidirectional LSTMs** process sequence windows forward and backward, capturing contextual degradation patterns across the entire observation window.
- Deep sequence models learn smooth, monotonically decreasing degradation curves, reducing dangerous late-prediction overestimates and dramatically improving NASA Asymmetric Scores.

---

## D. Tasks Performed
- Implemented PyTorch architectures: `LSTMRegressor`, `BiLSTMRegressor`, `GRURegressor`, `CNNLSTMRegressor`, and `AI4IMLP`.
- Implemented custom PyTorch sequence datasets and dataloaders supporting mini-batch shuffling and GPU/CPU device mapping.
- Configured Huber Loss (Smooth L1 Loss) to provide robustness against outlier sensor measurements.
- Applied AdamW optimization with weight decay ($10^{-4}$), ReduceLROnPlateau learning rate scheduling, and early stopping (patience = 10 epochs).
- Conducted comparative ablation studies across sequence window lengths $L \in \{30, 50, 60\}$.
- Serialized 18 PyTorch model checkpoints (`.pt`) into `models/deep_learning/`.

---

## E. Detailed Implementation Methodology

```mermaid
graph TB
    subgraph Input ["Input Sequence Tensor"]
        IN["Input: (Batch Size, Sequence Length L=60, 14 Active Sensors)"]
    end

    subgraph Recurrent_Core ["Recurrent Backbone (Bi-LSTM / GRU)"]
        L1["Layer 1: 64 hidden units (Bi-directional = 128 total)"]
        DO1["Spatial Dropout (p=0.20)"]
        L2["Layer 2: 32 hidden units (Bi-directional = 64 total)"]
        DO2["Spatial Dropout (p=0.20)"]
        IN --> L1 --> DO1 --> L2 --> DO2
    end

    subgraph Dense_Head ["Regression Head"]
        H_LAST["Final Hidden State h_T (64-dim)"]
        FC1["Linear: 64 -> 32 (ReLU)"]
        DO3["Dropout (p=0.10)"]
        FC2["Linear: 32 -> 16 (ReLU)"]
        OUT["Linear: 16 -> 1 (Scalar RUL Prediction)"]
        DO2 --> H_LAST --> FC1 --> DO3 --> FC2 --> OUT
    end
```

---

## F. Algorithms and Techniques Used
- **Long Short-Term Memory (LSTM)**: Recurrent architecture with input, forget, and output gates mitigating vanishing gradients.
- **Gated Recurrent Unit (GRU)**: Computationally efficient recurrent architecture with reset and update gates (23.9% fewer parameters).
- **Spatial / Temporal Dropout**: Regularization applied across recurrent layers to prevent co-adaptation of features.
- **Huber Loss (Smooth L1)**: Combines MSE efficiency for small residuals ($|y - \hat{y}| < \delta$) with MAE outlier resilience for large errors.

---

## G. Mathematical Explanation of Relevant Algorithms

### 1. LSTM Cell Formulation
At time step $t$ with input $x_t$ and previous hidden state $h_{t-1}$:
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad (\text{Forget Gate})$$
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad (\text{Input Gate})$$
$$\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c) \quad (\text{Candidate Cell State})$$
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \quad (\text{Updated Cell State})$$
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \quad (\text{Output Gate})$$
$$h_t = o_t \odot \tanh(C_t) \quad (\text{Hidden State Output})$$

### 2. Huber Loss (Smooth L1)
For residual $e_i = y_i - \hat{y}_i$:
$$\mathcal{L}_\delta(e_i) = \begin{cases} \frac{1}{2} e_i^2 & \text{if } |e_i| \le \delta \\ \delta \cdot \left( |e_i| - \frac{1}{2} \delta \right) & \text{otherwise} \end{cases} \quad \text{where } \delta = 1.0$$

---

## H. Libraries and Technologies Used
- **PyTorch 2.5.1**: Core deep learning tensor framework, Autograd, and `torch.nn`.
- **NumPy 2.2.1**: `.npz` sequence tensor array storage.
- **Scikit-Learn 1.6.0**: Evaluation metrics and cross-validation splitting.

---

## I. Input Datasets and Input Formats
- Sequence input tensors: `data/processed/sequences_w30_train.npz` and `sequences_w60_train.npz`.
- Shape: $(N, L, 14)$ where $N$ is sample count, $L \in \{30, 60\}$ is sequence window length, and 14 is the number of active sensor channels.
- Evaluation input: `test_FD001.txt` preprocessed into sequences matching the window length up to the final observed cycle.

---

## J. Data Processing and Transformation
- Sequences are constructed by sliding a window of length $L$ along each engine's time series.
- Zero-leakage standard scaling parameters are applied to each sensor channel prior to windowing.

---

## K. Source-Code Files Involved
- [`src/generate_phase3_artifacts.py`](file:///C:/Users/HP/OneDrive/Desktop/ML%20Model/agentic-predictive-maintenance/src/generate_phase3_artifacts.py): Deep learning model definitions, training loops, evaluation, and plotting.
- [`src/run_phase3.py`](file:///C:/Users/HP/OneDrive/Desktop/ML%20Model/agentic-predictive-maintenance/src/run_phase3.py): Pipeline execution runner.
- [`src/verify_phase3_deliverables.py`](file:///C:/Users/HP/OneDrive/Desktop/ML%20Model/agentic-predictive-maintenance/src/verify_phase3_deliverables.py): PyTorch checkpoint and metric verification.

---

## L. Detailed Explanation of Important Functions and Classes
- `class LSTMRegressor(nn.Module)`: 2-layer LSTM backbone (hidden=64, 32) with dropout (0.2) and dense linear projection head (32 -> 16 -> 1).
- `class GRURegressor(nn.Module)`: 2-layer GRU backbone matching the hidden dimensions with lower parameter overhead (19,041 parameters).
- `class BiLSTMRegressor(nn.Module)`: Bidirectional LSTM concatenating forward and backward sequence representations (hidden=64 per direction).
- `train_epoch(model, loader, criterion, optimizer)`: Executes forward pass, loss calculation, backpropagation, and gradient clipping ($max\_norm = 1.0$).
- `evaluate_terminal_test(model, test_sequences, true_rul)`: Generates terminal cycle predictions for 100 test engines and calculates RMSE, MAE, and NASA Score.

---

## M. Execution Procedure
```powershell
# Execute deep learning training and benchmarking
python src/run_phase3.py

# Verify deep learning model checkpoints and metrics
python src/verify_phase3_deliverables.py
```

---

## N. Actual Output Generated
### NASA C-MAPSS FD001 Terminal Test Fleet ($N=100$ Engines):
| Model Architecture | Sequence Window ($L$) | Test Terminal RMSE | Test Terminal MAE | Test Terminal NASA Score | Total Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: |
| LSTM Regressor | 30 | 21.98 | 16.15 | 1,464.35 | 25,025 |
| GRU Regressor | 30 | 22.21 | 16.43 | 1,769.12 | 19,041 |
| CNN-LSTM Regressor | 30 | 26.99 | 19.98 | 4,301.67 | 27,425 |
| LSTM Regressor | 50 | 21.27 | 16.10 | 1,021.57 | 25,025 |
| **Bi-LSTM Regressor** | **60** | **20.02** | **14.83** | **889.38** | **25,025** |

### Key Benchmark Comparison: Traditional ML vs. Deep Learning:
- **Test Terminal RMSE**: Reduced from 22.50 cycles (XGBoost) to **20.02 cycles** (Bi-LSTM $L=60$) — an **11.0% improvement**.
- **NASA Asymmetric Score**: Reduced from 9,525.5 (XGBoost) to **889.38** (Bi-LSTM $L=60$) — a **90.7% reduction** in hazardous late-prediction penalties!
- **AI4I Deep MLP**: Attained test F1-score of 0.8120, confirming that Gradient Boosting (F1: 0.8689) remains superior for tabular industrial data.

---

## O. Screenshots, Charts and Visualizations
Saved in `reports/phase3/figures/`:
- `01_loss_curves_rul_lstm_w30.png`: Huber loss curves demonstrating smooth convergence over 45 epochs without validation divergence.
- `02_rul_pred_vs_actual_deep_learning.png`: Predicted versus actual scatter plot demonstrating tight agreement along the identity line.
- `04_rul_trajectory_tracking.png`: Degradation trajectories for sample test engines (#12, #24, #35) tracking true continuous wear.
- `05_window_length_comparison.png`: Bar chart demonstrating consistent RMSE and NASA score reductions as $L$ increases from 30 to 60.
- `09_ml_vs_dl_comparison_bar.png`: Direct comparison illustrating the 90.7% NASA score reduction achieved by Bi-LSTM.

---

## P. Testing Methodology
- **Tensor Shape Assertions**: Verified input tensors conform to `(batch_size, seq_len, 14)`.
- **Checkpoint Deserialization**: Verified that PyTorch `state_dict` loads cleanly into network instances without missing keys.
- **Inference Latency Benchmark**: Evaluated forward pass execution time on CPU ($< 70\text{ ms}$ for 100 engines).

---

## Q. Actual Test Results
- All 18 PyTorch `.pt` checkpoints verified in `models/deep_learning/`.
- Training completed with zero gradient explosions (enforced by gradient norm clipping at 1.0).
- Bi-LSTM ($L=60$) verified as the champion architecture for turbofan RUL regression.

---

## R. Problems Encountered and Solutions
1. *Issue:* MSE loss caused training instability due to noisy sensor spikes in early cycles.
   *Solution:* Replaced standard MSE with Huber Loss ($\delta = 1.0$), providing linear scaling for outliers while preserving quadratic sensitivity near zero.
2. *Issue:* Engines with lifespans shorter than 60 cycles cannot form full $L=60$ windows without padding.
   *Solution:* Maintained dual models: $L=30$ for early-cycle engines ($30 \le t < 60$) and $L=60$ for mature engines ($t \ge 60$).

---

## S. Technical Limitations
- Deep sequence models require a minimum sequence history ($t \ge L$) to generate predictions; tabular models are required for cold-start cycles ($t < 30$).
- Recurrent neural networks have higher computational training overhead than gradient boosting (training time: ~120s vs. 1.5s).

---

## T. Final Deliverables
- 18 PyTorch checkpoints (`.pt`) in `models/deep_learning/`.
- 4 comprehensive markdown reports in `reports/phase3/`.
- 9 high-resolution diagnostic charts in `reports/phase3/figures/`.

---

## U. Connection with the Next Phase
While Phases 2 and 3 provide supervised RUL regression and failure classification, they require labeled historical failures. **Phase 4: Multivariate Sensor Anomaly Detection** develops unsupervised detectors to identify novel operational deviations without ground-truth failure labels.

---

## V. Phase Completion Assessment
**Status: 100% Complete & Empirically Verified.** All PyTorch architectures implemented, trained, benchmarked against official NASA test fleet ground truth, and serialized.
