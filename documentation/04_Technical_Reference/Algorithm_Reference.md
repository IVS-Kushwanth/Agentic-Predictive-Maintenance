# Comprehensive Algorithm Reference
## Agentic Predictive Maintenance Framework

This reference profiles every major machine learning, deep learning, anomaly detection, and explainability algorithm implemented and verified in the codebase.

---

## 1. Traditional Supervised Learning Algorithms

### 1.1 XGBoost Regressor & Classifier
- **Class**: `xgboost.XGBRegressor`, `xgboost.XGBClassifier`
- **Source Module**: `src/training/nasa_rul_trainer.py`, `src/training/ai4i_trainer.py`
- **Mathematical Principle**: Scalable tree boosting that minimizes regularized second-order Taylor expansions of the objective function.
- **Role in Framework**:
  - NASA C-MAPSS: Tabular RUL regression (Terminal RMSE = 22.50 cycles, MAE = 15.35 cycles).
  - NASA C-MAPSS: Early warning binary classification ($H=10$ PR-AUC = 0.8220).
  - AI4I: Machine failure classification (PR-AUC = 0.8630, F1 = 0.8615).
- **Hyperparameter Configuration**:
  - `n_estimators`: 150
  - `learning_rate`: 0.03
  - `max_depth`: 6
  - `subsample`: 0.80
  - `colsample_bytree`: 0.80
  - `reg_alpha`: 0.1 ($L_1$ leaf weight regularization)
  - `reg_lambda`: 1.0 ($L_2$ leaf weight regularization)

### 1.2 Random Forest Regressor & Classifier
- **Class**: `sklearn.ensemble.RandomForestRegressor`, `sklearn.ensemble.RandomForestClassifier`
- **Source Module**: `src/training/nasa_rul_trainer.py`, `src/training/ai4i_trainer.py`
- **Mathematical Principle**: Bootstrap aggregation (bagging) of 200 decorrelated decision trees using random feature subspace sampling at each split.
- **Role in Framework**: Multi-horizon classification ($H=30$ PR-AUC = 0.8958, F1 = 0.7780).
- **Hyperparameter Configuration**:
  - `n_estimators`: 200
  - `max_depth`: 12
  - `min_samples_split`: 5
  - `min_samples_leaf`: 2
  - `n_jobs`: -1

### 1.3 Gradient Boosting Classifier
- **Class**: `sklearn.ensemble.GradientBoostingClassifier`
- **Source Module**: `src/training/ai4i_trainer.py`
- **Mathematical Principle**: Sequential stage-wise additive boosting optimizing Bernoulli deviance loss.
- **Role in Framework**: Champion model for AI4I machine failure prediction under tuned threshold $\tau^* = 0.485$ (Precision = **98.15%**, Recall = **77.94%**, F1 = **0.8689**).

---

## 2. Deep Temporal Recurrent Architectures

### 2.1 Bidirectional LSTM (Bi-LSTM) Regressor
- **Class**: `BiLSTMRegressor(nn.Module)`
- **Source Module**: `src/generate_phase3_artifacts.py`
- **Mathematical Principle**: Dual-directional recurrent gating allowing both past and forward temporal sequence contexts to inform the terminal degradation state.
- **Architecture**:
  - Input: $(B, 60, 14)$
  - Layer 1: Bidirectional LSTM (64 hidden per direction = 128 total)
  - Layer 2: Bidirectional LSTM (32 hidden per direction = 64 total)
  - Dense Head: Linear(64, 32) $\to$ ReLU $\to$ Linear(32, 16) $\to$ ReLU $\to$ Linear(16, 1)
- **Role in Framework**: Champion NASA RUL model (Terminal RMSE = **20.02 cycles**, NASA Score = **889.4**).

### 2.2 Gated Recurrent Unit (GRU) Regressor
- **Class**: `GRURegressor(nn.Module)`
- **Source Module**: `src/generate_phase3_artifacts.py`
- **Mathematical Principle**: Simplified recurrent gating merging forget and input gates into a single update gate.
- **Role in Framework**: Parameter-efficient sequence regression (19,041 parameters, Terminal RMSE = 22.21 cycles).

---

## 3. Unsupervised Anomaly Detection Algorithms

### 3.1 Multivariate LSTM Autoencoder
- **Class**: `LSTMAutoencoder(nn.Module)`
- **Source Module**: `src/anomaly/`
- **Mathematical Principle**: Sequence compression into a low-dimensional bottleneck ($z \in \mathbb{R}^{32}$) followed by sequence reconstruction; reconstruction error measures deviation from nominal health.
- **Role in Framework**: Unsupervised sequence anomaly scoring for NASA turbofan fleets ($\tau = 0.0370$).

### 3.2 Isolation Forest
- **Class**: `sklearn.ensemble.IsolationForest`
- **Source Module**: `src/anomaly/`
- **Mathematical Principle**: Isolates anomalies in random recursive axis-aligned hyperplanes; anomalous points require fewer splits to isolate.
- **Role in Framework**: Champion unsupervised detector for AI4I milling data (Test ROC-AUC = **0.8824**, 9.1x random chance lift).

---

## 4. Explainable AI Algorithms

### 4.1 TreeSHAP
- **Class**: `shap.TreeExplainer`
- **Source Module**: `src/services/explanation_service.py`
- **Mathematical Principle**: Polynomial-time exact Shapley attribution algorithm for decision trees running in $\mathcal{O}(T L D^2)$ time where $T$ is trees, $L$ is leaves, and $D$ is maximum tree depth.
- **Role in Framework**: Explains all tree-based predictions in $< 200\text{ ms}$, enforcing additive efficiency.
