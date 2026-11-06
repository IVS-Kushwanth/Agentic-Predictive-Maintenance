# Project Evidence: Evaluation Results & Raw Metric Tables

This directory catalogs the raw performance metrics and experiment logs compiled during benchmarking.

---

## 1. Serialized Experiment Logs
- [`reports/phase2/experiment_log.csv`](file:///C:/Users/HP/OneDrive/Desktop/ML%20Model/agentic-predictive-maintenance/reports/phase2/experiment_log.csv): Full hyperparameter sweep logs for 32 traditional ML baselines.
- [`reports/phase3/experiment_log.csv`](file:///C:/Users/HP/OneDrive/Desktop/ML%20Model/agentic-predictive-maintenance/reports/phase3/experiment_log.csv): Training epoch logs for 18 PyTorch deep learning models.

---

## 2. Master Evaluation Table

| Task | Benchmark Dataset | Champion Architecture | Primary Metric | Secondary Metric |
| :--- | :--- | :--- | :---: | :---: |
| **RUL Estimation** | NASA C-MAPSS FD001 ($N=100$) | **Bidirectional LSTM ($L=60$)** | **RMSE: 20.02 cycles** | **NASA Score: 889.4** |
| **Tabular RUL Baseline** | NASA C-MAPSS FD001 ($N=100$) | **XGBoost Regressor** | **RMSE: 22.50 cycles** | MAE: 15.35 cycles |
| **Early Warning ($H=10$)**| NASA C-MAPSS FD001 | **XGBoost Classifier** | **PR-AUC: 0.8220** | Recall: 82.35% |
| **Early Warning ($H=30$)**| NASA C-MAPSS FD001 | **Random Forest Classifier** | **PR-AUC: 0.8958** | F1: 0.7780 |
| **Machine Failure** | UCI AI4I 2020 ($N=2,000$) | **Gradient Boosting Classifier** | **F1: 0.8689** | **Precision: 98.15%** |
| **Tabular Anomaly** | UCI AI4I 2020 ($N=2,000$) | **Isolation Forest** | **ROC-AUC: 0.8824** | PR-AUC: 0.3079 |
| **Sequence Anomaly** | NASA C-MAPSS FD001 | **LSTM Autoencoder** | **Recon Loss: 0.0285** | Threshold: 0.0370 |
