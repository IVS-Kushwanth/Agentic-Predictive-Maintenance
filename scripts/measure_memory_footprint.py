"""
Empirical Memory & CPU Profiler for Cloud Deployment Sizing.
Measures process RSS memory (MB) and peak memory across:
- Startup
- Model Loading
- NASA RUL Inference
- AI4I Failure Inference
- Anomaly Detection
- TreeSHAP Explanation
- LangGraph Multi-Agent Workflow
- Sustained Multi-Request Batch (50 iterations)
"""

import os
import sys
import gc
import json
import time
import psutil
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

process = psutil.Process(os.getpid())

def get_mem_mb() -> float:
    return process.memory_info().rss / (1024 * 1024)

mem_log = []

def record(stage: str):
    gc.collect()
    mem = get_mem_mb()
    mem_log.append((stage, mem))
    print(f"[{stage:35s}] Memory: {mem:6.2f} MB")
    return mem

record("1. Python Baseline")

# Step 2: Import FastAPI application
from app.backend.main import app
record("2. After app.backend.main Import")

# Load real validation data
with open("models/baseline/nasa_rul_xgboost_regressor_FD001_meta.json") as f:
    nasa_meta = json.load(f)
df_val = pd.read_csv("data/processed/nasa/FD001/val_tabular.csv")
nasa_features = df_val[nasa_meta["features"]].iloc[0].values

df_ai4i = pd.read_csv("data/processed/ai4i/X_test.csv")
ai4i_features = df_ai4i.iloc[0].values

record("3. After Validation Data Load")

# Step 4: Model Service - NASA RUL
from app.backend.services import get_model_service, get_explainability_manager, get_history_db
model_svc = get_model_service()
rul_res = model_svc.predict_nasa_rul(nasa_features)
record("4. After NASA RUL Inference (XGB)")

# Step 5: Model Service - AI4I Failure
fail_res = model_svc.predict_ai4i_failure(ai4i_features)
record("5. After AI4I Failure Inference")

# Step 6: Anomaly Detection
from app.backend.services.anomaly_service import AnomalyService
anom_svc = AnomalyService()
anom_res = anom_svc.detect_anomaly("CNC-01", "AI4I", ai4i_features.tolist(), "isolation_forest")
record("6. After Anomaly Detection (IForest)")

# Step 7: TreeSHAP Explanation
from app.backend.services.explain_service import ExplainService
exp_svc = ExplainService()
exp_res = exp_svc.explain("CNC-01", "AI4I", ai4i_features.tolist(), "shap", top_k=5)
record("7. After TreeSHAP Explanation")

# Step 8: LangGraph Multi-Agent Workflow
from src.graph.workflow import analyze_machine
agent_res = analyze_machine(
    input_data=ai4i_features,
    dataset_type="AI4I",
    machine_id="CNC-01",
    workflow_type="full_analysis"
)
record("8. After LangGraph Full Workflow")

# Step 9: Stress test - 50 consecutive full cycles to check memory leaks
print("\nRunning 50 consecutive full inference cycles...")
peak_mem = get_mem_mb()
t0 = time.time()
for i in range(50):
    model_svc.predict_nasa_rul(nasa_features)
    model_svc.predict_ai4i_failure(ai4i_features)
    anom_svc.detect_anomaly(f"CNC-{i}", "AI4I", ai4i_features.tolist(), "isolation_forest")
    exp_svc.explain(f"CNC-{i}", "AI4I", ai4i_features.tolist(), "shap", top_k=5)
    analyze_machine(ai4i_features, "AI4I", f"CNC-{i}", "full_analysis")
    cur_mem = get_mem_mb()
    if cur_mem > peak_mem:
        peak_mem = cur_mem

total_time = time.time() - t0
record("9. After 50 Full Cycles")

print("\n" + "=" * 65)
print("EMPIRICAL MEMORY PROFILING REPORT")
print("=" * 65)
print(f"Startup RAM (Application Loaded)   : {mem_log[1][1]:.2f} MB")
print(f"Post-Warmup RAM (All Models Cached): {mem_log[7][1]:.2f} MB")
print(f"Peak RAM under 50 cycles load      : {peak_mem:.2f} MB")
print(f"50 cycles execution time           : {total_time:.2f} s ({total_time/50:.3f} s/req)")
print("=" * 65)
