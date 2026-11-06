"""
Phase 6 Deliverables Verification Script.
Audits all agent, graph, service, and report files, and demonstrates end-to-end
LangGraph workflows on actual empirical dataset samples.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.graph import (
    analyze_machine,
    assess_failure_risk,
    estimate_remaining_useful_life,
    explain_machine_condition,
    recommend_maintenance
)


def verify_phase6():
    print("=" * 80)
    print("PHASE 6 DELIVERABLES VERIFICATION AUDIT")
    print("=" * 80)

    errors = []

    # 1. Audit Agent Modules
    print("\n1. Auditing Agent Modules in src/agents/...")
    agent_dir = os.path.join(BASE_DIR, "src", "agents")
    expected_agents = [
        "__init__.py",
        "supervisor.py",
        "prediction_agent.py",
        "anomaly_agent.py",
        "explanation_agent.py",
        "maintenance_agent.py"
    ]
    for a in expected_agents:
        p = os.path.join(agent_dir, a)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            print(f"  [OK] Found agent module: {a} ({os.path.getsize(p):,} bytes)")
        else:
            errors.append(f"Missing or empty agent file: {p}")
            print(f"  [FAIL] Missing agent: {p}")

    # 2. Audit Graph Modules
    print("\n2. Auditing Graph Modules in src/graph/...")
    graph_dir = os.path.join(BASE_DIR, "src", "graph")
    expected_graph = [
        "__init__.py",
        "state.py",
        "routing.py",
        "workflow.py"
    ]
    for g in expected_graph:
        p = os.path.join(graph_dir, g)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            print(f"  [OK] Found graph module: {g} ({os.path.getsize(p):,} bytes)")
        else:
            errors.append(f"Missing or empty graph file: {p}")
            print(f"  [FAIL] Missing graph file: {p}")

    # 3. Audit Service Modules
    print("\n3. Auditing Service Modules in src/services/...")
    services_dir = os.path.join(BASE_DIR, "src", "services")
    expected_services = [
        "__init__.py",
        "model_service.py",
        "explanation_service.py"
    ]
    for s in expected_services:
        p = os.path.join(services_dir, s)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            print(f"  [OK] Found service module: {s} ({os.path.getsize(p):,} bytes)")
        else:
            errors.append(f"Missing or empty service file: {p}")
            print(f"  [FAIL] Missing service file: {p}")

    # 4. Audit Technical Reports
    print("\n4. Auditing Technical Reports in reports/phase6/...")
    reports_dir = os.path.join(BASE_DIR, "reports", "phase6")
    expected_reports = [
        "agent_architecture.md",
        "workflow_documentation.md",
        "agent_testing_report.md"
    ]
    for r in expected_reports:
        p = os.path.join(reports_dir, r)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            print(f"  [OK] Found report: {r} ({os.path.getsize(p):,} bytes)")
        else:
            errors.append(f"Missing or empty report: {p}")
            print(f"  [FAIL] Missing report: {p}")

    # 5. Live Demonstration of Complete Workflows Using Real Dataset Samples
    print("\n5. Executing Live LangGraph Workflows on Real Samples...")

    val_npz = np.load(os.path.join(BASE_DIR, "data", "processed", "nasa", "FD001", "sequences_w30_val.npz"))
    nasa_healthy = val_npz["X"][10]
    deg_idx = int(np.argmin(val_npz["y_rul"]))
    nasa_degraded = val_npz["X"][deg_idx]

    X_test_ai4i = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "ai4i", "X_test.csv"))
    y_test_ai4i = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "ai4i", "y_test.csv")).values.ravel()
    ai4i_normal = X_test_ai4i[y_test_ai4i == 0].iloc[0].to_numpy(dtype=np.float32)
    ai4i_failure = X_test_ai4i[y_test_ai4i == 1].iloc[0].to_numpy(dtype=np.float32)

    # Demo A: NASA Nominal
    try:
        res_nasa_nom = analyze_machine(nasa_healthy, dataset_type="NASA", machine_id="TURBOFAN-NOMINAL-10")
        if res_nasa_nom["workflow_status"] != "COMPLETED":
            raise ValueError(f"Workflow failed: {res_nasa_nom['errors']}")
        print(f"  [OK] NASA Nominal Workflow -> Pred RUL: {res_nasa_nom['rul_prediction']} cycles | Anomaly: {res_nasa_nom['anomaly_severity']} | Human Review: {res_nasa_nom['human_review_status']['human_review_required']}")
    except Exception as e:
        errors.append(f"NASA Nominal Demo failed: {str(e)}")
        print(f"  [FAIL] NASA Nominal Demo: {str(e)}")

    # Demo B: NASA Degraded
    try:
        res_nasa_deg = analyze_machine(nasa_degraded, dataset_type="NASA", machine_id="TURBOFAN-CRITICAL-99")
        if res_nasa_deg["workflow_status"] != "COMPLETED":
            raise ValueError(f"Workflow failed: {res_nasa_deg['errors']}")
        print(f"  [OK] NASA Degraded Workflow -> Pred RUL: {res_nasa_deg['rul_prediction']} cycles | Anomaly: {res_nasa_deg['anomaly_severity']} | Human Review: {res_nasa_deg['human_review_status']['human_review_required']}")
    except Exception as e:
        errors.append(f"NASA Degraded Demo failed: {str(e)}")
        print(f"  [FAIL] NASA Degraded Demo: {str(e)}")

    # Demo C: AI4I Normal
    try:
        res_ai4i_nom = analyze_machine(ai4i_normal, dataset_type="AI4I", machine_id="CNC-NOMINAL-01")
        if res_ai4i_nom["workflow_status"] != "COMPLETED":
            raise ValueError(f"Workflow failed: {res_ai4i_nom['errors']}")
        print(f"  [OK] AI4I Normal Workflow  -> Prob: {res_ai4i_nom['failure_risk_score']} | Status: {res_ai4i_nom['failure_label']} | Human Review: {res_ai4i_nom['human_review_status']['human_review_required']}")
    except Exception as e:
        errors.append(f"AI4I Normal Demo failed: {str(e)}")
        print(f"  [FAIL] AI4I Normal Demo: {str(e)}")

    # Demo D: AI4I Breakdown
    try:
        res_ai4i_fail = analyze_machine(ai4i_failure, dataset_type="AI4I", machine_id="CNC-FAILURE-08")
        if res_ai4i_fail["workflow_status"] != "COMPLETED":
            raise ValueError(f"Workflow failed: {res_ai4i_fail['errors']}")
        print(f"  [OK] AI4I Failure Workflow -> Prob: {res_ai4i_fail['failure_risk_score']} | Status: {res_ai4i_fail['failure_label']} | Human Review: {res_ai4i_fail['human_review_status']['human_review_required']}")
    except Exception as e:
        errors.append(f"AI4I Failure Demo failed: {str(e)}")
        print(f"  [FAIL] AI4I Failure Demo: {str(e)}")

    print("\n" + "=" * 80)
    if errors:
        print(f"AUDIT FAILED WITH {len(errors)} ERROR(S):")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("ALL PHASE 6 DELIVERABLES VERIFIED SUCCESSFULLY! (100% PASS)")
        print("=" * 80)
        sys.exit(0)


if __name__ == "__main__":
    verify_phase6()
