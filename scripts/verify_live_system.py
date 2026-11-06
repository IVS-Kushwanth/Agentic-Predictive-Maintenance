"""
Comprehensive Live System Verification Script.
Tests:
1. FastAPI Health Check (/health)
2. Streamlit UI Health & Page Load
3. CORS Policy Enforcement (Permitted vs Rejected Origins)
4. NASA RUL Endpoint (POST /api/predict/rul)
5. AI4I Failure Endpoint (POST /api/predict/failure)
6. Anomaly Detection Endpoint (POST /api/anomaly/detect)
7. SHAP Explanation Endpoint (POST /api/explain)
8. LangGraph LLM-Free Agent Workflow (POST /api/agents/analyze)
9. Maintenance Recommendation Engine (POST /api/recommendations)
10. Database Persistence & Review Lifecycle (/api/history, /api/recommendations/review)
"""

import sys
import json
import time
import httpx
import pandas as pd
import numpy as np

BACKEND_URL = "http://127.0.0.1:8000"
FRONTEND_URL = "http://127.0.0.1:8501"

results = {}

with httpx.Client(timeout=15.0) as client:
    # 1. FastAPI Health Check
    try:
        r = client.get(f"{BACKEND_URL}/health")
        data = r.json()
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        assert data.get("status") == "healthy", f"Expected healthy status, got {data}"
        results["FastAPI Health Check"] = {"status": "PASS", "data": data}
        print(" [PASS] 1. FastAPI Health Check responded 200 OK with healthy status.")
    except Exception as e:
        results["FastAPI Health Check"] = {"status": "FAIL", "error": str(e)}
        print(f" [FAIL] 1. FastAPI Health Check failed: {e}")

    # 2. Streamlit UI Load
    try:
        r = client.get(FRONTEND_URL)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        assert "Streamlit" in r.text or "streamlit" in r.text, "Expected Streamlit HTML in response"
        # Test Streamlit core health
        r_health = client.get(f"{FRONTEND_URL}/_stcore/health")
        results["Streamlit UI Load"] = {"status": "PASS", "http_status": r.status_code, "st_health": r_health.status_code}
        print(f" [PASS] 2. Streamlit UI loaded successfully (HTTP {r.status_code}, _stcore/health {r_health.status_code}).")
    except Exception as e:
        results["Streamlit UI Load"] = {"status": "FAIL", "error": str(e)}
        print(f" [FAIL] 2. Streamlit UI Load failed: {e}")

    # 3. CORS Enforcement Test
    try:
        # 3a: Permitted origin
        headers_allowed = {"Origin": "http://localhost:8501"}
        r_cors_allowed = client.options(
            f"{BACKEND_URL}/api/predict/rul",
            headers={**headers_allowed, "Access-Control-Request-Method": "POST"}
        )
        allow_origin = r_cors_allowed.headers.get("access-control-allow-origin")

        # 3b: Unauthorized origin
        headers_blocked = {"Origin": "http://malicious-site.example.com"}
        r_cors_blocked = client.options(
            f"{BACKEND_URL}/api/predict/rul",
            headers={**headers_blocked, "Access-Control-Request-Method": "POST"}
        )
        blocked_origin = r_cors_blocked.headers.get("access-control-allow-origin")

        assert allow_origin == "http://localhost:8501", f"Expected allow-origin for localhost:8501, got {allow_origin}"
        assert blocked_origin is None or blocked_origin != "http://malicious-site.example.com", f"Unauthorized origin should be blocked, got {blocked_origin}"
        results["CORS Enforcement"] = {"status": "PASS", "allowed_origin": allow_origin, "blocked_header": blocked_origin}
        print(f" [PASS] 3. CORS Enforcement verified: Permitted {allow_origin}, Blocked {blocked_origin}.")
    except Exception as e:
        results["CORS Enforcement"] = {"status": "FAIL", "error": str(e)}
        print(f" [FAIL] 3. CORS Enforcement failed: {e}")

    # Load test data
    with open("models/baseline/nasa_rul_xgboost_regressor_FD001_meta.json") as f:
        nasa_meta = json.load(f)
    nasa_feat_names = nasa_meta["features"]
    df_val = pd.read_csv("data/processed/nasa/FD001/val_tabular.csv")
    nasa_features = df_val[nasa_feat_names].iloc[0].tolist()

    df_ai4i = pd.read_csv("data/processed/ai4i/X_test.csv")
    ai4i_features = df_ai4i.iloc[0].tolist()

    # 4. NASA RUL Prediction Endpoint
    try:
        payload = {
            "machine_id": "ENGINE-FD001-VAL-001",
            "sensor_features": nasa_features,
            "model_type": "xgboost"
        }
        r = client.post(f"{BACKEND_URL}/api/predict/rul", json=payload)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        data = r.json()
        assert "predicted_rul" in data, f"Missing predicted_rul: {data}"
        results["NASA RUL Prediction"] = {"status": "PASS", "predicted_rul": data["predicted_rul"], "unit": data["unit"]}
        print(f" [PASS] 4. NASA RUL Prediction: {data['predicted_rul']} {data['unit']} (Model: {data['model_used']}).")
    except Exception as e:
        results["NASA RUL Prediction"] = {"status": "FAIL", "error": str(e)}
        print(f" [FAIL] 4. NASA RUL Prediction failed: {e}")

    # 5. AI4I Failure Prediction Endpoint
    try:
        payload = {
            "machine_id": "CNC-AI4I-TEST-001",
            "operating_features": ai4i_features,
            "model_type": "xgboost"
        }
        r = client.post(f"{BACKEND_URL}/api/predict/failure", json=payload)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        data = r.json()
        assert "failure_risk_score" in data, f"Missing failure_risk_score: {data}"
        results["AI4I Failure Prediction"] = {"status": "PASS", "failure_risk_score": data["failure_risk_score"], "label": data["failure_label"]}
        print(f" [PASS] 5. AI4I Failure Prediction: Risk={data['failure_risk_score']:.4f}, Label='{data['failure_label']}'.")
    except Exception as e:
        results["AI4I Failure Prediction"] = {"status": "FAIL", "error": str(e)}
        print(f" [FAIL] 5. AI4I Failure Prediction failed: {e}")

    # 6. Anomaly Detection Endpoint
    try:
        payload = {
            "machine_id": "CNC-AI4I-TEST-001",
            "dataset_type": "AI4I",
            "features": ai4i_features,
            "detector_type": "isolation_forest"
        }
        r = client.post(f"{BACKEND_URL}/api/anomaly/detect", json=payload)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        data = r.json()
        assert "anomaly_score" in data, f"Missing anomaly_score: {data}"
        results["Anomaly Detection"] = {"status": "PASS", "score": data["anomaly_score"], "severity": data["severity"]}
        print(f" [PASS] 6. Anomaly Detection: Score={data['anomaly_score']:.4f}, Severity='{data['severity']}'.")
    except Exception as e:
        results["Anomaly Detection"] = {"status": "FAIL", "error": str(e)}
        print(f" [FAIL] 6. Anomaly Detection failed: {e}")

    # 7. SHAP Explanation Endpoint
    try:
        payload = {
            "machine_id": "CNC-AI4I-TEST-001",
            "dataset_type": "AI4I",
            "features": ai4i_features,
            "method": "shap",
            "top_k": 5
        }
        r = client.post(f"{BACKEND_URL}/api/explain", json=payload)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        data = r.json()
        top_sensors = data.get("top_contributing_sensors", [])
        assert len(top_sensors) > 0, f"Expected sensors, got {data}"
        results["SHAP Explanation"] = {"status": "PASS", "top_sensor": top_sensors[0]["feature"], "importance": top_sensors[0]["importance"]}
        print(f" [PASS] 7. SHAP Explanation: Top feature='{top_sensors[0]['feature']}', Importance={top_sensors[0]['importance']:.4f}.")
    except Exception as e:
        results["SHAP Explanation"] = {"status": "FAIL", "error": str(e)}
        print(f" [FAIL] 7. SHAP Explanation failed: {e}")

    # 8. LangGraph LLM-Free Agent Workflow
    try:
        payload = {
            "machine_id": "CNC-AI4I-TEST-001",
            "dataset_type": "AI4I",
            "sensor_data": ai4i_features,
            "workflow_type": "full_analysis"
        }
        r = client.post(f"{BACKEND_URL}/api/agents/analyze", json=payload)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        data = r.json()
        assert "workflow_status" in data
        assert "priority" in data
        results["LangGraph Multi-Agent Analysis"] = {"status": "PASS", "status_msg": data["workflow_status"], "priority": data.get("priority")}
        print(f" [PASS] 8. LangGraph Multi-Agent Workflow: Priority='{data.get('priority')}', Status='{data['workflow_status']}'.")
    except Exception as e:
        results["LangGraph Multi-Agent Analysis"] = {"status": "FAIL", "error": str(e)}
        print(f" [FAIL] 8. LangGraph Multi-Agent Workflow failed: {e}")

    # 9. Maintenance Recommendation Engine
    try:
        payload = {
            "dataset_type": "AI4I",
            "machine_id": "CNC-AI4I-TEST-001",
            "failure_risk_score": 0.88,
            "anomaly_severity": "High anomaly",
            "anomaly_score": 0.72,
            "model_confidence": 0.91
        }
        r = client.post(f"{BACKEND_URL}/api/recommendations", json=payload)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        data = r.json()
        assert "priority" in data
        assert "recommended_actions" in data
        rec_id = data.get("recommendation_id")
        results["Maintenance Recommendations"] = {"status": "PASS", "rec_id": rec_id, "priority": data["priority"]}
        print(f" [PASS] 9. Maintenance Recommendation Engine: ID={rec_id}, Priority='{data['priority']}'.")
    except Exception as e:
        results["Maintenance Recommendations"] = {"status": "FAIL", "error": str(e)}
        print(f" [FAIL] 9. Maintenance Recommendation Engine failed: {e}")
        rec_id = None

    # 10. Database Persistence & Human Review Lifecycle
    try:
        # Check history contains recommendations
        r_hist = client.get(f"{BACKEND_URL}/api/history?limit=5")
        assert r_hist.status_code == 200, f"Expected 200, got {r_hist.status_code}"
        hist_data = r_hist.json()
        assert hist_data["total_records"] > 0, "Expected non-empty history in database"

        # Submit human review on the recommendation
        if rec_id:
            review_payload = {
                "recommendation_id": rec_id,
                "status": "APPROVED",
                "reviewed_by": "Chief Reliability Engineer",
                "review_notes": "Verified tool wear exceeding limits. Immediate replacement approved."
            }
            r_rev = client.post(f"{BACKEND_URL}/api/recommendations/review", json=review_payload)
            assert r_rev.status_code == 200, f"Expected 200, got {r_rev.status_code}: {r_rev.text}"
            rev_data = r_rev.json()
            assert rev_data.get("review_status") == "APPROVED"
            results["Database Persistence & Review Lifecycle"] = {"status": "PASS", "total_records": hist_data["total_records"], "review_status": rev_data.get("review_status")}
            print(f" [PASS] 10. Database Persistence & Human Review: Total records={hist_data['total_records']}, Review updated to '{rev_data.get('review_status')}'.")
        else:
            results["Database Persistence & Review Lifecycle"] = {"status": "PARTIAL", "total_records": hist_data["total_records"]}
    except Exception as e:
        results["Database Persistence & Review Lifecycle"] = {"status": "FAIL", "error": str(e)}
        print(f" [FAIL] 10. Database Persistence & Review Lifecycle failed: {e}")

print("\n" + "="*70)
all_pass = all(v.get("status") == "PASS" for v in results.values())
print(f"OVERALL LIVE SYSTEM STATUS: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
print("="*70)
