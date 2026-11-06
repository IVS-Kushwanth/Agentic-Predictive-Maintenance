"""
Unit and Integration Test Suite for Phase 8 FastAPI Backend.
Tests all endpoints:
1. GET /health
2. GET /api/models
3. POST /api/predict/rul
4. POST /api/predict/failure
5. POST /api/anomaly/detect
6. POST /api/explain
7. POST /api/agents/analyze
8. POST /api/recommendations
9. GET /api/history
10. POST /api/recommendations/review
Plus error handling and validation tests.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.backend.main import app


class TestPhase8Backend(unittest.TestCase):
    """Exhaustive tests for FastAPI backend routes and services."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

        # Real test samples
        cls.val_npz = np.load(os.path.join(BASE_DIR, "data", "processed", "nasa", "FD001", "sequences_w30_val.npz"))
        cls.nasa_sample = cls.val_npz["X"][10][-1, :].tolist()  # 153 features

        cls.X_test_ai4i = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "ai4i", "X_test.csv"))
        cls.ai4i_sample = cls.X_test_ai4i.iloc[0].tolist()  # 14 features

    def test_01_health_endpoint(self):
        """Test GET /health returns 200 and healthy status."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("timestamp", data)
        print("  [PASS] Test 1: GET /health Verified")

    def test_02_models_catalog(self):
        """Test GET /api/models returns model catalog."""
        resp = self.client.get("/api/models")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertGreaterEqual(data["total_models"], 5)
        self.assertTrue(any(m["dataset"] == "NASA" for m in data["models"]))
        self.assertTrue(any(m["dataset"] == "AI4I" for m in data["models"]))
        print("  [PASS] Test 2: GET /api/models Verified")

    def test_03_predict_nasa_rul(self):
        """Test POST /api/predict/rul returns RUL and early warning."""
        payload = {
            "machine_id": "ENG-TEST-01",
            "sensor_features": self.nasa_sample,
            "model_type": "xgboost"
        }
        resp = self.client.post("/api/predict/rul", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["dataset"], "NASA")
        self.assertIsInstance(data["predicted_rul"], float)
        self.assertGreater(data["predicted_rul"], 0.0)
        self.assertIn("early_warning_status", data)
        print("  [PASS] Test 3: POST /api/predict/rul Verified")

    def test_04_predict_ai4i_failure(self):
        """Test POST /api/predict/failure returns breakdown probability."""
        payload = {
            "machine_id": "CNC-TEST-02",
            "operating_features": self.ai4i_sample,
            "model_type": "xgboost"
        }
        resp = self.client.post("/api/predict/failure", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["dataset"], "AI4I")
        self.assertIsInstance(data["failure_risk_score"], float)
        self.assertTrue(0.0 <= data["failure_risk_score"] <= 1.0)
        self.assertIn(data["failure_prediction"], [0, 1])
        print("  [PASS] Test 4: POST /api/predict/failure Verified")

    def test_05_anomaly_detection(self):
        """Test POST /api/anomaly/detect calculates severity and reconstruction error."""
        payload = {
            "machine_id": "ENG-ANOM-01",
            "dataset_type": "NASA",
            "features": self.nasa_sample,
            "detector_type": "autoencoder"
        }
        resp = self.client.post("/api/anomaly/detect", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("anomaly_score", data)
        self.assertIn(data["severity"], ["Normal", "Low anomaly", "Medium anomaly", "High anomaly"])
        print("  [PASS] Test 5: POST /api/anomaly/detect Verified")

    def test_06_explain_endpoint(self):
        """Test POST /api/explain calculates TreeSHAP attributions."""
        payload = {
            "machine_id": "CNC-EXP-01",
            "dataset_type": "AI4I",
            "features": self.ai4i_sample,
            "method": "shap",
            "top_k": 3
        }
        resp = self.client.post("/api/explain", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["top_contributing_sensors"]), 3)
        self.assertIn("feature", data["top_contributing_sensors"][0])
        print("  [PASS] Test 6: POST /api/explain Verified")

    def test_07_agents_analyze_endpoint(self):
        """Test POST /api/agents/analyze executes full multi-agent LangGraph workflow."""
        payload = {
            "machine_id": "CNC-AGENT-01",
            "dataset_type": "AI4I",
            "sensor_data": self.ai4i_sample,
            "workflow_type": "full_analysis"
        }
        resp = self.client.post("/api/agents/analyze", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["workflow_status"], "COMPLETED")
        self.assertIsNotNone(data["recommendation_id"])
        self.assertIn("suggested_maintenance_actions", data)
        print("  [PASS] Test 7: POST /api/agents/analyze Verified")

    def test_08_recommendation_endpoint(self):
        """Test POST /api/recommendations generates evidence-linked recommendations."""
        payload = {
            "dataset_type": "NASA",
            "machine_id": "ENG-REC-01",
            "predicted_rul": 15.0,
            "anomaly_severity": "High anomaly",
            "anomaly_score": 0.22,
            "important_sensors": [{"feature": "sensor_11", "importance": 0.45, "contribution": "positive"}]
        }
        resp = self.client.post("/api/recommendations", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["priority"], "Critical review required")
        self.assertTrue(data["human_approval_required"])
        self.assertGreater(len(data["recommended_actions"]), 0)
        print("  [PASS] Test 8: POST /api/recommendations Verified")

    def test_09_history_and_review_workflow(self):
        """Test GET /api/history and POST /api/recommendations/review lifecycle."""
        # 1. Create recommendation to review
        payload = {
            "dataset_type": "NASA",
            "machine_id": "ENG-AUDIT-99",
            "predicted_rul": 45.0,
            "anomaly_severity": "Normal"
        }
        create_resp = self.client.post("/api/recommendations", json=payload)
        self.assertEqual(create_resp.status_code, 200)
        rec_id = create_resp.json()["recommendation_id"]

        # 2. Check history contains this record
        hist_resp = self.client.get(f"/api/history?machine_id=ENG-AUDIT-99")
        self.assertEqual(hist_resp.status_code, 200)
        hist_data = hist_resp.json()
        self.assertGreater(hist_data["total_records"], 0)

        # 3. Approve recommendation
        review_payload = {
            "recommendation_id": rec_id,
            "status": "APPROVED",
            "reviewed_by": "Senior Propulsion Specialist Smith",
            "review_notes": "Flight authorization granted with standard 50-cycle inspection interval"
        }
        review_resp = self.client.post("/api/recommendations/review", json=review_payload)
        self.assertEqual(review_resp.status_code, 200)
        review_data = review_resp.json()
        self.assertTrue(review_data["success"])
        self.assertEqual(review_data["review_status"], "APPROVED")
        print("  [PASS] Test 9: History & Review Lifecycle Verified")

    def test_10_validation_error_handling(self):
        """Test validation error handling returns 422 for malformed requests."""
        # Case A: Wrong number of features (5 instead of 14 for AI4I)
        bad_payload = {
            "machine_id": "CNC-BAD",
            "operating_features": [1.0, 2.0, 3.0, 4.0, 5.0]
        }
        resp = self.client.post("/api/predict/failure", json=bad_payload)
        self.assertEqual(resp.status_code, 422)

        # Case B: Invalid review status
        bad_review = {
            "recommendation_id": "REC-NONEXISTENT",
            "status": "MAYBE_LATER",
            "reviewed_by": "Engineer"
        }
        resp_rev = self.client.post("/api/recommendations/review", json=bad_review)
        self.assertEqual(resp_rev.status_code, 422)
        print("  [PASS] Test 10: Validation Error Handling (422) Verified")


if __name__ == "__main__":
    unittest.main()
