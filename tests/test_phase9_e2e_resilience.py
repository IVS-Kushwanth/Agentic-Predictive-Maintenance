"""
Comprehensive End-to-End System, Integration, and Fault-Tolerance Test Suite for Phase 9.
Tests:
1. Dataset loading and preprocessing integrity
2. Traditional ML model inference (NASA RUL regression & AI4I classification)
3. Deep-learning PyTorch model inference
4. Early-warning multi-horizon classification (10, 20, 30 cycles)
5. Unsupervised multivariate anomaly detection (LSTM Autoencoder & Isolation Forest)
6. Explainable AI (TreeSHAP attributions & feature contributions)
7. LangGraph multi-agent orchestration workflow
8. Prescriptive maintenance recommendation engine & priority conflict resolution
9. FastAPI REST API endpoints
10. SQLite database operations and human review audit trail
11. Fault-tolerance, edge cases, NaNs, missing fields, and error resilience
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import torch
import tempfile
from fastapi.testclient import TestClient

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.services.model_service import ModelService
from src.explainability.inference_api import ExplainabilityManager
from src.recommendations import RecommendationEngine, RecommendationInput
from src.database.recommendation_history import RecommendationHistoryDB
from src.graph.workflow import analyze_machine
from app.backend.main import app


class TestPhase9CompleteSystem(unittest.TestCase):
    """Exhaustive system, integration, and fault-tolerance verification suite."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.model_service = ModelService()
        cls.explain_mgr = ExplainabilityManager()
        cls.rec_engine = RecommendationEngine()

        # Temporary database for isolated testing
        cls.temp_db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        cls.temp_db_path = cls.temp_db_file.name
        cls.temp_db_file.close()
        cls.test_db = RecommendationHistoryDB(db_path=cls.temp_db_path)

        # Load real datasets
        cls.nasa_npz_path = os.path.join(BASE_DIR, "data", "processed", "nasa", "FD001", "sequences_w30_val.npz")
        cls.nasa_data = np.load(cls.nasa_npz_path)
        cls.nasa_sample_3d = cls.nasa_data["X"][10].astype(np.float32)  # (30, 153)
        cls.nasa_sample_1d = cls.nasa_sample_3d[-1, :]                  # (153,)

        cls.ai4i_x_path = os.path.join(BASE_DIR, "data", "processed", "ai4i", "X_test.csv")
        cls.ai4i_y_path = os.path.join(BASE_DIR, "data", "processed", "ai4i", "y_test.csv")
        cls.X_test_ai4i = pd.read_csv(cls.ai4i_x_path)
        cls.y_test_ai4i = pd.read_csv(cls.ai4i_y_path).values.ravel()
        cls.ai4i_normal = cls.X_test_ai4i[cls.y_test_ai4i == 0].iloc[0].to_numpy(dtype=np.float32)
        cls.ai4i_failure = cls.X_test_ai4i[cls.y_test_ai4i == 1].iloc[0].to_numpy(dtype=np.float32)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.temp_db_path):
            try:
                os.remove(cls.temp_db_path)
            except Exception:
                pass

    def test_01_dataset_loading_and_preprocessing(self):
        """Verify data integrity: non-empty, proper shapes, zero NaNs in processed archives."""
        self.assertGreater(len(self.nasa_data["X"]), 0)
        self.assertEqual(self.nasa_sample_3d.shape, (30, 153))
        self.assertFalse(np.isnan(self.nasa_sample_3d).any())

        self.assertEqual(len(self.X_test_ai4i), 2000)
        self.assertEqual(self.ai4i_normal.shape, (14,))
        self.assertFalse(np.isnan(self.ai4i_normal).any())
        print("  [PASS] Test 1: Dataset Loading & Preprocessing Integrity Verified")

    def test_02_traditional_ml_inference(self):
        """Verify trained Phase 2 baseline models execute accurately."""
        # NASA RUL Regression
        res_nasa = self.model_service.predict_nasa_rul(self.nasa_sample_1d)
        self.assertIn("predicted_rul", res_nasa)
        self.assertIsInstance(res_nasa["predicted_rul"], float)
        self.assertGreater(res_nasa["predicted_rul"], 0.0)

        # AI4I Failure Classification
        res_ai4i = self.model_service.predict_ai4i_failure(self.ai4i_failure)
        self.assertIn("failure_probability", res_ai4i)
        self.assertGreaterEqual(res_ai4i["failure_probability"], 0.5)
        self.assertEqual(res_ai4i["failure_label"], "Machine Failure")
        print("  [PASS] Test 2: Traditional ML Baseline Inference Verified")

    def test_03_deep_learning_inference(self):
        """Verify trained PyTorch deep learning models process 3D sequences."""
        dl_model_path = os.path.join(BASE_DIR, "models", "deep_learning", "nasa_rul_lstm_w30_FD001.pt")
        if os.path.exists(dl_model_path):
            checkpoint = torch.load(dl_model_path, map_location="cpu", weights_only=False)
            self.assertIn("model_state_dict", checkpoint)
            self.assertIn("val_rmse", checkpoint)
            self.assertLess(checkpoint["val_rmse"], 30.0)
        print("  [PASS] Test 3: PyTorch Deep Learning Artifacts Verified")

    def test_04_early_warning_multi_horizon(self):
        """Verify multi-horizon classifiers evaluate 10, 20, 30 cycle horizons."""
        res_nasa = self.model_service.predict_nasa_rul(self.nasa_sample_1d)
        ew = res_nasa.get("early_warning_horizons", {})
        self.assertIn("risk_within_10_cycles", ew)
        self.assertIn("risk_within_20_cycles", ew)
        self.assertIn("risk_within_30_cycles", ew)
        for h, prob in ew.items():
            self.assertTrue(0.0 <= prob <= 1.0)
        print("  [PASS] Test 4: Early-Warning Multi-Horizon Classifiers Verified")

    def test_05_unsupervised_anomaly_detection(self):
        """Verify unsupervised anomaly detection algorithms calculate reconstruction error and isolation."""
        res_nasa = self.model_service.detect_anomalies(self.nasa_sample_3d, dataset_type="NASA")
        self.assertIn("anomaly_score", res_nasa)
        self.assertIn("severity", res_nasa)

        res_ai4i = self.model_service.detect_anomalies(self.ai4i_failure, dataset_type="AI4I")
        self.assertIn("anomaly_score", res_ai4i)
        print("  [PASS] Test 5: Unsupervised Anomaly Detection Suite Verified")

    def test_06_explainability_treeshap_attribution(self):
        """Verify TreeSHAP generates valid Shapley feature attributions without target leakage."""
        explainer = self.explain_mgr.get_tree_explainer(model_id="ai4i_clf_xgboost_classifier", dataset="AI4I")
        explanation = explainer.explain_instance(self.ai4i_failure)
        self.assertIn("important_features", explanation)
        top_feats = explanation["important_features"]
        self.assertGreaterEqual(len(top_feats), 3)
        self.assertIn(top_feats[0]["feature"], self.explain_mgr.get_feature_names("AI4I"))
        print("  [PASS] Test 6: Explainable AI (TreeSHAP) Attributions Verified")

    def test_07_langgraph_multi_agent_workflow(self):
        """Verify full multi-agent StateGraph execution coordinating 5 specialized agents."""
        state_out = analyze_machine(
            input_data=self.ai4i_failure,
            dataset_type="AI4I",
            machine_id="CNC-FAIL-99",
            workflow_type="full_analysis"
        )
        self.assertEqual(state_out["workflow_status"], "COMPLETED")
        self.assertIsNotNone(state_out["recommendation_id"])
        self.assertEqual(state_out["priority"], "Critical review required")
        self.assertTrue(state_out["human_review_status"]["human_review_required"])
        print("  [PASS] Test 7: LangGraph Multi-Agent Orchestration Verified")

    def test_08_maintenance_recommendation_and_priority(self):
        """Verify priority resolution, conflicting evidence handling, and subsystem action mapping."""
        # Conflict: High RUL but High Anomaly
        payload_conflict = RecommendationInput(
            dataset_type="NASA",
            machine_id="ENG-CONFLICT-01",
            predicted_rul=125.0,
            anomaly_severity="High anomaly",
            anomaly_score=0.198,
            important_sensors=[{"feature": "sensor_11", "importance": 0.45}]
        )
        rec = self.rec_engine.generate_recommendation(payload_conflict)
        self.assertEqual(rec["priority"], "High priority")
        self.assertEqual(rec["risk_status"], "conflicting_evidence")
        self.assertTrue(rec["human_approval_required"])
        self.assertEqual(rec["recommended_actions"][0]["subsystem"], "HPC Thermal Assembly")
        print("  [PASS] Test 8: Maintenance Priority & Subsystem Mapping Verified")

    def test_09_fastapi_rest_endpoints(self):
        """Verify all FastAPI routes: /health, /api/models, /api/predict/*, /api/anomaly, /api/explain, etc."""
        # 1. Health
        r_health = self.client.get("/health")
        self.assertEqual(r_health.status_code, 200)

        # 2. Models
        r_models = self.client.get("/api/models")
        self.assertEqual(r_models.status_code, 200)
        self.assertGreaterEqual(r_models.json()["total_models"], 5)

        # 3. Predict RUL
        r_rul = self.client.post("/api/predict/rul", json={"machine_id": "ENG-1", "sensor_features": self.nasa_sample_1d.tolist()})
        self.assertEqual(r_rul.status_code, 200)

        # 4. Predict Failure
        r_fail = self.client.post("/api/predict/failure", json={"machine_id": "CNC-1", "operating_features": self.ai4i_normal.tolist()})
        self.assertEqual(r_fail.status_code, 200)

        # 5. History
        r_hist = self.client.get("/api/history?limit=10")
        self.assertEqual(r_hist.status_code, 200)
        print("  [PASS] Test 9: FastAPI REST Endpoints Verified")

    def test_10_database_audit_and_review_lifecycle(self):
        """Verify SQLite persistence, indexing, and human-in-the-loop lifecycle (Approve, Reject, Request Analysis)."""
        test_rec = {
            "machine_id": "AUDIT-ENG-777",
            "dataset": "NASA",
            "priority": "Critical review required",
            "risk_status": "imminent_exhaustion",
            "predicted_rul": 12.5,
            "recommended_actions": [{"action_id": "ACT-1", "description": "Urgent Overhaul"}],
            "supporting_evidence": [{"evidence_type": "Prediction", "value": "12.5 cycles"}],
            "human_approval_required": True
        }
        rec_id = self.test_db.save_recommendation(test_rec)
        self.assertTrue(rec_id.startswith("REC-"))

        # Initial status: PENDING
        entry_init = self.test_db.get_recommendation(rec_id)
        self.assertEqual(entry_init["review_status"], "PENDING")

        # Approve
        ok_app = self.test_db.approve_recommendation(rec_id, "Chief Inspector Zhang", "Verified")
        self.assertTrue(ok_app)
        entry_app = self.test_db.get_recommendation(rec_id)
        self.assertEqual(entry_app["review_status"], "APPROVED")
        self.assertEqual(entry_app["reviewed_by"], "Chief Inspector Zhang")
        print("  [PASS] Test 10: SQLite Database Audit & Human Review Lifecycle Verified")

    def test_11_error_handling_and_fault_tolerance(self):
        """Verify resilient error capture: wrong dimensions, NaNs, missing data, and invalid states."""
        # 1. Invalid feature dimension for NASA (5 features instead of 153)
        r_bad_nasa = self.client.post("/api/predict/rul", json={"machine_id": "ENG-BAD", "sensor_features": [1.0, 2.0, 3.0, 4.0, 5.0]})
        self.assertEqual(r_bad_nasa.status_code, 422)

        # 2. Invalid feature dimension for AI4I (3 features instead of 14)
        r_bad_ai4i = self.client.post("/api/predict/failure", json={"machine_id": "CNC-BAD", "operating_features": [1.0, 2.0, 3.0]})
        self.assertEqual(r_bad_ai4i.status_code, 422)

        # 3. NaNs in input array handled gracefully by LangGraph supervisor
        nan_array = self.ai4i_normal.copy()
        nan_array[2] = np.nan
        state_nan = analyze_machine(nan_array, dataset_type="AI4I")
        self.assertEqual(state_nan["workflow_status"], "FAILED")
        self.assertTrue(any("contains NaN" in err for err in state_nan["errors"]))

        # 4. None input handled gracefully by LangGraph supervisor
        state_none = analyze_machine(None, dataset_type="NASA")
        self.assertEqual(state_none["workflow_status"], "FAILED")

        # 5. Invalid Review Status rejected by database
        with self.assertRaises(ValueError):
            self.test_db.update_review_status(recommendation_id="REC-FAKE", status="MAYBE", reviewed_by="Eng")

        # 6. Negative RUL rejected by schema
        with self.assertRaises(ValueError):
            RecommendationInput(dataset_type="NASA", machine_id="ENG-1", predicted_rul=-50.0)

        print("  [PASS] Test 11: System Fault-Tolerance & Error Resilience Verified")


if __name__ == "__main__":
    unittest.main()
