"""
Unit Test Suite for Phase 7: Intelligent Maintenance Recommendation Engine.
Tests all required decision scenarios:
1. Normal machine condition
2. Low anomaly condition
3. High anomaly condition
4. Low Remaining Useful Life (RUL)
5. High failure risk score
6. Conflicting model outputs (e.g., High RUL + High Anomaly)
7. Missing model evidence handling
8. Invalid input validation and error raising
9. Human-in-the-loop review workflow (Approve, Reject, Request Further Analysis)
10. End-to-end multi-agent LangGraph integration and SQLite persistence
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import tempfile
import sqlite3

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.recommendations import (
    RecommendationEngine,
    RecommendationInput,
    PriorityEngine,
    RecommendationRules
)
from src.database import RecommendationHistoryDB, init_db
from src.graph.workflow import analyze_machine


class TestPhase7RecommendationEngine(unittest.TestCase):
    """Exhaustive test suite for the Phase 7 Recommendation Engine & Decision Support."""

    @classmethod
    def setUpClass(cls):
        cls.engine = RecommendationEngine()
        cls.temp_db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        cls.temp_db_path = cls.temp_db_file.name
        cls.temp_db_file.close()
        cls.db = RecommendationHistoryDB(db_path=cls.temp_db_path)

        # Load real test samples for end-to-end verification
        cls.val_npz = np.load(os.path.join(BASE_DIR, "data", "processed", "nasa", "FD001", "sequences_w30_val.npz"))
        cls.nasa_healthy = cls.val_npz["X"][10].astype(np.float32)

        cls.X_test_ai4i = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "ai4i", "X_test.csv"))
        cls.y_test_ai4i = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "ai4i", "y_test.csv")).values.ravel()
        cls.ai4i_failure = cls.X_test_ai4i[cls.y_test_ai4i == 1].iloc[0].to_numpy(dtype=np.float32)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.temp_db_path):
            try:
                os.remove(cls.temp_db_path)
            except Exception:
                pass

    def test_01_normal_machine_condition(self):
        """Test nominal health: High RUL / Low risk, Normal anomaly -> Informational priority."""
        payload_nasa = RecommendationInput(
            dataset_type="NASA",
            machine_id="ENG-NOMINAL-101",
            predicted_rul=145.0,
            anomaly_severity="Normal",
            anomaly_score=0.012,
            important_sensors=[{"feature": "sensor_11", "importance": 0.05, "contribution": "positive"}]
        )
        rec_nasa = self.engine.generate_recommendation(payload_nasa)
        self.assertEqual(rec_nasa["priority"], "Informational")
        self.assertEqual(rec_nasa["risk_status"], "nominal")
        self.assertFalse(rec_nasa["human_approval_required"])
        self.assertGreater(len(rec_nasa["recommended_actions"]), 0)
        self.assertEqual(rec_nasa["recommended_actions"][0]["priority"], "INFO")

        payload_ai4i = RecommendationInput(
            dataset_type="AI4I",
            machine_id="CNC-NOMINAL-202",
            failure_risk_score=0.08,
            anomaly_severity="Normal",
            anomaly_score=0.015
        )
        rec_ai4i = self.engine.generate_recommendation(payload_ai4i)
        self.assertEqual(rec_ai4i["priority"], "Informational")
        self.assertEqual(rec_ai4i["risk_status"], "nominal")
        self.assertFalse(rec_ai4i["human_approval_required"])
        print("  [PASS] Test 1: Normal Machine Condition -> Informational Priority Verified")

    def test_02_low_anomaly_condition(self):
        """Test mild sensor drift: Low anomaly severity -> Low priority advisory."""
        payload = RecommendationInput(
            dataset_type="NASA",
            machine_id="ENG-DRIFT-303",
            predicted_rul=95.0,
            anomaly_severity="Low anomaly",
            anomaly_score=0.038,
            important_sensors=[{"feature": "sensor_12", "importance": 0.12}]
        )
        rec = self.engine.generate_recommendation(payload)
        self.assertEqual(rec["priority"], "Low priority")
        self.assertEqual(rec["risk_status"], "incipient_wear")
        self.assertFalse(rec["human_approval_required"])
        self.assertTrue(any("monitoring" in a["description"].lower() or "telemetry" in a["description"].lower() or "inspection" in a["description"].lower() for a in rec["recommended_actions"]))
        print("  [PASS] Test 2: Low Anomaly -> Low Priority Advisory Verified")

    def test_03_high_anomaly_condition(self):
        """Test acute sensor deviation: High anomaly severity -> Escalated priority."""
        payload = RecommendationInput(
            dataset_type="NASA",
            machine_id="ENG-ANOM-404",
            predicted_rul=65.0,
            anomaly_severity="High anomaly",
            anomaly_score=0.185,
            important_sensors=[{"feature": "sensor_11", "importance": 0.42}]
        )
        rec = self.engine.generate_recommendation(payload)
        self.assertIn(rec["priority"], ["High priority", "Critical review required"])
        self.assertTrue(rec["human_approval_required"])
        self.assertEqual(rec["recommended_actions"][0]["priority"], "HIGH")
        print("  [PASS] Test 3: High Anomaly -> High Priority & Sign-off Verified")

    def test_04_low_rul_condition(self):
        """Test impending failure: Predicted RUL <= 20 cycles -> Critical review required."""
        payload = RecommendationInput(
            dataset_type="NASA",
            machine_id="ENG-CRIT-505",
            predicted_rul=14.2,
            anomaly_severity="High anomaly",
            anomaly_score=0.210,
            important_sensors=[{"feature": "sensor_11", "importance": 0.55}]
        )
        rec = self.engine.generate_recommendation(payload)
        self.assertEqual(rec["priority"], "Critical review required")
        self.assertEqual(rec["risk_status"], "imminent_exhaustion")
        self.assertTrue(rec["human_approval_required"])
        self.assertEqual(rec["recommended_actions"][0]["priority"], "HIGH")
        self.assertIn("IMMEDIATE", rec["recommended_actions"][0]["urgency"])
        print("  [PASS] Test 4: Low RUL (<=20 cycles) -> Critical Review Required Verified")

    def test_05_high_failure_risk_condition(self):
        """Test catastrophic tool risk: AI4I failure risk >= 0.75 -> Critical review required."""
        payload = RecommendationInput(
            dataset_type="AI4I",
            machine_id="CNC-CRIT-606",
            failure_risk_score=0.88,
            anomaly_severity="High anomaly",
            anomaly_score=0.245,
            important_sensors=[{"feature": "Mechanical power", "importance": 0.62}]
        )
        rec = self.engine.generate_recommendation(payload)
        self.assertEqual(rec["priority"], "Critical review required")
        self.assertEqual(rec["risk_status"], "imminent_breakdown")
        self.assertTrue(rec["human_approval_required"])
        self.assertEqual(rec["recommended_actions"][0]["priority"], "HIGH")
        print("  [PASS] Test 5: High Failure Risk (>=0.75) -> Critical Review Required Verified")

    def test_06_conflicting_model_outputs(self):
        """Test conflicting evidence: High RUL (>100) + High Anomaly -> High priority flagged as conflicting_evidence."""
        payload = RecommendationInput(
            dataset_type="NASA",
            machine_id="ENG-CONFLICT-707",
            predicted_rul=128.5,  # Models indicate high remaining life
            anomaly_severity="High anomaly",  # But unsupervised sensors detect severe anomalous shift
            anomaly_score=0.195,
            important_sensors=[{"feature": "sensor_4", "importance": 0.38}]
        )
        rec = self.engine.generate_recommendation(payload)
        self.assertEqual(rec["priority"], "High priority")
        self.assertEqual(rec["risk_status"], "conflicting_evidence")
        self.assertTrue(rec["human_approval_required"])
        self.assertTrue(any("conflict" in r.lower() or "discrepancy" in r.lower() for r in rec["reason"]))
        print("  [PASS] Test 6: Conflicting Model Outputs -> Handled & Flagged with Human Review")

    def test_07_missing_model_evidence(self):
        """Test robustness when specific model outputs are missing without fabricating metrics."""
        # AI4I without RUL or sensors
        payload_missing = RecommendationInput(
            dataset_type="AI4I",
            machine_id="CNC-PARTIAL-808",
            failure_risk_score=0.45
            # anomaly_severity defaults to "Normal", sensors empty
        )
        rec = self.engine.generate_recommendation(payload_missing)
        self.assertIsNotNone(rec)
        self.assertIsNone(rec["predicted_rul"])
        self.assertEqual(rec["failure_risk_score"], 0.45)
        self.assertIn("Medium priority", rec["priority"])
        self.assertGreater(len(rec["recommended_actions"]), 0)

        # NASA without failure risk score
        payload_nasa_missing = RecommendationInput(
            dataset_type="NASA",
            machine_id="ENG-PARTIAL-809",
            predicted_rul=42.0
        )
        rec_nasa = self.engine.generate_recommendation(payload_nasa_missing)
        self.assertIsNone(rec_nasa["failure_risk_score"])
        self.assertEqual(rec_nasa["predicted_rul"], 42.0)
        self.assertEqual(rec_nasa["priority"], "Medium priority")
        print("  [PASS] Test 7: Missing Model Evidence Gracefully Handled Without Fabrication")

    def test_08_invalid_input_handling(self):
        """Test validation schema strictly catches invalid datasets and out-of-range metrics."""
        # Invalid dataset type
        with self.assertRaises(ValueError):
            RecommendationInput(dataset_type="TURBINE_DATASET", machine_id="M1")

        # Empty machine_id
        with self.assertRaises(ValueError):
            RecommendationInput(dataset_type="NASA", machine_id="  ")

        # Negative RUL
        with self.assertRaises(ValueError):
            RecommendationInput(dataset_type="NASA", machine_id="ENG-1", predicted_rul=-15.0)

        # Failure risk score > 1.0
        with self.assertRaises(ValueError):
            RecommendationInput(dataset_type="AI4I", machine_id="CNC-1", failure_risk_score=1.55)

        # Model confidence < 0.0
        with self.assertRaises(ValueError):
            RecommendationInput(dataset_type="NASA", machine_id="ENG-1", model_confidence=-0.1)

        print("  [PASS] Test 8: Schema Validation Catches All Invalid Inputs")

    def test_09_human_review_workflow(self):
        """Test SQLite persistence and human review status transitions (Approve, Reject, Request Further Analysis)."""
        payload = RecommendationInput(
            dataset_type="NASA",
            machine_id="ENG-REVIEW-901",
            predicted_rul=18.5,
            anomaly_severity="High anomaly"
        )
        rec = self.engine.generate_recommendation(payload)
        rec_id = self.db.save_recommendation(rec)
        self.assertIsNotNone(rec_id)
        self.assertTrue(rec_id.startswith("REC-"))

        # Initial state should be PENDING
        entry = self.db.get_recommendation(rec_id)
        self.assertIsNotNone(entry)
        self.assertEqual(entry["review_status"], "PENDING")
        self.assertIsNone(entry["reviewed_by"])

        # 1. Test Approval
        ok_approve = self.db.approve_recommendation(rec_id, reviewed_by="Lead Propulsion Eng. Miller", notes="Overhaul scheduled")
        self.assertTrue(ok_approve)
        entry_app = self.db.get_recommendation(rec_id)
        self.assertEqual(entry_app["review_status"], "APPROVED")
        self.assertEqual(entry_app["reviewed_by"], "Lead Propulsion Eng. Miller")
        self.assertEqual(entry_app["review_notes"], "Overhaul scheduled")

        # 2. Test Rejection on second record
        rec_id_2 = self.db.save_recommendation(rec)
        ok_reject = self.db.reject_recommendation(rec_id_2, reviewed_by="Chief Safety Officer Zhang", notes="Sensor glitch confirmed; unit safe")
        self.assertTrue(ok_reject)
        entry_rej = self.db.get_recommendation(rec_id_2)
        self.assertEqual(entry_rej["review_status"], "REJECTED")

        # 3. Test Request Further Analysis on third record
        rec_id_3 = self.db.save_recommendation(rec)
        ok_fa = self.db.request_further_analysis(rec_id_3, reviewed_by="Reliability Specialist Patel", notes="Request oil debris inspection")
        self.assertTrue(ok_fa)
        entry_fa = self.db.get_recommendation(rec_id_3)
        self.assertEqual(entry_fa["review_status"], "FURTHER_ANALYSIS_REQUESTED")

        # 4. Test listing history and filtering
        history_approved = self.db.list_history(review_status="APPROVED")
        self.assertTrue(any(h["recommendation_id"] == rec_id for h in history_approved))

        history_machine = self.db.list_history(machine_id="ENG-REVIEW-901")
        self.assertGreaterEqual(len(history_machine), 3)

        print("  [PASS] Test 9: Human Review Workflow (Approve, Reject, Request Analysis, History) Fully Verified")

    def test_10_end_to_end_langgraph_integration(self):
        """Test full LangGraph pipeline execution logs recommendation and returns valid recommendation_id."""
        res = analyze_machine(self.ai4i_failure, dataset_type="AI4I", machine_id="CNC-E2E-TEST")
        self.assertEqual(res["workflow_status"], "COMPLETED")
        self.assertIsNotNone(res.get("recommendation_id"))
        self.assertIn("priority", res)
        self.assertIn("risk_status", res)
        self.assertTrue(res["human_review_status"]["human_review_required"])
        self.assertGreater(len(res["suggested_maintenance_actions"]), 0)
        self.assertIn("disclaimer", res["human_review_status"])
        print("  [PASS] Test 10: End-to-End Multi-Agent Integration with Database Persistence Verified")


if __name__ == "__main__":
    unittest.main()
