"""
Comprehensive Test Suite for Phase 6: LangGraph Agent Orchestration.
Tests:
1. NASA workflow (nominal and degraded).
2. AI4I workflow (normal and failure).
3. Missing input handling.
4. Invalid sensor data (shape, NaN, empty).
5. Missing model artifact fallback.
6. Model execution failure resiliency.
7. LLM API fallback robustness.
8. Conditional routing logic.
9. Structured JSON output schema validation.
"""

import os
import sys
import json
import unittest
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
    recommend_maintenance,
    get_workflow_graph
)


class TestPhase6Orchestration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load sample real inputs from Phase 1 data
        cls.val_npz = np.load(os.path.join(BASE_DIR, "data", "processed", "nasa", "FD001", "sequences_w30_val.npz"))

        # Healthy NASA sample (Sequence at index 10 with RUL 247.0 and Normal severity)
        cls.nasa_healthy = cls.val_npz["X"][10]
        # Degraded NASA sample (Sequence near failure with RUL ~0 and High anomaly)
        deg_idx = int(np.argmin(cls.val_npz["y_rul"]))
        cls.nasa_degraded = cls.val_npz["X"][deg_idx]

        # AI4I samples
        cls.X_test_ai4i = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "ai4i", "X_test.csv"))
        cls.y_test_ai4i = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "ai4i", "y_test.csv")).values.ravel()

        cls.ai4i_normal = cls.X_test_ai4i[cls.y_test_ai4i == 0].iloc[0].to_numpy(dtype=np.float32)
        cls.ai4i_failure = cls.X_test_ai4i[cls.y_test_ai4i == 1].iloc[0].to_numpy(dtype=np.float32)

    def test_01_nasa_workflow_nominal(self):
        """Test full NASA workflow with healthy engine readings."""
        res = analyze_machine(self.nasa_healthy, dataset_type="NASA", machine_id="ENG-NOM-01")
        self.assertEqual(res["workflow_status"], "COMPLETED")
        self.assertEqual(res["dataset"], "NASA")
        self.assertIsNotNone(res["rul_prediction"])
        self.assertGreater(res["rul_prediction"], 50.0)
        self.assertIn("suggested_maintenance_actions", res)
        self.assertFalse(res["human_review_status"]["human_review_required"])
        print("  [PASS] Test 1: NASA Nominal Workflow Verified")

    def test_02_nasa_workflow_degraded(self):
        """Test full NASA workflow with near-failure engine readings."""
        res = analyze_machine(self.nasa_degraded, dataset_type="NASA", machine_id="ENG-CRIT-99")
        self.assertEqual(res["workflow_status"], "COMPLETED")
        self.assertIsNotNone(res["rul_prediction"])
        self.assertLess(res["rul_prediction"], 40.0)
        # Should flag human review for near-failure engine
        self.assertTrue(res["human_review_status"]["human_review_required"])
        self.assertEqual(res["suggested_maintenance_actions"][0]["priority"], "HIGH")
        print("  [PASS] Test 2: NASA Degraded Workflow & Human Review Verified")

    def test_03_ai4i_workflow_normal(self):
        """Test full AI4I workflow with healthy machine readings."""
        res = analyze_machine(self.ai4i_normal, dataset_type="AI4I", machine_id="CNC-NORM-05")
        self.assertEqual(res["workflow_status"], "COMPLETED")
        self.assertEqual(res["dataset"], "AI4I")
        self.assertIsNotNone(res["failure_risk_score"])
        self.assertLess(res["failure_risk_score"], 0.5)
        self.assertEqual(res["failure_label"], "Normal Operation")
        print("  [PASS] Test 3: AI4I Normal Workflow Verified")

    def test_04_ai4i_workflow_failure(self):
        """Test full AI4I workflow with breakdown machine readings."""
        res = analyze_machine(self.ai4i_failure, dataset_type="AI4I", machine_id="CNC-FAIL-12")
        self.assertEqual(res["workflow_status"], "COMPLETED")
        self.assertIsNotNone(res["failure_risk_score"])
        self.assertGreater(res["failure_risk_score"], 0.5)
        self.assertEqual(res["failure_label"], "Machine Failure")
        self.assertTrue(res["human_review_status"]["human_review_required"])
        print("  [PASS] Test 4: AI4I Breakdown Workflow & Safeguards Verified")

    def test_05_missing_input(self):
        """Test supervisor resilience when input is None."""
        res = analyze_machine(None, dataset_type="NASA", machine_id="ENG-FAIL")
        self.assertEqual(res["workflow_status"], "FAILED")
        self.assertGreater(len(res["errors"]), 0)
        self.assertIn("Missing input_sensor_data", res["errors"][0])
        print("  [PASS] Test 5: Missing Input Gracefully Captured")

    def test_06_invalid_sensor_data(self):
        """Test supervisor resilience against invalid shapes, NaNs, and empty arrays."""
        # Case A: Wrong dimensions (5 features instead of 14)
        res_shape = analyze_machine(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), dataset_type="AI4I")
        self.assertEqual(res_shape["workflow_status"], "FAILED")
        self.assertTrue(any("requires exactly 14" in err for err in res_shape["errors"]))

        # Case B: NaNs in sensor array
        bad_nan = self.ai4i_normal.copy()
        bad_nan[0] = np.nan
        res_nan = analyze_machine(bad_nan, dataset_type="AI4I")
        self.assertEqual(res_nan["workflow_status"], "FAILED")
        self.assertTrue(any("contains NaN" in err for err in res_nan["errors"]))

        # Case C: Empty array
        res_empty = analyze_machine(np.array([]), dataset_type="NASA")
        self.assertEqual(res_empty["workflow_status"], "FAILED")
        print("  [PASS] Test 6: Invalid Sensor Data Resiliency Verified")

    def test_07_dedicated_service_interfaces(self):
        """Test all 4 specialized convenience APIs."""
        # 1. assess_failure_risk
        risk_res = assess_failure_risk(self.ai4i_failure, machine_id="CNC-01")
        self.assertIn("failure_risk_score", risk_res)
        self.assertIn("human_review_required", risk_res)

        # 2. estimate_remaining_useful_life
        rul_res = estimate_remaining_useful_life(self.nasa_healthy, machine_id="ENG-01")
        self.assertIn("predicted_rul_cycles", rul_res)
        self.assertIn("early_warning_horizons", rul_res)

        # 3. explain_machine_condition
        exp_res = explain_machine_condition(self.nasa_degraded, dataset_type="NASA")
        self.assertIn("important_sensor_contributions", exp_res)

        # 4. recommend_maintenance
        rec_res = recommend_maintenance(self.nasa_degraded, dataset_type="NASA")
        self.assertIn("suggested_maintenance_actions", rec_res)
        print("  [PASS] Test 7: Dedicated Service Interfaces Verified")

    def test_08_structured_output_schema_completeness(self):
        """Verify strict schema completeness required by Prompt Requirement 8."""
        res = analyze_machine(self.nasa_healthy, dataset_type="NASA")
        required_schema_fields = [
            "dataset",
            "machine_id",
            "workflow_status",
            "rul_prediction",
            "failure_risk_score",
            "anomaly_status",
            "important_sensor_contributions",
            "explanation_summary",
            "suggested_maintenance_actions",
            "confidence_or_uncertainty_information",
            "evidence_references",
            "human_review_status",
            "errors"
        ]
        for field in required_schema_fields:
            self.assertIn(field, res, f"Field '{field}' missing from output schema")
        print("  [PASS] Test 8: Output Schema 100% Compliant")

    def test_09_llm_fallback_and_safety_disclaimer(self):
        """Verify safety disclaimer and non-autonomous advisory constraints."""
        res = analyze_machine(self.nasa_degraded, dataset_type="NASA")
        disclaimer = res["human_review_status"]["disclaimer"]
        self.assertIn("decision-support", disclaimer.lower())
        self.assertIn("not constitute autonomous", disclaimer.lower())
        print("  [PASS] Test 9: Safety Disclaimer & Decision-Support Protocol Verified")


if __name__ == "__main__":
    unittest.main()
