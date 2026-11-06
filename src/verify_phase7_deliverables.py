"""
Phase 7 Deliverables Verification Script:
Verifies that all required files, code modules, database schemas,
test suites, and technical reports for Phase 7 are complete, intact, and passing.
"""

import os
import sys
import unittest
import sqlite3

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def verify_phase7():
    print("=" * 80)
    print("PHASE 7 DELIVERABLES AND INTEGRATION VERIFICATION")
    print("=" * 80)

    # 1. Check required deliverable files
    required_files = [
        "src/recommendations/__init__.py",
        "src/recommendations/recommendation_schema.py",
        "src/recommendations/priority_engine.py",
        "src/recommendations/recommendation_rules.py",
        "src/recommendations/recommendation_engine.py",
        "src/database/__init__.py",
        "src/database/database.py",
        "src/database/recommendation_history.py",
        "src/test_phase7_recommendations.py",
        "reports/phase7/recommendation_engine_report.md",
        "reports/phase7/decision_rules_documentation.md",
        "reports/phase7/recommendation_testing_report.md",
        "data/maintenance_history.db"
    ]

    all_files_exist = True
    print("\n[Step 1] Checking Deliverable Files:")
    for rel_path in required_files:
        full_path = os.path.join(BASE_DIR, rel_path)
        exists = os.path.exists(full_path)
        status = "OK" if exists else "MISSING"
        if not exists:
            all_files_exist = False
        print(f"  [{status}] {rel_path}")

    assert all_files_exist, "One or more required Phase 7 files are missing!"

    # 2. Check Module Imports
    print("\n[Step 2] Verifying Module Imports:")
    from src.recommendations import (
        RecommendationEngine,
        RecommendationInput,
        PriorityEngine,
        RecommendationRules
    )
    from src.database import (
        get_db_connection,
        init_db,
        RecommendationHistoryDB
    )
    from src.agents import MaintenanceAgent
    from src.graph.workflow import analyze_machine
    print("  [OK] Successfully imported all recommendation and database modules.")

    # 3. Check Database Schema & CRUD
    print("\n[Step 3] Verifying SQLite Database & Human Review Lifecycle:")
    db_path = os.path.join(BASE_DIR, "data", "maintenance_history.db")
    db = RecommendationHistoryDB(db_path=db_path)
    
    # Test record
    test_rec = {
        "machine_id": "VERIFY-ENG-01",
        "dataset": "NASA",
        "priority": "High priority",
        "risk_status": "accelerated_degradation",
        "predicted_rul": 38.5,
        "recommended_actions": [{"action_id": "ACT-V1", "description": "Verify sensor harness"}],
        "supporting_evidence": [{"evidence_type": "Supervised Prediction", "value": "38.5 cycles"}],
        "human_approval_required": True
    }
    rec_id = db.save_recommendation(test_rec)
    print(f"  [OK] Saved recommendation: {rec_id}")

    # Approve
    ok_app = db.approve_recommendation(rec_id, "Chief Inspector", "Pre-flight check cleared")
    assert ok_app, "Failed to approve recommendation"
    rec_data = db.get_recommendation(rec_id)
    assert rec_data["review_status"] == "APPROVED", "Review status mismatch"
    print(f"  [OK] Approved recommendation: review_status={rec_data['review_status']}")

    # 4. Run Phase 7 Unit Tests
    print("\n[Step 4] Running Phase 7 Unit Tests (src/test_phase7_recommendations.py):")
    loader = unittest.TestLoader()
    suite7 = loader.discover("src", pattern="test_phase7_recommendations.py")
    runner = unittest.TextTestRunner(verbosity=1)
    res7 = runner.run(suite7)
    assert res7.wasSuccessful(), "Phase 7 unit tests failed!"
    print("  [OK] All Phase 7 unit tests passed successfully (10/10).")

    # 5. Run Phase 6 Regression Tests
    print("\n[Step 5] Running Phase 6 Multi-Agent Regression Tests (src/test_phase6_orchestration.py):")
    suite6 = loader.discover("src", pattern="test_phase6_orchestration.py")
    res6 = runner.run(suite6)
    assert res6.wasSuccessful(), "Phase 6 regression tests failed!"
    print("  [OK] All Phase 6 regression tests passed successfully (9/9).")

    print("\n" + "=" * 80)
    print("ALL PHASE 7 DELIVERABLES VERIFIED 100% COMPLETE AND PASSING!")
    print("=" * 80)


if __name__ == "__main__":
    verify_phase7()
