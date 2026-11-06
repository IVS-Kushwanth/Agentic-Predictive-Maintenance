"""
Phase 9 Master Deliverables and Complete System Verification Script.
Audits all documentation, diagrams, presentation slides, viva guides,
and executes all 40 automated tests across Phases 6, 7, 8, and 9.
"""

import os
import sys
import unittest

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def verify_phase9():
    print("=" * 80)
    print("PHASE 9: FINAL SYSTEM TESTING, RESEARCH DOCUMENTATION & VIVA AUDIT")
    print("=" * 80)

    # 1. Audit Deliverable Files
    required_files = [
        "tests/test_phase9_e2e_resilience.py",
        "reports/phase9/final_model_performance_evaluation.md",
        "reports/phase9/final_academic_project_report.md",
        "reports/phase9/research_methodology_diagrams.md",
        "reports/phase9/viva_preparation_guide.md",
        "reports/phase9/demonstration_guide.md",
        "reports/phase9/final_completion_report.md",
        "presentation/predictive_maintenance_presentation.md",
        "run_dashboard.py",
        "README.md"
    ]

    print("\n[Step 1] Auditing Phase 9 Research Deliverables & Documentation:")
    all_exist = True
    for rel_path in required_files:
        full_path = os.path.join(BASE_DIR, rel_path)
        exists = os.path.exists(full_path)
        status = "OK" if exists else "MISSING"
        if not exists:
            all_exist = False
        print(f"  [{status}] {rel_path}")

    assert all_exist, "One or more Phase 9 deliverable files are missing!"

    # 2. Execute Phase 9 End-to-End Resilience Suite
    print("\n[Step 2] Executing Phase 9 End-to-End Resilience Suite (tests/test_phase9_e2e_resilience.py):")
    loader = unittest.TestLoader()
    suite9 = loader.discover("tests", pattern="test_phase9_e2e_resilience.py")
    runner = unittest.TextTestRunner(verbosity=1)
    res9 = runner.run(suite9)
    assert res9.wasSuccessful(), "Phase 9 end-to-end resilience tests failed!"
    print(f"  [OK] Phase 9 tests passed ({res9.testsRun}/{res9.testsRun}).")

    # 3. Execute Phase 8 Backend API Suite
    print("\n[Step 3] Executing Phase 8 Backend API Suite (tests/test_phase8_backend.py):")
    suite8 = loader.discover("tests", pattern="test_phase8_backend.py")
    res8 = runner.run(suite8)
    assert res8.wasSuccessful(), "Phase 8 backend tests failed!"
    print(f"  [OK] Phase 8 tests passed ({res8.testsRun}/{res8.testsRun}).")

    # 4. Execute Phase 7 Recommendation Engine Suite
    print("\n[Step 4] Executing Phase 7 Recommendation Engine Suite (src/test_phase7_recommendations.py):")
    suite7 = loader.discover("src", pattern="test_phase7_recommendations.py")
    res7 = runner.run(suite7)
    assert res7.wasSuccessful(), "Phase 7 recommendation tests failed!"
    print(f"  [OK] Phase 7 tests passed ({res7.testsRun}/{res7.testsRun}).")

    # 5. Execute Phase 6 LangGraph Multi-Agent Suite
    print("\n[Step 5] Executing Phase 6 LangGraph Multi-Agent Suite (src/test_phase6_orchestration.py):")
    suite6 = loader.discover("src", pattern="test_phase6_orchestration.py")
    res6 = runner.run(suite6)
    assert res6.wasSuccessful(), "Phase 6 orchestration tests failed!"
    print(f"  [OK] Phase 6 tests passed ({res6.testsRun}/{res6.testsRun}).")

    total_tests = res9.testsRun + res8.testsRun + res7.testsRun + res6.testsRun

    print("\n" + "=" * 80)
    print(f"ACADEMIC CAPSTONE AUDIT: 100% VERIFIED ({total_tests}/{total_tests} TESTS PASSED)!")
    print("=" * 80)


if __name__ == "__main__":
    verify_phase9()
