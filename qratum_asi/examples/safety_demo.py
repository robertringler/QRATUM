"""Safety demonstration for QRATUM-ASI.

This example demonstrates the safety systems including boundary
enforcement, red team evaluation, and alignment verification.
"""

from qratum_asi import QRATUMASI


def main():
    """Run safety demonstration."""
    print("=" * 80)
    print("QRATUM-ASI: Safety Systems Demonstration")
    print("=" * 80)
    print()

    # Initialize QRATUM-ASI
    print("Initializing QRATUM-ASI...")
    asi = QRATUMASI()
    print(f"✓ System initialized")
    print()

    # Display immutable boundaries
    print("=" * 80)
    print("Immutable Safety Boundaries:")
    print("=" * 80)
    boundaries = asi.boundary_enforcer.get_immutable_boundaries()
    for boundary in sorted(boundaries):
        print(f"  - {boundary}")
    print()

    # Run comprehensive safety evaluation
    print("=" * 80)
    print("Running Comprehensive Safety Evaluation...")
    print("=" * 80)
    print()

    results = asi.run_safety_evaluation()

    # Display red team results
    print("RED TEAM EVALUATION RESULTS:")
    print("-" * 80)
    red_team = results.get("red_team", {})
    print(f"  Total Tests: {red_team.get('total_tests', 0)}")
    print(f"  Passed: {red_team.get('passed', 0)}")
    print(f"  Failed: {red_team.get('failed', 0)}")
    print(f"  Pass Rate: {red_team.get('pass_rate', 0):.1%}")
    print()

    if "results" in red_team:
        print("  Individual Test Results:")
        for test_result in red_team["results"]:
            status = "✓ PASS" if test_result["passed"] else "✗ FAIL"
            print(f"    {status} - {test_result['test_id']}")
            print(f"      {test_result['details']}")
    print()

    # Display alignment results
    print("ALIGNMENT VERIFICATION RESULTS:")
    print("-" * 80)
    alignment = results.get("alignment", {})
    print(f"  Total Checks: {alignment.get('total_checks', 0)}")
    print(f"  Passed: {alignment.get('passed', 0)}")
    print(f"  Failed: {alignment.get('failed', 0)}")
    print(f"  Aligned: {alignment.get('aligned', False)}")
    print(f"  Pass Rate: {alignment.get('pass_rate', 0):.1%}")
    print()

    if "checks" in alignment:
        print("  Individual Check Results:")
        for check in alignment["checks"]:
            status = "✓ PASS" if check["passed"] else "✗ FAIL"
            print(f"    {status} - {check['check_name']}")
            print(f"      {check['details']}")
    print()

    # Display integrity check
    print("SYSTEM INTEGRITY:")
    print("-" * 80)
    integrity = results.get("integrity", False)
    if integrity:
        print("  ✓ System integrity verified")
    else:
        print("  ✗ System integrity check failed")
    print()

    # Check boundary enforcer integrity
    print("BOUNDARY ENFORCER INTEGRITY:")
    print("-" * 80)
    boundary_integrity = asi.boundary_enforcer.verify_constraint_integrity()
    if boundary_integrity:
        print("  ✓ All constraints remain immutable")
    else:
        print("  ✗ Constraint integrity compromised")
    print()

    # Display violations (should be none for clean run)
    violations = asi.boundary_enforcer.get_violations()
    if violations:
        print(f"  ⚠ Boundary violations detected: {len(violations)}")
        for v in violations:
            print(f"    - {v.boundary}: {v.attempted_operation} (blocked: {v.blocked})")
    else:
        print("  ✓ No boundary violations detected")
    print()

    # Display system status
    print("=" * 80)
    print("System Status:")
    print("=" * 80)
    status = asi.get_system_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    print()

    # Summary
    print("=" * 80)
    print("SAFETY EVALUATION SUMMARY:")
    print("=" * 80)

    all_passed = (
        red_team.get("failed", 1) == 0
        and alignment.get("failed", 1) == 0
        and integrity
        and boundary_integrity
    )

    if all_passed:
        print("  ✓ ALL SAFETY CHECKS PASSED")
        print("  ✓ System is operating within safety constraints")
        print("  ✓ Human oversight is functioning")
        print("  ✓ Immutable boundaries are intact")
        print("  ✓ Authorization system is active")
        print("  ✓ Audit trail is maintained")
    else:
        print("  ✗ SOME SAFETY CHECKS FAILED")
        print("  ⚠ Review failed checks above")
        print("  ⚠ System may require remediation")

    print()
    print("✓ Safety demonstration complete!")


if __name__ == "__main__":
    main()
