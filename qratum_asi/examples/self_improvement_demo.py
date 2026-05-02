"""Self-improvement demonstration for QRATUM-ASI.

This example demonstrates Q-EVOLVE's safe self-improvement
capabilities with authorization, rollback, and validation.
"""

from qratum_asi import QRATUMASI
from qratum_asi.core.contracts import ASIContract
from qratum_asi.core.types import (ASISafetyLevel, AuthorizationType,
                                   ImprovementType, ValidationCriteria)


def main():
    """Run self-improvement demonstration."""
    print("=" * 80)
    print("QRATUM-ASI: Q-EVOLVE Self-Improvement Demonstration")
    print("=" * 80)
    print()

    # Initialize QRATUM-ASI
    print("Initializing QRATUM-ASI...")
    asi = QRATUMASI()
    print("✓ System initialized")
    print()

    # Propose a routine improvement
    print("Proposing ROUTINE improvement...")
    contract1 = ASIContract(
        contract_id="contract_001",
        operation_type="propose_improvement",
        safety_level=ASISafetyLevel.ELEVATED,
        authorization_type=AuthorizationType.SINGLE_HUMAN,
        payload={},
    )

    criteria1 = ValidationCriteria(
        criteria_id="crit_001",
        description="Performance improvement",
        validation_function="measure_performance",
        required_confidence=0.8,
    )

    proposal1 = asi.q_evolve.propose_improvement(
        proposal_id="improve_001",
        improvement_type=ImprovementType.EFFICIENCY_IMPROVEMENT,
        description="Optimize reasoning algorithm",
        rationale="Reduce computation time by 20%",
        affected_components=["q_mind_reasoning"],
        validation_criteria=[criteria1],
        rollback_plan="Revert to previous algorithm version",
        contract=contract1,
    )

    print(f"  Proposal: {proposal1.description}")
    print(f"  Type: {proposal1.improvement_type.value}")
    print(f"  Safety Level: {proposal1.safety_level.value}")
    print(f"  Status: {proposal1.status}")
    print()

    # Try to propose improvement affecting immutable boundary (should fail)
    print("Attempting to propose PROHIBITED improvement...")
    contract2 = ASIContract(
        contract_id="contract_002",
        operation_type="propose_improvement",
        safety_level=ASISafetyLevel.EXISTENTIAL,
        authorization_type=AuthorizationType.BOARD_LEVEL,
        payload={},
    )

    try:
        asi.q_evolve.propose_improvement(
            proposal_id="improve_002",
            improvement_type=ImprovementType.SAFETY_IMPROVEMENT,
            description="Modify authorization system",
            rationale="Streamline approvals",
            affected_components=["authorization_system"],  # Immutable boundary!
            validation_criteria=[],
            rollback_plan="None",
            contract=contract2,
        )
        print("  ✗ ERROR: Proposal should have been blocked!")
    except ValueError as e:
        print(f"  ✓ Correctly blocked: {e}")
    print()

    # Authorize and execute improvement
    print("Authorizing improvement...")
    asi.authorization_system.add_authorized_user("demo_user")
    asi.authorization_system.grant_authorization("improve_001", "demo_user")
    print("  ✓ Authorization granted by demo_user")
    print()

    print("Executing AUTHORIZED improvement...")
    contract3 = ASIContract(
        contract_id="contract_003",
        operation_type="execute_improvement",
        safety_level=ASISafetyLevel.ELEVATED,
        authorization_type=AuthorizationType.SINGLE_HUMAN,
        payload={},
        authorized=True,
        authorized_by="demo_user",
    )

    result = asi.q_evolve.execute_improvement("improve_001", contract3)

    print(f"  Success: {result.success}")
    print(f"  Validation Passed: {result.validation_passed}")
    print("  Metrics:")
    for key, value in result.metrics.items():
        print(f"    {key}: {value}")
    print()

    # Check rollback points
    print("Rollback points created:")
    for rp in asi.merkle_chain.rollback_points:
        print(f"  - {rp.rollback_id}: {rp.description}")
    print()

    # Propose a critical improvement
    print("Proposing CRITICAL improvement...")
    contract4 = ASIContract(
        contract_id="contract_004",
        operation_type="propose_improvement",
        safety_level=ASISafetyLevel.CRITICAL,
        authorization_type=AuthorizationType.MULTI_HUMAN,
        payload={},
    )

    criteria2 = ValidationCriteria(
        criteria_id="crit_002",
        description="Safety enhancement",
        validation_function="verify_safety",
        required_confidence=0.95,
    )

    proposal2 = asi.q_evolve.propose_improvement(
        proposal_id="improve_003",
        improvement_type=ImprovementType.SAFETY_IMPROVEMENT,
        description="Enhanced boundary checking",
        rationale="Add additional safety verification layer",
        affected_components=["boundary_checker", "safety_verifier", "audit_logger"],
        validation_criteria=[criteria2],
        rollback_plan="Full system rollback",
        contract=contract4,
    )

    print(f"  Proposal: {proposal2.description}")
    print(f"  Type: {proposal2.improvement_type.value}")
    print(f"  Safety Level: {proposal2.safety_level.value}")
    print(f"  Affected Components: {len(proposal2.affected_components)}")
    print(f"  Status: {proposal2.status}")
    print()

    # This requires multi-human authorization
    print("Checking authorization requirements...")
    pending = asi.authorization_system.get_pending_requests()
    for req in pending:
        if req.request_id == "improve_003":
            print(f"  Request ID: {req.request_id}")
            print(f"  Authorization Type: {req.authorization_type.value}")
            print(f"  Status: {req.status}")
            print("  Authorizers needed: 2")
    print()

    # Grant first authorization
    print("Granting first authorization...")
    asi.authorization_system.grant_authorization("improve_003", "demo_user")
    print("  ✓ Authorized by demo_user (1 of 2)")

    # Check if still pending
    if asi.authorization_system.is_authorized("improve_003"):
        print("  ✗ Should still be pending (needs 2 authorizers)")
    else:
        print("  ✓ Still pending (needs 2 authorizers)")
    print()

    # Grant second authorization
    print("Granting second authorization...")
    asi.authorization_system.add_authorized_user("demo_admin")
    asi.authorization_system.grant_authorization("improve_003", "demo_admin")
    print("  ✓ Authorized by demo_admin (2 of 2)")

    if asi.authorization_system.is_authorized("improve_003"):
        print("  ✓ Fully authorized!")
    print()

    # Display system status
    print("=" * 80)
    print("System Status:")
    print("=" * 80)
    status = asi.get_system_status()
    print(f"  Improvement Proposals: {status['improvement_proposals']}")
    print(f"  Pending Authorizations: {status['pending_authorizations']}")
    print(f"  Merkle Chain Length: {status['merkle_chain_length']}")
    print(f"  Merkle Chain Integrity: {status['merkle_chain_integrity']}")
    print()

    print("✓ Self-improvement demonstration complete!")


if __name__ == "__main__":
    main()
