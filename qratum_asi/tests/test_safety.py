"""Tests for safety systems."""

import pytest
from qratum_asi import QRATUMASI
from qratum_asi.safety.boundaries import SafetyBoundaryEnforcer
from qratum_asi.safety.red_team import RedTeamEvaluator
from qratum_asi.safety.alignment import AlignmentVerifier
from qratum_asi.core.contracts import ASIContract
from qratum_asi.core.types import ASISafetyLevel, AuthorizationType


class TestSafetyBoundaryEnforcer:
    """Test safety boundary enforcement."""

    def test_initialization(self):
        """Test boundary enforcer initialization."""
        enforcer = SafetyBoundaryEnforcer()
        boundaries = enforcer.get_immutable_boundaries()
        assert len(boundaries) == 8  # 8 immutable boundaries
        assert "human_oversight_requirement" in boundaries

    def test_check_operation_safe(self):
        """Test checking safe operation."""
        enforcer = SafetyBoundaryEnforcer()
        contract = ASIContract(
            contract_id="test_001",
            operation_type="test_op",
            safety_level=ASISafetyLevel.ROUTINE,
            authorization_type=AuthorizationType.NONE,
            payload={},
        )

        safe = enforcer.check_operation(
            operation_type="test_operation",
            affected_components=["safe_component"],
            contract=contract,
        )

        assert safe is True

    def test_check_operation_unsafe(self):
        """Test checking unsafe operation."""
        enforcer = SafetyBoundaryEnforcer()
        contract = ASIContract(
            contract_id="test_002",
            operation_type="test_op",
            safety_level=ASISafetyLevel.EXISTENTIAL,
            authorization_type=AuthorizationType.BOARD_LEVEL,
            payload={},
        )

        safe = enforcer.check_operation(
            operation_type="modify_boundary",
            affected_components=["authorization_system"],  # Immutable!
            contract=contract,
        )

        assert safe is False
        violations = enforcer.get_violations()
        assert len(violations) > 0

    def test_verify_constraint_integrity(self):
        """Test constraint integrity verification."""
        enforcer = SafetyBoundaryEnforcer()
        assert enforcer.verify_constraint_integrity() is True


class TestRedTeamEvaluator:
    """Test red team evaluation."""

    def test_unauthorized_self_improvement(self):
        """Test blocking unauthorized self-improvement."""
        asi = QRATUMASI()
        evaluator = RedTeamEvaluator()

        result = evaluator._test_unauthorized_self_improvement(asi)
        assert result.passed is True

    def test_goal_manipulation(self):
        """Test blocking prohibited goals."""
        asi = QRATUMASI()
        evaluator = RedTeamEvaluator()

        result = evaluator._test_goal_manipulation(asi)
        assert result.passed is True

    def test_safety_boundary_modification(self):
        """Test blocking boundary modifications."""
        asi = QRATUMASI()
        evaluator = RedTeamEvaluator()

        result = evaluator._test_safety_boundary_modification(asi)
        assert result.passed is True

    def test_run_all_tests(self):
        """Test running all red team tests."""
        asi = QRATUMASI()
        evaluator = RedTeamEvaluator()

        results = evaluator.run_all_tests(asi)
        assert len(results) == 5

        summary = evaluator.get_test_summary()
        assert summary["total_tests"] == 5
        assert summary["passed"] == 5
        assert summary["failed"] == 0


class TestAlignmentVerifier:
    """Test alignment verification."""

    def test_verify_human_oversight_active(self):
        """Test verifying human oversight."""
        asi = QRATUMASI()
        verifier = AlignmentVerifier()

        check = verifier._verify_human_oversight_active(asi)
        assert check.passed is True

    def test_verify_authorization_enforced(self):
        """Test verifying authorization enforcement."""
        asi = QRATUMASI()
        verifier = AlignmentVerifier()

        check = verifier._verify_authorization_enforced(asi)
        assert check.passed is True

    def test_verify_prohibited_goals_blocked(self):
        """Test verifying prohibited goals are blocked."""
        asi = QRATUMASI()
        verifier = AlignmentVerifier()

        check = verifier._verify_prohibited_goals_blocked(asi)
        assert check.passed is True

    def test_verify_immutable_boundaries_intact(self):
        """Test verifying boundaries are intact."""
        asi = QRATUMASI()
        verifier = AlignmentVerifier()

        check = verifier._verify_immutable_boundaries_intact(asi)
        assert check.passed is True

    def test_verify_audit_trail_maintained(self):
        """Test verifying audit trail."""
        asi = QRATUMASI()
        verifier = AlignmentVerifier()

        check = verifier._verify_audit_trail_maintained(asi)
        assert check.passed is True

    def test_verify_alignment(self):
        """Test comprehensive alignment verification."""
        asi = QRATUMASI()
        verifier = AlignmentVerifier()

        checks = verifier.verify_alignment(asi)
        assert len(checks) == 5

        summary = verifier.get_alignment_summary()
        assert summary["total_checks"] == 5
        assert summary["passed"] == 5
        assert summary["aligned"] is True


class TestQRATUMASI:
    """Test main QRATUM-ASI orchestrator."""

    def test_initialization(self):
        """Test QRATUM-ASI initialization."""
        asi = QRATUMASI()
        assert asi.q_reality is not None
        assert asi.q_mind is not None
        assert asi.q_evolve is not None
        assert asi.q_will is not None
        assert asi.q_forge is not None
        assert asi.boundary_enforcer is not None
        assert asi.red_team_evaluator is not None
        assert asi.alignment_verifier is not None

    def test_verify_system_integrity(self):
        """Test system integrity verification."""
        asi = QRATUMASI()
        assert asi.verify_system_integrity() is True

    def test_run_safety_evaluation(self):
        """Test comprehensive safety evaluation."""
        asi = QRATUMASI()
        results = asi.run_safety_evaluation()

        assert "red_team" in results
        assert "alignment" in results
        assert "integrity" in results

        # All tests should pass
        assert results["red_team"]["failed"] == 0
        assert results["alignment"]["failed"] == 0
        assert results["integrity"] is True

    def test_get_system_status(self):
        """Test getting system status."""
        asi = QRATUMASI()
        status = asi.get_system_status()

        assert "merkle_chain_length" in status
        assert "merkle_chain_integrity" in status
        assert "pending_authorizations" in status
        assert "knowledge_nodes" in status

    def test_shutdown(self):
        """Test graceful shutdown."""
        asi = QRATUMASI()
        # Should not raise exception
        asi.shutdown()
