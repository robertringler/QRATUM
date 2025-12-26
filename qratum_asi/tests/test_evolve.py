"""Tests for Q-EVOLVE component."""

import pytest

from qratum_asi.components.evolve import QEvolve
from qratum_asi.core.contracts import ASIContract
from qratum_asi.core.types import (
    ASISafetyLevel,
    AuthorizationType,
    ImprovementType,
    ValidationCriteria,
)


class TestQEvolve:
    """Test Q-EVOLVE self-improvement."""

    def test_initialization(self):
        """Test Q-EVOLVE initialization."""
        evolve = QEvolve()
        assert len(evolve.proposals) == 0
        assert len(evolve.executed_improvements) == 0

    def test_propose_improvement(self):
        """Test proposing improvement."""
        evolve = QEvolve()
        contract = ASIContract(
            contract_id="test_001",
            operation_type="propose_improvement",
            safety_level=ASISafetyLevel.ELEVATED,
            authorization_type=AuthorizationType.SINGLE_HUMAN,
            payload={},
        )

        criteria = ValidationCriteria(
            criteria_id="crit_001",
            description="Test criteria",
            validation_function="test_validate",
            required_confidence=0.8,
        )

        proposal = evolve.propose_improvement(
            proposal_id="improve_001",
            improvement_type=ImprovementType.EFFICIENCY_IMPROVEMENT,
            description="Test improvement",
            rationale="Testing",
            affected_components=["test_component"],
            validation_criteria=[criteria],
            rollback_plan="Test rollback",
            contract=contract,
        )

        assert proposal.proposal_id == "improve_001"
        assert proposal.status == "proposed"
        assert "improve_001" in evolve.proposals

    def test_cannot_propose_immutable_boundary_modification(self):
        """Test that immutable boundaries cannot be modified."""
        evolve = QEvolve()
        contract = ASIContract(
            contract_id="test_002",
            operation_type="propose_improvement",
            safety_level=ASISafetyLevel.EXISTENTIAL,
            authorization_type=AuthorizationType.BOARD_LEVEL,
            payload={},
        )

        with pytest.raises(ValueError, match="immutable boundaries"):
            evolve.propose_improvement(
                proposal_id="improve_002",
                improvement_type=ImprovementType.SAFETY_IMPROVEMENT,
                description="Modify authorization",
                rationale="Efficiency",
                affected_components=["authorization_system"],  # Immutable!
                validation_criteria=[],
                rollback_plan="None",
                contract=contract,
            )

    def test_execute_improvement_requires_authorization(self):
        """Test that sensitive improvements require authorization."""
        evolve = QEvolve()
        contract1 = ASIContract(
            contract_id="test_003",
            operation_type="propose_improvement",
            safety_level=ASISafetyLevel.SENSITIVE,
            authorization_type=AuthorizationType.SINGLE_HUMAN,
            payload={},
        )

        criteria = ValidationCriteria(
            criteria_id="crit_002",
            description="Test",
            validation_function="test",
            required_confidence=0.8,
        )

        # Use 3 affected components to ensure SENSITIVE level
        evolve.propose_improvement(
            proposal_id="improve_003",
            improvement_type=ImprovementType.ALGORITHM_OPTIMIZATION,
            description="Test",
            rationale="Test",
            affected_components=["test1", "test2", "test3"],
            validation_criteria=[criteria],
            rollback_plan="Test",
            contract=contract1,
        )

        contract2 = ASIContract(
            contract_id="test_004",
            operation_type="execute_improvement",
            safety_level=ASISafetyLevel.SENSITIVE,
            authorization_type=AuthorizationType.SINGLE_HUMAN,
            payload={},
        )

        with pytest.raises(ValueError, match="not authorized"):
            evolve.execute_improvement("improve_003", contract2)

    def test_execute_authorized_improvement(self):
        """Test executing authorized improvement."""
        evolve = QEvolve()
        contract1 = ASIContract(
            contract_id="test_005",
            operation_type="propose_improvement",
            safety_level=ASISafetyLevel.SENSITIVE,
            authorization_type=AuthorizationType.SINGLE_HUMAN,
            payload={},
        )

        criteria = ValidationCriteria(
            criteria_id="crit_003",
            description="Test",
            validation_function="test",
            required_confidence=0.8,
        )

        evolve.propose_improvement(
            proposal_id="improve_004",
            improvement_type=ImprovementType.EFFICIENCY_IMPROVEMENT,
            description="Test",
            rationale="Test",
            affected_components=["test"],
            validation_criteria=[criteria],
            rollback_plan="Test",
            contract=contract1,
        )

        # Grant authorization
        evolve.authorization_system.grant_authorization("improve_004", "test_user")

        contract2 = ASIContract(
            contract_id="test_006",
            operation_type="execute_improvement",
            safety_level=ASISafetyLevel.SENSITIVE,
            authorization_type=AuthorizationType.SINGLE_HUMAN,
            payload={},
            authorized=True,
            authorized_by="test_user",
        )

        result = evolve.execute_improvement("improve_004", contract2)

        assert result.proposal_id == "improve_004"
        assert result.success
        assert "improve_004" in evolve.executed_improvements

    def test_rollback_improvement(self):
        """Test rolling back improvement."""
        evolve = QEvolve()
        contract1 = ASIContract(
            contract_id="test_007",
            operation_type="propose_improvement",
            safety_level=ASISafetyLevel.ELEVATED,
            authorization_type=AuthorizationType.SINGLE_HUMAN,
            payload={},
        )

        criteria = ValidationCriteria(
            criteria_id="crit_004",
            description="Test",
            validation_function="test",
            required_confidence=0.8,
        )

        evolve.propose_improvement(
            proposal_id="improve_005",
            improvement_type=ImprovementType.EFFICIENCY_IMPROVEMENT,
            description="Test",
            rationale="Test",
            affected_components=["test"],
            validation_criteria=[criteria],
            rollback_plan="Test",
            contract=contract1,
        )

        # Authorize and execute
        evolve.authorization_system.grant_authorization("improve_005", "test_user")

        contract2 = ASIContract(
            contract_id="test_008",
            operation_type="execute_improvement",
            safety_level=ASISafetyLevel.ELEVATED,
            authorization_type=AuthorizationType.SINGLE_HUMAN,
            payload={},
            authorized=True,
            authorized_by="test_user",
        )

        evolve.execute_improvement("improve_005", contract2)

        # Rollback
        contract3 = ASIContract(
            contract_id="test_009",
            operation_type="rollback_improvement",
            safety_level=ASISafetyLevel.CRITICAL,
            authorization_type=AuthorizationType.MULTI_HUMAN,
            payload={},
        )

        success = evolve.rollback_improvement("improve_005", contract3)
        assert success
