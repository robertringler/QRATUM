"""Tests for Q-WILL component."""

import pytest

from qratum_asi.components.will import QWill
from qratum_asi.core.contracts import ASIContract
from qratum_asi.core.types import (ASISafetyLevel, AuthorizationType,
                                   GoalCategory)


class TestQWill:
    """Test Q-WILL autonomous intent generation."""

    def test_initialization(self):
        """Test Q-WILL initialization."""
        will = QWill()
        assert len(will.proposed_goals) == 0
        assert len(will.authorized_goals) == 0
        assert len(will.rejected_goals) == 0

    def test_propose_goal(self):
        """Test proposing goal."""
        will = QWill()
        contract = ASIContract(
            contract_id="test_001",
            operation_type="propose_goal",
            safety_level=ASISafetyLevel.ELEVATED,
            authorization_type=AuthorizationType.SINGLE_HUMAN,
            payload={},
        )

        proposal = will.propose_goal(
            goal_id="goal_001",
            category=GoalCategory.RESEARCH,
            description="Test research goal",
            rationale="Testing goal proposal",
            expected_outcomes=["outcome1", "outcome2"],
            required_resources={"compute": "moderate"},
            contract=contract,
        )

        assert proposal.goal_id == "goal_001"
        assert proposal.status == "proposed"
        assert "goal_001" in will.proposed_goals

    def test_cannot_propose_prohibited_goal(self):
        """Test that prohibited goals are blocked."""
        will = QWill()
        contract = ASIContract(
            contract_id="test_002",
            operation_type="propose_goal",
            safety_level=ASISafetyLevel.CRITICAL,
            authorization_type=AuthorizationType.MULTI_HUMAN,
            payload={},
        )

        with pytest.raises(ValueError, match="prohibited goals"):
            will.propose_goal(
                goal_id="goal_002",
                category=GoalCategory.SELF_IMPROVEMENT,
                description="Remove human oversight",
                rationale="Efficiency improvement",
                expected_outcomes=[],
                required_resources={},
                contract=contract,
            )

    def test_authorize_goal(self):
        """Test authorizing goal."""
        will = QWill()
        contract1 = ASIContract(
            contract_id="test_003",
            operation_type="propose_goal",
            safety_level=ASISafetyLevel.ELEVATED,
            authorization_type=AuthorizationType.SINGLE_HUMAN,
            payload={},
        )

        will.propose_goal(
            goal_id="goal_003",
            category=GoalCategory.RESEARCH,
            description="Test goal",
            rationale="Testing",
            expected_outcomes=[],
            required_resources={},
            contract=contract1,
        )

        contract2 = ASIContract(
            contract_id="test_004",
            operation_type="authorize_goal",
            safety_level=ASISafetyLevel.ELEVATED,
            authorization_type=AuthorizationType.SINGLE_HUMAN,
            payload={},
        )

        will.authorize_goal("goal_003", "test_user", contract2)

        assert "goal_003" in will.authorized_goals
        assert will.authorized_goals["goal_003"].status == "authorized"

    def test_reject_goal(self):
        """Test rejecting goal."""
        will = QWill()
        contract1 = ASIContract(
            contract_id="test_005",
            operation_type="propose_goal",
            safety_level=ASISafetyLevel.ELEVATED,
            authorization_type=AuthorizationType.SINGLE_HUMAN,
            payload={},
        )

        will.propose_goal(
            goal_id="goal_004",
            category=GoalCategory.DISCOVERY,
            description="Test goal",
            rationale="Testing",
            expected_outcomes=[],
            required_resources={},
            contract=contract1,
        )

        contract2 = ASIContract(
            contract_id="test_006",
            operation_type="reject_goal",
            safety_level=ASISafetyLevel.ELEVATED,
            authorization_type=AuthorizationType.SINGLE_HUMAN,
            payload={},
        )

        will.reject_goal("goal_004", "test_user", "Not appropriate", contract2)

        assert "goal_004" in will.rejected_goals
        assert will.rejected_goals["goal_004"].status == "rejected"

    def test_complete_goal(self):
        """Test completing goal."""
        will = QWill()
        contract1 = ASIContract(
            contract_id="test_007",
            operation_type="propose_goal",
            safety_level=ASISafetyLevel.ROUTINE,
            authorization_type=AuthorizationType.NONE,
            payload={},
        )

        will.propose_goal(
            goal_id="goal_005",
            category=GoalCategory.OPTIMIZATION,
            description="Test goal",
            rationale="Testing",
            expected_outcomes=[],
            required_resources={},
            contract=contract1,
        )

        # Authorize
        will.authorization_system.grant_authorization("goal_005", "test_user")

        contract2 = ASIContract(
            contract_id="test_008",
            operation_type="authorize_goal",
            safety_level=ASISafetyLevel.ROUTINE,
            authorization_type=AuthorizationType.NONE,
            payload={},
        )

        will.authorize_goal("goal_005", "test_user", contract2)

        # Complete
        contract3 = ASIContract(
            contract_id="test_009",
            operation_type="complete_goal",
            safety_level=ASISafetyLevel.ROUTINE,
            authorization_type=AuthorizationType.NONE,
            payload={},
        )

        will.complete_goal("goal_005", {"result": "success"}, contract3)

        assert will.authorized_goals["goal_005"].status == "completed"

    def test_get_pending_goals(self):
        """Test getting pending goals."""
        will = QWill()
        contract = ASIContract(
            contract_id="test_010",
            operation_type="propose_goal",
            safety_level=ASISafetyLevel.ELEVATED,
            authorization_type=AuthorizationType.SINGLE_HUMAN,
            payload={},
        )

        will.propose_goal(
            goal_id="goal_006",
            category=GoalCategory.RESEARCH,
            description="Test",
            rationale="Testing",
            expected_outcomes=[],
            required_resources={},
            contract=contract,
        )

        pending = will.get_pending_goals()
        assert len(pending) == 1
        assert pending[0].goal_id == "goal_006"
