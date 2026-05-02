"""Tests for Q-MIND component."""

from qratum_asi.components.mind import QMind
from qratum_asi.components.reality import QReality
from qratum_asi.core.contracts import ASIContract
from qratum_asi.core.types import (ASISafetyLevel, AuthorizationType,
                                   ReasoningStrategy)


class TestQMind:
    """Test Q-MIND reasoning core."""

    def test_initialization(self):
        """Test Q-MIND initialization."""
        reality = QReality()
        mind = QMind(reality=reality)
        assert len(mind.reasoning_chains) == 0
        assert mind.reality == reality

    def test_deductive_reasoning(self):
        """Test deductive reasoning."""
        reality = QReality()
        mind = QMind(reality=reality)
        contract = ASIContract(
            contract_id="test_001",
            operation_type="reasoning",
            safety_level=ASISafetyLevel.ROUTINE,
            authorization_type=AuthorizationType.NONE,
            payload={},
        )

        chain = mind.reason(
            query="Test query",
            strategy=ReasoningStrategy.DEDUCTIVE,
            context={},
            contract=contract,
        )

        assert chain.query == "Test query"
        assert len(chain.steps) > 0
        assert chain.steps[0].strategy == ReasoningStrategy.DEDUCTIVE
        assert chain.overall_confidence > 0

    def test_causal_reasoning(self):
        """Test causal reasoning."""
        reality = QReality()
        mind = QMind(reality=reality)
        contract = ASIContract(
            contract_id="test_002",
            operation_type="reasoning",
            safety_level=ASISafetyLevel.ROUTINE,
            authorization_type=AuthorizationType.NONE,
            payload={},
        )

        chain = mind.reason(
            query="What causes X?",
            strategy=ReasoningStrategy.CAUSAL,
            context={},
            contract=contract,
        )

        assert chain.steps[0].strategy == ReasoningStrategy.CAUSAL
        assert len(chain.steps) > 0

    def test_bayesian_reasoning(self):
        """Test Bayesian reasoning."""
        reality = QReality()
        mind = QMind(reality=reality)
        contract = ASIContract(
            contract_id="test_003",
            operation_type="reasoning",
            safety_level=ASISafetyLevel.ROUTINE,
            authorization_type=AuthorizationType.NONE,
            payload={},
        )

        chain = mind.reason(
            query="Update belief given evidence",
            strategy=ReasoningStrategy.BAYESIAN,
            context={},
            contract=contract,
        )

        assert chain.steps[0].strategy == ReasoningStrategy.BAYESIAN

    def test_cross_domain_synthesis(self):
        """Test cross-domain synthesis."""
        reality = QReality()
        mind = QMind(reality=reality)

        # Add some knowledge to reality
        contract1 = ASIContract(
            contract_id="test_004",
            operation_type="add_knowledge",
            safety_level=ASISafetyLevel.ROUTINE,
            authorization_type=AuthorizationType.NONE,
            payload={},
        )

        reality.add_knowledge_node(
            node_id="node_001",
            content={"statement": "Test"},
            source_vertical="JURIS",
            confidence=0.9,
            provenance=["test"],
            contract=contract1,
        )

        # Perform synthesis
        contract2 = ASIContract(
            contract_id="test_005",
            operation_type="synthesis",
            safety_level=ASISafetyLevel.ELEVATED,
            authorization_type=AuthorizationType.SINGLE_HUMAN,
            payload={},
            authorized=True,
            authorized_by="test_user",
        )

        synthesis = mind.cross_domain_synthesis(
            domains=["JURIS", "AGI"],
            synthesis_goal="Test synthesis",
            contract=contract2,
        )

        assert "goal" in synthesis
        assert "domains" in synthesis
        assert "insights" in synthesis
        assert len(synthesis["domains"]) == 2

    def test_reasoning_creates_merkle_events(self):
        """Test that reasoning creates merkle chain events."""
        reality = QReality()
        mind = QMind(reality=reality)
        contract = ASIContract(
            contract_id="test_006",
            operation_type="reasoning",
            safety_level=ASISafetyLevel.ROUTINE,
            authorization_type=AuthorizationType.NONE,
            payload={},
        )

        initial_length = mind.merkle_chain.get_chain_length()

        mind.reason(
            query="Test",
            strategy=ReasoningStrategy.INDUCTIVE,
            context={},
            contract=contract,
        )

        # Should create at least 2 events (started and completed)
        assert mind.merkle_chain.get_chain_length() >= initial_length + 2
