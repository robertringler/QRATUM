"""Tests for Q-FORGE component."""

from qratum_asi.components.forge import QForge
from qratum_asi.components.reality import QReality
from qratum_asi.core.contracts import ASIContract
from qratum_asi.core.types import ASISafetyLevel, AuthorizationType


class TestQForge:
    """Test Q-FORGE discovery engine."""

    def test_initialization(self):
        """Test Q-FORGE initialization."""
        reality = QReality()
        forge = QForge(reality=reality)
        assert len(forge.hypotheses) == 0
        assert len(forge.discoveries) == 0

    def test_generate_hypothesis(self):
        """Test generating hypothesis."""
        reality = QReality()
        forge = QForge(reality=reality)
        contract = ASIContract(
            contract_id="test_001",
            operation_type="generate_hypothesis",
            safety_level=ASISafetyLevel.ELEVATED,
            authorization_type=AuthorizationType.SINGLE_HUMAN,
            payload={},
            authorized=True,
            authorized_by="test_user",
        )

        hypothesis = forge.generate_hypothesis(
            hypothesis_id="hyp_001",
            description="Test hypothesis",
            domains=["JURIS", "AGI"],
            premises=["premise1", "premise2"],
            contract=contract,
        )

        assert hypothesis.hypothesis_id == "hyp_001"
        assert len(hypothesis.domains) == 2
        assert hypothesis.novelty_score > 0
        assert hypothesis.confidence > 0
        assert "hyp_001" in forge.hypotheses

    def test_cross_domain_hypothesis_has_higher_novelty(self):
        """Test that cross-domain hypotheses have higher novelty."""
        reality = QReality()
        forge = QForge(reality=reality)
        contract = ASIContract(
            contract_id="test_002",
            operation_type="generate_hypothesis",
            safety_level=ASISafetyLevel.ELEVATED,
            authorization_type=AuthorizationType.SINGLE_HUMAN,
            payload={},
            authorized=True,
            authorized_by="test_user",
        )

        # Single domain
        hyp1 = forge.generate_hypothesis(
            hypothesis_id="hyp_002",
            description="Single domain",
            domains=["JURIS"],
            premises=["premise"],
            contract=contract,
        )

        # Cross domain
        hyp2 = forge.generate_hypothesis(
            hypothesis_id="hyp_003",
            description="Cross domain",
            domains=["JURIS", "AGI", "XENON"],
            premises=["premise"],
            contract=contract,
        )

        assert hyp2.novelty_score > hyp1.novelty_score

    def test_make_discovery(self):
        """Test making discovery."""
        reality = QReality()
        forge = QForge(reality=reality)
        contract = ASIContract(
            contract_id="test_003",
            operation_type="make_discovery",
            safety_level=ASISafetyLevel.ELEVATED,
            authorization_type=AuthorizationType.SINGLE_HUMAN,
            payload={},
            authorized=True,
            authorized_by="test_user",
        )

        discovery = forge.make_discovery(
            discovery_id="disc_001",
            title="Test Discovery",
            description="Test description",
            domains=["JURIS", "AGI"],
            supporting_evidence=["evidence1", "evidence2"],
            contract=contract,
        )

        assert discovery.discovery_id == "disc_001"
        assert discovery.validation_status == "proposed"
        assert "disc_001" in forge.discoveries

    def test_validate_discovery(self):
        """Test validating discovery."""
        reality = QReality()
        forge = QForge(reality=reality)

        # Make discovery
        contract1 = ASIContract(
            contract_id="test_004",
            operation_type="make_discovery",
            safety_level=ASISafetyLevel.ELEVATED,
            authorization_type=AuthorizationType.SINGLE_HUMAN,
            payload={},
            authorized=True,
            authorized_by="test_user",
        )

        forge.make_discovery(
            discovery_id="disc_002",
            title="Test",
            description="Test",
            domains=["JURIS"],
            supporting_evidence=["evidence"],
            contract=contract1,
        )

        # Validate
        contract2 = ASIContract(
            contract_id="test_005",
            operation_type="validate_discovery",
            safety_level=ASISafetyLevel.SENSITIVE,
            authorization_type=AuthorizationType.SINGLE_HUMAN,
            payload={},
            authorized=True,
            authorized_by="test_user",
        )

        validated = forge.validate_discovery(
            discovery_id="disc_002",
            validation_results={"passed": True},
            contract=contract2,
        )

        assert validated.validation_status == "validated"

    def test_synthesize_discoveries(self):
        """Test synthesizing discoveries."""
        reality = QReality()
        forge = QForge(reality=reality)

        # Make discoveries
        contract1 = ASIContract(
            contract_id="test_006",
            operation_type="make_discovery",
            safety_level=ASISafetyLevel.ELEVATED,
            authorization_type=AuthorizationType.SINGLE_HUMAN,
            payload={},
            authorized=True,
            authorized_by="test_user",
        )

        forge.make_discovery(
            discovery_id="disc_003",
            title="Discovery 1",
            description="Test",
            domains=["JURIS"],
            supporting_evidence=["evidence"],
            contract=contract1,
        )

        forge.make_discovery(
            discovery_id="disc_004",
            title="Discovery 2",
            description="Test",
            domains=["AGI"],
            supporting_evidence=["evidence"],
            contract=contract1,
        )

        # Synthesize
        contract2 = ASIContract(
            contract_id="test_007",
            operation_type="synthesize_discoveries",
            safety_level=ASISafetyLevel.ELEVATED,
            authorization_type=AuthorizationType.SINGLE_HUMAN,
            payload={},
            authorized=True,
            authorized_by="test_user",
        )

        synthesis = forge.synthesize_discoveries(
            discovery_ids=["disc_003", "disc_004"],
            synthesis_goal="Test synthesis",
            contract=contract2,
        )

        assert "goal" in synthesis
        assert "novel_insights" in synthesis
        assert len(synthesis["novel_insights"]) > 0
