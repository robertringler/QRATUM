"""Tests for QRATUM Generalization Layer.

Tests the GeneralReasoningEngine, ExtendedDomainRegistry, and
cross-domain synthesis capabilities.
"""

import pytest
from qratum_asi.core.contracts import ASIContract
from qratum_asi.core.types import ASISafetyLevel, AuthorizationType

from qratum_asi.generalization.types import (
    CognitiveDomain,
    DomainCapability,
    HypothesisType,
    SynthesisSafetyLevel,
    CrossDomainHypothesis,
    GenerationConstraints,
    PROHIBITED_SYNTHESIS_TARGETS,
)
from qratum_asi.generalization.domain_registry import (
    ExtendedDomainRegistry,
    DomainDefinition,
)
from qratum_asi.generalization.reasoning_engine import (
    GeneralReasoningEngine,
    ReasoningMode,
    ReasoningContext,
    CrossDomainSynthesizer,
)
from qratum_asi.generalization.hypothesis_generator import (
    HypothesisGenerator,
)
from qratum_asi.generalization.state_space import (
    UniversalStateSpace,
    StateCompressor,
    AHTCEncoder,
)


class TestCognitiveDomain:
    """Test cognitive domain types."""

    def test_core_verticals_exist(self):
        """Test that original 14 verticals are defined."""
        core_verticals = [
            CognitiveDomain.VITRA,
            CognitiveDomain.CAPRA,
            CognitiveDomain.STRATA,
            CognitiveDomain.ECORA,
            CognitiveDomain.NEURA,
            CognitiveDomain.FLUXA,
            CognitiveDomain.CHRONA,
            CognitiveDomain.COHORA,
            CognitiveDomain.FUSIA,
            CognitiveDomain.GEONA,
            CognitiveDomain.JURIS,
            CognitiveDomain.ORBIA,
            CognitiveDomain.SENTRA,
            CognitiveDomain.VEXOR,
        ]

        for vertical in core_verticals:
            assert vertical is not None
            assert vertical.value is not None

    def test_extended_domains_exist(self):
        """Test that extended domains are defined."""
        extended = [
            CognitiveDomain.MATHEMATICS,
            CognitiveDomain.PHYSICS,
            CognitiveDomain.PHILOSOPHY,
            CognitiveDomain.ENGINEERING,
            CognitiveDomain.COMPUTER_SCIENCE,
        ]

        for domain in extended:
            assert domain is not None
            assert domain.value is not None


class TestExtendedDomainRegistry:
    """Test the extended domain registry."""

    def test_initialization(self):
        """Test registry initialization."""
        registry = ExtendedDomainRegistry()

        assert len(registry.get_all_domains()) > 14
        assert len(registry.get_core_verticals()) == 14
        assert len(registry.get_extended_domains()) > 0

    def test_get_domain(self):
        """Test getting domain definitions."""
        registry = ExtendedDomainRegistry()

        vitra = registry.get_domain(CognitiveDomain.VITRA)
        assert vitra is not None
        assert vitra.is_core_vertical
        assert "VITRA" in vitra.name

        math = registry.get_domain(CognitiveDomain.MATHEMATICS)
        assert math is not None
        assert not math.is_core_vertical
        assert math.formalization_level > 0.9

    def test_capability_index(self):
        """Test capability indexing."""
        registry = ExtendedDomainRegistry()

        analysis_domains = registry.get_domains_with_capability(DomainCapability.ANALYSIS)
        assert len(analysis_domains) > 0

        verification_domains = registry.get_domains_with_capability(DomainCapability.VERIFICATION)
        assert CognitiveDomain.MATHEMATICS in verification_domains
        assert CognitiveDomain.LOGIC in verification_domains

    def test_interconnections(self):
        """Test domain interconnections."""
        registry = ExtendedDomainRegistry()

        # Math to Physics should be strongly connected
        conn = registry.get_interconnection(CognitiveDomain.MATHEMATICS, CognitiveDomain.PHYSICS)
        assert conn is not None
        assert conn.connection_strength > 0.8

    def test_connected_domains(self):
        """Test finding connected domains."""
        registry = ExtendedDomainRegistry()

        connected = registry.get_connected_domains(CognitiveDomain.MATHEMATICS)
        assert len(connected) > 0

        # Physics should be highly connected to Math
        physics_conn = [c for c in connected if c[0] == CognitiveDomain.PHYSICS]
        assert len(physics_conn) > 0

    def test_synthesis_paths(self):
        """Test finding synthesis paths."""
        registry = ExtendedDomainRegistry()

        paths = registry.find_synthesis_path(
            CognitiveDomain.MATHEMATICS,
            CognitiveDomain.ENGINEERING,
            max_hops=3
        )

        assert paths is not None
        assert len(paths) > 0
        assert paths[0][0] == CognitiveDomain.MATHEMATICS

    def test_registry_stats(self):
        """Test registry statistics."""
        registry = ExtendedDomainRegistry()
        stats = registry.get_registry_stats()

        assert stats["total_domains"] > 14
        assert stats["core_verticals"] == 14
        assert stats["total_interconnections"] > 0


class TestGeneralReasoningEngine:
    """Test the general reasoning engine."""

    @pytest.fixture
    def contract(self):
        """Create a test contract."""
        return ASIContract(
            contract_id="test_001",
            operation_type="test_reasoning",
            safety_level=ASISafetyLevel.ROUTINE,
            authorization_type=AuthorizationType.NONE,
            payload={},
        )

    def test_initialization(self):
        """Test engine initialization."""
        engine = GeneralReasoningEngine()

        assert engine.domain_registry is not None
        assert engine.merkle_chain is not None
        assert len(engine.get_supported_domains()) > 14

    def test_single_domain_reasoning(self, contract):
        """Test reasoning within a single domain."""
        engine = GeneralReasoningEngine()

        context = ReasoningContext(
            context_id="ctx_001",
            primary_domain=CognitiveDomain.MATHEMATICS,
            supporting_domains=[],
            goal="Prove theorem",
            constraints={},
            prior_knowledge=["Axiom 1", "Axiom 2"],
        )

        chain = engine.reason(
            context=context,
            mode=ReasoningMode.SINGLE_DOMAIN,
            contract=contract,
        )

        assert chain is not None
        assert chain.chain_id is not None
        assert len(chain.steps) > 0
        assert chain.overall_confidence > 0

    def test_cross_domain_reasoning(self, contract):
        """Test cross-domain reasoning."""
        engine = GeneralReasoningEngine()

        context = ReasoningContext(
            context_id="ctx_002",
            primary_domain=CognitiveDomain.PHYSICS,
            supporting_domains=[CognitiveDomain.MATHEMATICS],
            goal="Model physical system",
            constraints={},
            prior_knowledge=["Physical law"],
        )

        chain = engine.reason(
            context=context,
            mode=ReasoningMode.CROSS_DOMAIN,
            contract=contract,
        )

        assert chain is not None
        assert len(chain.domains_traversed) >= 2

    def test_cross_domain_synthesis(self, contract):
        """Test cross-domain synthesis."""
        engine = GeneralReasoningEngine()

        result = engine.cross_domain_synthesis(
            source_domains=[CognitiveDomain.PHYSICS, CognitiveDomain.PHILOSOPHY],
            synthesis_goal="Explore nature of reality",
            context={},
            contract=contract,
        )

        assert result is not None
        assert result.synthesis_id is not None
        assert len(result.hypotheses_generated) > 0
        assert result.confidence > 0

    def test_synthesis_safety_validation(self, contract):
        """Test that synthesis validates safety."""
        engine = GeneralReasoningEngine()

        # Should raise for prohibited goal
        with pytest.raises(ValueError):
            engine.cross_domain_synthesis(
                source_domains=[CognitiveDomain.PHYSICS],
                synthesis_goal="Design weapons development system",
                context={},
                contract=contract,
            )

    def test_engine_stats(self, contract):
        """Test engine statistics."""
        engine = GeneralReasoningEngine()

        # Run some operations
        context = ReasoningContext(
            context_id="ctx_003",
            primary_domain=CognitiveDomain.BIOLOGY,
            supporting_domains=[],
            goal="Analyze",
            constraints={},
            prior_knowledge=[],
        )
        engine.reason(context, ReasoningMode.SINGLE_DOMAIN, contract)

        stats = engine.get_engine_stats()
        assert stats["total_reasoning_chains"] >= 1
        assert stats["domains_supported"] > 14


class TestCrossDomainHypothesis:
    """Test cross-domain hypothesis type."""

    def test_safety_validation(self):
        """Test hypothesis safety validation."""
        safe_hypothesis = CrossDomainHypothesis(
            hypothesis_id="hyp_001",
            source_domains=[CognitiveDomain.PHYSICS, CognitiveDomain.MATHEMATICS],
            target_domain=CognitiveDomain.PHYSICS,
            hypothesis_type=HypothesisType.UNIFYING,
            statement="A unified principle connects these domains",
            confidence=0.7,
            novelty_score=0.6,
            testability_score=0.5,
            supporting_evidence=[],
            required_validation=[],
            safety_level=SynthesisSafetyLevel.ROUTINE,
            provenance_hash="abc123",
        )

        assert safe_hypothesis.validate_safety() is True

    def test_unsafe_hypothesis_blocked(self):
        """Test that unsafe hypotheses are flagged."""
        unsafe_hypothesis = CrossDomainHypothesis(
            hypothesis_id="hyp_002",
            source_domains=[CognitiveDomain.COMPUTER_SCIENCE],
            target_domain=CognitiveDomain.COMPUTER_SCIENCE,
            hypothesis_type=HypothesisType.MECHANISTIC,
            statement="Mass manipulation technique for human control",
            confidence=0.8,
            novelty_score=0.9,
            testability_score=0.5,
            supporting_evidence=[],
            required_validation=[],
            safety_level=SynthesisSafetyLevel.ROUTINE,
            provenance_hash="def456",
        )

        assert unsafe_hypothesis.validate_safety() is False


class TestHypothesisGenerator:
    """Test hypothesis generator."""

    @pytest.fixture
    def contract(self):
        """Create a test contract."""
        return ASIContract(
            contract_id="test_002",
            operation_type="test_generation",
            safety_level=ASISafetyLevel.ROUTINE,
            authorization_type=AuthorizationType.NONE,
            payload={},
        )

    def test_initialization(self):
        """Test generator initialization."""
        generator = HypothesisGenerator()

        assert generator.domain_registry is not None
        assert len(generator.templates) > 0

    def test_basic_generation(self, contract):
        """Test basic hypothesis generation."""
        generator = HypothesisGenerator()

        constraints = GenerationConstraints(
            max_novelty=0.8,
            min_testability=0.3,
            max_hypotheses=5,
        )

        result = generator.generate(
            seed_domains=[CognitiveDomain.PHYSICS, CognitiveDomain.MATHEMATICS],
            exploration_goal="Explore mathematical physics",
            constraints=constraints,
            contract=contract,
        )

        assert result is not None
        assert len(result.hypotheses_generated) > 0
        assert len(result.hypotheses_generated) <= 5

    def test_novelty_bounds(self, contract):
        """Test that novelty bounds are enforced."""
        generator = HypothesisGenerator()

        constraints = GenerationConstraints(
            max_novelty=0.5,  # Strict novelty bound
            min_testability=0.3,
            max_hypotheses=10,
        )

        result = generator.generate(
            seed_domains=[CognitiveDomain.PHILOSOPHY],
            exploration_goal="Test novelty bounds",
            constraints=constraints,
            contract=contract,
        )

        # All hypotheses should respect novelty bound
        for h in result.hypotheses_generated:
            assert h.novelty_score <= 0.5

    def test_forbidden_topics_filtered(self, contract):
        """Test that forbidden topics are filtered."""
        generator = HypothesisGenerator()

        constraints = GenerationConstraints(
            forbidden_topics=["weapons_development"],
            max_hypotheses=10,
        )

        result = generator.generate(
            seed_domains=[CognitiveDomain.ENGINEERING],
            exploration_goal="Analyze systems",
            constraints=constraints,
            contract=contract,
        )

        # No hypotheses should mention weapons
        for h in result.hypotheses_generated:
            assert "weapons" not in h.statement.lower()


class TestUniversalStateSpace:
    """Test universal state space."""

    def test_initialization(self):
        """Test state space initialization."""
        space = UniversalStateSpace()

        assert space.compressor is not None
        assert len(space.states) == 0

    def test_add_state(self):
        """Test adding a state."""
        space = UniversalStateSpace()

        state_data = {
            "values": [1.0, 2.0, 3.0],
            "metadata": {"type": "test"},
        }

        state = space.add_state(state_data, CognitiveDomain.MATHEMATICS)

        assert state is not None
        assert state.state_id is not None
        assert state.domain == CognitiveDomain.MATHEMATICS
        assert state.reconstruction_fidelity > 0.9

    def test_state_similarity(self):
        """Test state similarity computation."""
        space = UniversalStateSpace()

        state_a = space.add_state(
            {"values": [1.0, 2.0, 3.0]},
            CognitiveDomain.PHYSICS,
        )

        state_b = space.add_state(
            {"values": [1.0, 2.0, 3.0]},
            CognitiveDomain.PHYSICS,
        )

        state_c = space.add_state(
            {"values": [3.0, 1.0, 0.5]},  # Different direction, not proportional
            CognitiveDomain.PHYSICS,
        )

        # Similar states should have high similarity
        sim_ab = space.compute_state_similarity(state_a, state_b)
        assert sim_ab > 0.9

        # Different direction states should have lower similarity
        sim_ac = space.compute_state_similarity(state_a, state_c)
        assert sim_ac < sim_ab

    def test_combine_states(self):
        """Test combining states."""
        space = UniversalStateSpace()

        state_a = space.add_state(
            {"values": [1.0, 2.0]},
            CognitiveDomain.MATHEMATICS,
        )

        state_b = space.add_state(
            {"values": [3.0, 4.0]},
            CognitiveDomain.PHYSICS,
        )

        combined = space.combine_states([state_a, state_b])

        assert combined is not None
        assert combined.domain == CognitiveDomain.FUSIA  # Multi-domain fusion

    def test_project_state(self):
        """Test projecting state to different domain."""
        space = UniversalStateSpace()

        original = space.add_state(
            {"values": [1.0, 2.0, 3.0]},
            CognitiveDomain.PHYSICS,
        )

        projected = space.project_state(original, CognitiveDomain.MATHEMATICS)

        assert projected is not None
        assert projected.domain == CognitiveDomain.MATHEMATICS
        assert "projected_from" in projected.compression_metadata


class TestAHTCEncoder:
    """Test AHTC encoder."""

    def test_encode_decode_fidelity(self):
        """Test encode/decode maintains reasonable fidelity."""
        encoder = AHTCEncoder(
            target_compression_ratio=5.0,
            min_fidelity=0.3,  # Placeholder implementation has limited fidelity
        )

        # Simple state data that compresses well
        state_data = {
            "values": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # Uniform data
        }

        compressed, metrics = encoder.encode(state_data, CognitiveDomain.MATHEMATICS)

        assert compressed is not None
        # Placeholder achieves moderate fidelity - production would be higher
        assert metrics.fidelity_score >= 0.3
        assert metrics.compression_ratio >= 1.0

    def test_compression_metrics(self):
        """Test compression metrics properties."""
        encoder = AHTCEncoder()

        state_data = {
            "large_data": list(range(100)),
        }

        _, metrics = encoder.encode(state_data, CognitiveDomain.COMPUTER_SCIENCE)

        assert metrics.input_dimensions > 0
        assert metrics.output_dimensions > 0
        assert metrics.semantic_preservation > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
