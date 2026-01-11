"""Tests for Epistemic Heat Sink Module.

Tests the neurosymbolic reasoning, zkML proofs, and heat sink
functionality, verifying invariant preservation and cryptographic
attestation.

Version: 1.0.0
"""

import pytest
import numpy as np
from typing import Any

from epistemic_heat_sink import (
    # Neurosymbolic
    ConceptBottleneck,
    NeurosymbolicReasoner,
    SymbolicConcept,
    ReasoningTrace,
    create_concept_bottleneck,
    # zkML
    ZKMLInferenceProof,
    Plonky3ProofSystem,
    FoldingScheme,
    IncrementalProofChain,
    create_zkml_prover,
    # Heat Sink
    EpistemicHeatSink,
    EpistemicState,
    ErrorCost,
    create_heat_sink,
)
from epistemic_heat_sink.neurosymbolic import ConceptType, ReasoningMode
from epistemic_heat_sink.zkml import ProofSystemType, FoldingSchemeType
from epistemic_heat_sink.heat_sink import EpistemicPhase


class TestSymbolicConcept:
    """Tests for SymbolicConcept."""
    
    def test_concept_creation(self) -> None:
        """Test basic concept creation."""
        concept = SymbolicConcept(
            name="test_concept",
            concept_type=ConceptType.BINARY,
            activation=0.8,
            confidence=0.9,
        )
        
        assert concept.name == "test_concept"
        assert concept.activation == 0.8
        assert concept.confidence == 0.9
    
    def test_concept_is_active(self) -> None:
        """Test is_active property."""
        active_concept = SymbolicConcept(
            name="active",
            concept_type=ConceptType.BINARY,
            activation=0.7,
        )
        inactive_concept = SymbolicConcept(
            name="inactive",
            concept_type=ConceptType.BINARY,
            activation=0.3,
        )
        
        assert active_concept.is_active is True
        assert inactive_concept.is_active is False
    
    def test_weighted_activation(self) -> None:
        """Test weighted activation computation."""
        concept = SymbolicConcept(
            name="test",
            concept_type=ConceptType.CONTINUOUS,
            activation=0.8,
            confidence=0.5,
        )
        
        assert concept.weighted_activation == 0.4
    
    def test_invalid_activation_raises(self) -> None:
        """Test that invalid activation raises ValueError."""
        with pytest.raises(ValueError):
            SymbolicConcept(
                name="invalid",
                concept_type=ConceptType.BINARY,
                activation=1.5,  # Invalid: > 1
            )
    
    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        concept = SymbolicConcept(
            name="test",
            concept_type=ConceptType.CATEGORICAL,
            activation=0.6,
            confidence=0.8,
            description="Test concept",
        )
        
        d = concept.to_dict()
        assert d["name"] == "test"
        assert d["activation"] == 0.6
        assert d["type"] == "CATEGORICAL"


class TestConceptBottleneck:
    """Tests for ConceptBottleneck."""
    
    def test_bottleneck_creation(self) -> None:
        """Test bottleneck creation with dimensions."""
        bottleneck = ConceptBottleneck(
            input_dim=10,
            concept_dim=5,
        )
        
        assert bottleneck.input_dim == 10
        assert bottleneck.concept_dim == 5
        assert bottleneck.encoder_weights is not None
    
    def test_encode_input(self) -> None:
        """Test encoding inputs through bottleneck."""
        bottleneck = ConceptBottleneck(
            input_dim=8,
            concept_dim=4,
        )
        
        np.random.seed(42)
        inputs = np.random.randn(8)
        
        concepts = bottleneck.encode(inputs)
        
        assert len(concepts) == 4
        for c in concepts:
            assert 0 <= c.activation <= 1
    
    def test_concept_intervention(self) -> None:
        """Test concept intervention."""
        bottleneck = ConceptBottleneck(
            input_dim=4,
            concept_dim=2,
        )
        
        np.random.seed(42)
        inputs = np.random.randn(4)
        concepts = bottleneck.encode(inputs)
        
        original_activation = concepts[0].activation
        
        bottleneck.intervene("concept_0", 0.99)
        
        assert bottleneck.concepts[0].activation == 0.99
        assert bottleneck.concepts[0].confidence == 1.0
    
    def test_get_active_concepts(self) -> None:
        """Test getting active concepts."""
        bottleneck = ConceptBottleneck(
            input_dim=4,
            concept_dim=4,
        )
        
        np.random.seed(42)
        inputs = np.random.randn(4)
        bottleneck.encode(inputs)
        
        active = bottleneck.get_active_concepts(threshold=0.5)
        
        # Some concepts should be active, some not
        assert len(active) <= 4
    
    def test_state_hash_deterministic(self) -> None:
        """Test that state hash is deterministic."""
        bottleneck = ConceptBottleneck(
            input_dim=4,
            concept_dim=2,
        )
        
        np.random.seed(42)
        inputs = np.random.randn(4)
        bottleneck.encode(inputs)
        
        hash1 = bottleneck.compute_hash()
        hash2 = bottleneck.compute_hash()
        
        assert hash1 == hash2


class TestNeurosymbolicReasoner:
    """Tests for NeurosymbolicReasoner."""
    
    def test_reasoner_creation(self) -> None:
        """Test reasoner creation."""
        reasoner = NeurosymbolicReasoner(
            input_dim=16,
            concept_dim=8,
        )
        
        assert reasoner.bottleneck.input_dim == 16
        assert reasoner.bottleneck.concept_dim == 8
    
    def test_reason_deductive(self) -> None:
        """Test deductive reasoning."""
        reasoner = NeurosymbolicReasoner(
            input_dim=8,
            concept_dim=4,
        )
        
        np.random.seed(42)
        inputs = np.random.randn(8)
        
        trace = reasoner.reason(inputs, mode=ReasoningMode.DEDUCTIVE)
        
        assert isinstance(trace, ReasoningTrace)
        assert trace.mode == ReasoningMode.DEDUCTIVE
        assert len(trace.steps) > 0
    
    def test_reasoning_trace_confidence(self) -> None:
        """Test reasoning trace confidence computation."""
        reasoner = NeurosymbolicReasoner(
            input_dim=8,
            concept_dim=4,
        )
        
        np.random.seed(42)
        inputs = np.random.randn(8)
        
        trace = reasoner.reason(inputs)
        
        assert 0 <= trace.total_confidence <= 1
    
    def test_add_custom_rule(self) -> None:
        """Test adding custom reasoning rule."""
        reasoner = NeurosymbolicReasoner(
            input_dim=4,
            concept_dim=2,
        )
        
        def custom_rule(concepts):
            return [SymbolicConcept(
                name="custom_output",
                concept_type=ConceptType.BINARY,
                activation=max(c.activation for c in concepts),
                confidence=0.95,
            )]
        
        reasoner.add_rule("custom", custom_rule)
        
        np.random.seed(42)
        inputs = np.random.randn(4)
        
        trace = reasoner.reason(inputs, rules_to_apply=["custom"])
        
        # Check custom rule was applied
        rule_names = [s.rule_applied for s in trace.steps]
        assert "custom" in rule_names
    
    def test_explain_trace(self) -> None:
        """Test trace explanation generation."""
        reasoner = NeurosymbolicReasoner(
            input_dim=4,
            concept_dim=2,
        )
        
        np.random.seed(42)
        inputs = np.random.randn(4)
        
        trace = reasoner.reason(inputs)
        explanation = reasoner.explain(trace)
        
        assert isinstance(explanation, str)
        assert trace.trace_id in explanation
    
    def test_trace_provenance_hash(self) -> None:
        """Test trace provenance hash computation."""
        reasoner = NeurosymbolicReasoner(
            input_dim=4,
            concept_dim=2,
        )
        
        np.random.seed(42)
        inputs = np.random.randn(4)
        
        trace = reasoner.reason(inputs)
        provenance_hash = trace.compute_provenance_hash()
        
        assert len(provenance_hash) == 64  # SHA-256 hex


class TestPlonky3ProofSystem:
    """Tests for Plonky3ProofSystem."""
    
    def test_proof_system_creation(self) -> None:
        """Test proof system creation."""
        prover = Plonky3ProofSystem(
            circuit_size=2**8,
            security_level=128,
        )
        
        assert prover.circuit_size == 2**8
        assert prover.security_level == 128
    
    def test_prove_inference(self) -> None:
        """Test inference proof generation."""
        prover = Plonky3ProofSystem()
        
        np.random.seed(42)
        inputs = np.random.randn(8)
        weights = np.random.randn(4, 8) * 0.1
        outputs = inputs @ weights.T
        
        proof = prover.prove_inference(inputs, outputs, weights)
        
        assert proof.proof_type == ProofSystemType.PLONKY3
        assert proof.input_commitment is not None
        assert proof.output_commitment is not None
    
    def test_verify_proof(self) -> None:
        """Test proof verification."""
        prover = Plonky3ProofSystem()
        
        np.random.seed(42)
        inputs = np.random.randn(8)
        weights = np.random.randn(4, 8) * 0.1
        outputs = inputs @ weights.T
        
        proof = prover.prove_inference(inputs, outputs, weights)
        
        assert prover.verify_proof(proof) is True
    
    def test_proof_hash_unique(self) -> None:
        """Test that proof hashes are unique."""
        prover = Plonky3ProofSystem()
        
        np.random.seed(42)
        inputs1 = np.random.randn(4)
        inputs2 = np.random.randn(4)
        weights = np.random.randn(2, 4) * 0.1
        
        proof1 = prover.prove_inference(inputs1, inputs1 @ weights.T, weights)
        proof2 = prover.prove_inference(inputs2, inputs2 @ weights.T, weights)
        
        assert proof1.proof_hash != proof2.proof_hash


class TestFoldingScheme:
    """Tests for FoldingScheme."""
    
    def test_folding_scheme_creation(self) -> None:
        """Test folding scheme creation."""
        folder = FoldingScheme(
            scheme_type=FoldingSchemeType.NOVA,
            recursion_limit=50,
        )
        
        assert folder.scheme_type == FoldingSchemeType.NOVA
        assert folder.recursion_limit == 50
    
    def test_create_initial_instance(self) -> None:
        """Test initial instance creation."""
        folder = FoldingScheme()
        
        instance = folder.create_initial_instance(
            data=b"test_data",
            public_inputs=[1, 2, 3],
        )
        
        assert instance.fold_count == 0
        assert instance.public_inputs == [1, 2, 3]
    
    def test_fold_instances(self) -> None:
        """Test folding two instances."""
        folder = FoldingScheme()
        
        instance1 = folder.create_initial_instance(b"data1", [1, 2])
        instance2 = folder.create_initial_instance(b"data2", [3, 4])
        
        folded = folder.fold(instance1, instance2)
        
        assert folded.fold_count == 1
        assert len(folded.public_inputs) == 2
    
    def test_multiple_folds(self) -> None:
        """Test multiple sequential folds."""
        folder = FoldingScheme()
        
        instances = [
            folder.create_initial_instance(f"data{i}".encode(), [i])
            for i in range(4)
        ]
        
        # Fold pairwise
        folded1 = folder.fold(instances[0], instances[1])
        folded2 = folder.fold(instances[2], instances[3])
        final = folder.fold(folded1, folded2)
        
        assert final.fold_count >= 3


class TestIncrementalProofChain:
    """Tests for IncrementalProofChain."""
    
    def test_chain_creation(self) -> None:
        """Test proof chain creation."""
        folder = FoldingScheme()
        chain = IncrementalProofChain(
            chain_id="test_chain",
            folding_scheme=folder,
        )
        
        assert chain.chain_id == "test_chain"
        assert chain.chain_length == 0
    
    def test_extend_chain(self) -> None:
        """Test extending proof chain."""
        prover = Plonky3ProofSystem()
        folder = FoldingScheme()
        chain = IncrementalProofChain(
            chain_id="test",
            folding_scheme=folder,
        )
        
        np.random.seed(42)
        inputs = np.random.randn(4)
        weights = np.random.randn(2, 4) * 0.1
        outputs = inputs @ weights.T
        
        proof = prover.prove_inference(inputs, outputs, weights)
        chain.extend(proof)
        
        assert chain.chain_length == 1
    
    def test_verify_chain(self) -> None:
        """Test chain verification."""
        prover = Plonky3ProofSystem()
        folder = FoldingScheme()
        chain = IncrementalProofChain(
            chain_id="test",
            folding_scheme=folder,
        )
        
        np.random.seed(42)
        for i in range(3):
            inputs = np.random.randn(4)
            weights = np.random.randn(2, 4) * 0.1
            outputs = inputs @ weights.T
            proof = prover.prove_inference(inputs, outputs, weights)
            chain.extend(proof)
        
        assert chain.verify_chain() is True


class TestEpistemicState:
    """Tests for EpistemicState."""
    
    def test_state_creation(self) -> None:
        """Test state creation."""
        state = EpistemicState(
            phase=EpistemicPhase.GROUND,
            entropy=0.5,
            trust_balance=1.0,
        )
        
        assert state.phase == EpistemicPhase.GROUND
        assert state.entropy == 0.5
        assert state.trust_balance == 1.0
    
    def test_trust_conserved(self) -> None:
        """Test trust conservation check."""
        valid_state = EpistemicState(trust_balance=0.5)
        assert valid_state.trust_conserved is True
        
        invalid_state = EpistemicState(trust_balance=-0.1)
        assert invalid_state.trust_conserved is False
    
    def test_verification_ratio(self) -> None:
        """Test verification ratio computation."""
        state = EpistemicState(
            verified_assertions=7,
            total_assertions=10,
        )
        
        assert state.verification_ratio == 0.7
    
    def test_free_energy_update(self) -> None:
        """Test free energy update."""
        state = EpistemicState(
            entropy=0.5,
            temperature=1.0,
            verified_assertions=8,
            total_assertions=10,
        )
        
        state.update_free_energy()
        
        # Free energy should be computed
        assert state.free_energy is not None


class TestErrorCost:
    """Tests for ErrorCost."""
    
    def test_error_entropy_verified(self) -> None:
        """Test entropy for verified assertions."""
        cost = ErrorCost()
        
        entropy_verified = cost.compute_error_entropy(
            confidence=0.9,
            is_verified=True,
            is_valid=True,
        )
        entropy_unverified = cost.compute_error_entropy(
            confidence=0.9,
            is_verified=False,
            is_valid=True,
        )
        
        # Verified should have lower entropy
        assert entropy_verified < entropy_unverified
    
    def test_invalid_state_infinite_cost(self) -> None:
        """Test that invalid states have infinite cost."""
        cost = ErrorCost()
        
        entropy = cost.compute_error_entropy(
            confidence=0.9,
            is_verified=True,
            is_valid=False,  # Invalid!
        )
        
        assert entropy == float("inf")
    
    def test_total_cost_with_invalid(self) -> None:
        """Test total cost with invalid state."""
        cost = ErrorCost()
        
        entropies = [0.1, 0.2, float("inf")]
        total = cost.total_cost(entropies)
        
        assert total == float("inf")


class TestEpistemicHeatSink:
    """Tests for EpistemicHeatSink."""
    
    def test_heat_sink_creation(self) -> None:
        """Test heat sink creation."""
        sink = EpistemicHeatSink(
            input_dim=16,
            concept_dim=8,
        )
        
        assert sink.state.phase == EpistemicPhase.GROUND
        assert sink.trust_balance >= 0
    
    def test_assert_with_proof(self) -> None:
        """Test making assertion with proof."""
        sink = EpistemicHeatSink(
            input_dim=8,
            concept_dim=4,
        )
        
        np.random.seed(42)
        inputs = np.random.randn(8)
        
        verified, proof, trace = sink.assert_with_proof(
            inputs=inputs,
            assertion="test assertion",
        )
        
        assert verified is True
        assert proof is not None
        assert trace is not None
    
    def test_trust_invariant_preserved(self) -> None:
        """Test that trust invariant ℛ(t) ≥ 0 is preserved."""
        sink = EpistemicHeatSink(
            input_dim=8,
            concept_dim=4,
        )
        
        np.random.seed(42)
        
        for i in range(5):
            inputs = np.random.randn(8)
            sink.assert_with_proof(inputs, f"assertion_{i}")
            
            # Trust must always be preserved
            assert sink.trust_balance >= 0
            assert sink.state.trust_conserved
    
    def test_cool_down(self) -> None:
        """Test cooling down the heat sink."""
        sink = EpistemicHeatSink()
        
        initial_temp = sink.state.temperature
        sink.cool_down(0.2)
        
        assert sink.state.temperature < initial_temp
    
    def test_heat_up(self) -> None:
        """Test heating up the heat sink."""
        sink = EpistemicHeatSink()
        
        initial_temp = sink.state.temperature
        sink.heat_up(0.2)
        
        assert sink.state.temperature > initial_temp
    
    def test_verify_evolution(self) -> None:
        """Test evolution verification."""
        sink = EpistemicHeatSink(
            input_dim=8,
            concept_dim=4,
        )
        
        from_state = EpistemicState(
            trust_balance=1.0,
            entropy=0.5,
            temperature=1.0,
        )
        
        to_state = EpistemicState(
            trust_balance=0.9,
            entropy=0.7,
            temperature=1.0,
        )
        
        assert sink.verify_evolution(from_state, to_state) is True
    
    def test_verify_evolution_trust_violated(self) -> None:
        """Test evolution verification with trust violation."""
        sink = EpistemicHeatSink(
            input_dim=8,
            concept_dim=4,
        )
        
        from_state = EpistemicState(
            trust_balance=1.0,
            entropy=0.5,
        )
        
        to_state = EpistemicState(
            trust_balance=-0.1,  # Violates trust invariant!
            entropy=0.5,
        )
        
        assert sink.verify_evolution(from_state, to_state) is False
    
    def test_audit_report(self) -> None:
        """Test audit report generation."""
        sink = EpistemicHeatSink(
            input_dim=8,
            concept_dim=4,
        )
        
        np.random.seed(42)
        inputs = np.random.randn(8)
        sink.assert_with_proof(inputs, "test")
        
        report = sink.get_audit_report()
        
        assert "heat_sink_version" in report
        assert "current_state" in report
        assert "trust_invariant_satisfied" in report
        assert report["trust_invariant_satisfied"] is True
        assert "compliance_statement" in report


class TestFactoryFunctions:
    """Tests for factory functions."""
    
    def test_create_concept_bottleneck(self) -> None:
        """Test create_concept_bottleneck factory."""
        bottleneck = create_concept_bottleneck(
            input_dim=10,
            concept_dim=5,
        )
        
        assert isinstance(bottleneck, ConceptBottleneck)
        assert bottleneck.input_dim == 10
    
    def test_create_zkml_prover(self) -> None:
        """Test create_zkml_prover factory."""
        prover, folder = create_zkml_prover(
            circuit_size=2**8,
            folding_scheme=FoldingSchemeType.NOVA,
        )
        
        assert isinstance(prover, Plonky3ProofSystem)
        assert isinstance(folder, FoldingScheme)
    
    def test_create_heat_sink(self) -> None:
        """Test create_heat_sink factory."""
        sink = create_heat_sink(
            input_dim=32,
            concept_dim=16,
        )
        
        assert isinstance(sink, EpistemicHeatSink)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
