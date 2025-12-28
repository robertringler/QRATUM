"""Tests for Topological Observer Module.

Tests the read-only persistent homology instrumentation layer,
verifying that all observations are non-authoritative and that
invariants are preserved.

Version: 1.0.0
"""

import pytest
import numpy as np
from typing import Any

from topological_observer import (
    PersistentHomologyObserver,
    BettiNumbers,
    PersistenceDiagram,
    TopologicalAnnotation,
    compute_persistent_homology,
    compute_betti_numbers,
    TopologicalInstrumentationLayer,
    ObservationResult,
    InvariantAssertion,
)
from topological_observer.observer import ObservationStatus, InvariantType


class TestBettiNumbers:
    """Tests for BettiNumbers dataclass."""
    
    def test_betti_numbers_creation(self) -> None:
        """Test basic Betti numbers creation."""
        betti = BettiNumbers(beta_0=3, beta_1=2, beta_2=1)
        assert betti.beta_0 == 3
        assert betti.beta_1 == 2
        assert betti.beta_2 == 1
    
    def test_euler_characteristic(self) -> None:
        """Test Euler characteristic computation."""
        betti = BettiNumbers(beta_0=1, beta_1=0, beta_2=0)
        assert betti.euler_characteristic == 1
        
        betti = BettiNumbers(beta_0=2, beta_1=1, beta_2=0)
        assert betti.euler_characteristic == 1  # 2 - 1 + 0 = 1
        
        betti = BettiNumbers(beta_0=1, beta_1=2, beta_2=1)
        assert betti.euler_characteristic == 0  # 1 - 2 + 1 = 0
    
    def test_total_features(self) -> None:
        """Test total features count."""
        betti = BettiNumbers(beta_0=3, beta_1=2, beta_2=1)
        assert betti.total_features == 6
    
    def test_negative_betti_raises(self) -> None:
        """Test that negative Betti numbers raise ValueError."""
        with pytest.raises(ValueError):
            BettiNumbers(beta_0=-1, beta_1=0, beta_2=0)
    
    def test_hash_deterministic(self) -> None:
        """Test that hash computation is deterministic."""
        betti1 = BettiNumbers(beta_0=1, beta_1=2, beta_2=3, timestamp=12345.0)
        betti2 = BettiNumbers(beta_0=1, beta_1=2, beta_2=3, timestamp=12345.0)
        assert betti1.compute_hash() == betti2.compute_hash()
    
    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        betti = BettiNumbers(beta_0=1, beta_1=2, beta_2=3)
        d = betti.to_dict()
        assert d["beta_0"] == 1
        assert d["beta_1"] == 2
        assert d["beta_2"] == 3
        assert "euler_characteristic" in d


class TestPersistenceDiagram:
    """Tests for PersistenceDiagram."""
    
    def test_empty_diagram(self) -> None:
        """Test empty persistence diagram."""
        diagram = PersistenceDiagram()
        assert len(diagram.intervals) == 0
        betti = diagram.get_betti_numbers()
        assert betti.beta_0 == 0
    
    def test_get_betti_at_threshold(self) -> None:
        """Test Betti number computation at threshold."""
        from topological_observer.homology import PersistenceInterval
        
        diagram = PersistenceDiagram(intervals=[
            PersistenceInterval(birth=0.0, death=float("inf"), dimension=0),
            PersistenceInterval(birth=0.0, death=1.0, dimension=0),
            PersistenceInterval(birth=0.5, death=1.5, dimension=1),
        ])
        
        betti_0 = diagram.get_betti_numbers(threshold=0.0)
        assert betti_0.beta_0 >= 1  # At least one component
        
        betti_0_7 = diagram.get_betti_numbers(threshold=0.7)
        assert betti_0_7.beta_1 == 1  # Cycle exists at 0.7
    
    def test_persistent_features(self) -> None:
        """Test getting persistent features."""
        from topological_observer.homology import PersistenceInterval
        
        diagram = PersistenceDiagram(intervals=[
            PersistenceInterval(birth=0.0, death=0.05, dimension=0),  # Short-lived
            PersistenceInterval(birth=0.0, death=1.0, dimension=0),   # Long-lived
        ])
        
        persistent = diagram.get_persistent_features(min_persistence=0.1)
        assert len(persistent) == 1


class TestTopologicalAnnotation:
    """Tests for TopologicalAnnotation."""
    
    def test_annotation_non_authoritative(self) -> None:
        """Test that annotations are always non-authoritative."""
        betti = BettiNumbers(beta_0=1, beta_1=0, beta_2=0)
        diagram = PersistenceDiagram()
        
        annotation = TopologicalAnnotation(
            source_id="test",
            betti_numbers=betti,
            persistence_diagram=diagram,
        )
        
        # Must always be non-authoritative
        assert annotation.is_authoritative is False
        
        # Cannot be overridden
        annotation.is_authoritative = True  # This should have no effect
        assert annotation.is_authoritative is False
    
    def test_annotation_summary(self) -> None:
        """Test annotation summary generation."""
        betti = BettiNumbers(beta_0=2, beta_1=1, beta_2=0)
        diagram = PersistenceDiagram()
        
        annotation = TopologicalAnnotation(
            source_id="test_source",
            betti_numbers=betti,
            persistence_diagram=diagram,
        )
        
        summary = annotation.summary
        assert "test_source" in summary
        assert "β₀=2" in summary
        assert "β₁=1" in summary
    
    def test_attestation_hash_deterministic(self) -> None:
        """Test that attestation hash is deterministic."""
        betti = BettiNumbers(beta_0=1, beta_1=1, beta_2=1)
        diagram = PersistenceDiagram()
        
        ann1 = TopologicalAnnotation(
            source_id="test",
            betti_numbers=betti,
            persistence_diagram=diagram,
        )
        ann2 = TopologicalAnnotation(
            source_id="test",
            betti_numbers=betti,
            persistence_diagram=diagram,
        )
        
        # Same content should produce same hash
        assert ann1.compute_attestation_hash() == ann2.compute_attestation_hash()


class TestPersistentHomologyObserver:
    """Tests for PersistentHomologyObserver."""
    
    def test_observe_point_cloud(self) -> None:
        """Test observing a simple point cloud."""
        observer = PersistentHomologyObserver(max_dimension=2)
        
        # Create a simple point cloud
        np.random.seed(42)
        points = np.random.randn(20, 3)
        
        annotation = observer.observe(points, source_id="test_cloud")
        
        assert annotation.betti_numbers.beta_0 >= 1
        assert annotation.is_authoritative is False
        assert annotation.source_id == "test_cloud"
    
    def test_observation_is_read_only(self) -> None:
        """Test that observation does not modify input data."""
        observer = PersistentHomologyObserver()
        
        np.random.seed(42)
        original_points = np.random.randn(10, 3)
        points_copy = original_points.copy()
        
        observer.observe(original_points, source_id="test")
        
        # Original data should be unchanged
        np.testing.assert_array_equal(original_points, points_copy)
    
    def test_observation_count_increments(self) -> None:
        """Test that observation count increments."""
        observer = PersistentHomologyObserver()
        
        assert observer.observation_count == 0
        
        np.random.seed(42)
        points = np.random.randn(5, 2)
        observer.observe(points)
        
        assert observer.observation_count == 1
        
        observer.observe(points)
        assert observer.observation_count == 2
    
    def test_observe_1d_data(self) -> None:
        """Test observing 1D data."""
        observer = PersistentHomologyObserver()
        
        data = np.array([1.0, 2.0, 3.0, 5.0, 8.0])
        annotation = observer.observe(data, source_id="1d_test")
        
        assert annotation.betti_numbers.beta_0 >= 1


class TestTopologicalInstrumentationLayer:
    """Tests for TopologicalInstrumentationLayer."""
    
    def test_layer_initialization(self) -> None:
        """Test layer initialization."""
        layer = TopologicalInstrumentationLayer(max_dimension=2)
        
        assert layer.observation_count == 0
        assert layer.trust_balance >= 0
    
    def test_observe_with_invariants(self) -> None:
        """Test observation with invariant checking."""
        layer = TopologicalInstrumentationLayer(enable_invariant_checking=True)
        
        np.random.seed(42)
        data = np.random.randn(15, 3)
        
        result = layer.observe(data, source_id="test_observation")
        
        assert result.status == ObservationStatus.COMPLETED
        assert result.all_invariants_satisfied
        assert result.trust_conserved
    
    def test_trust_invariant_preserved(self) -> None:
        """Test that trust invariant ℛ(t) ≥ 0 is preserved."""
        layer = TopologicalInstrumentationLayer()
        
        np.random.seed(42)
        
        # Multiple observations
        for i in range(5):
            data = np.random.randn(10, 2)
            result = layer.observe(data, source_id=f"obs_{i}")
            
            # Trust must always be preserved
            assert result.trust_conserved
            assert layer.trust_balance >= 0
    
    def test_read_only_invariant(self) -> None:
        """Test that read-only invariant is checked."""
        layer = TopologicalInstrumentationLayer()
        
        np.random.seed(42)
        data = np.random.randn(8, 2)
        
        result = layer.observe(data, source_id="read_only_test")
        
        # Find read-only invariant
        read_only_inv = None
        for inv in result.invariants:
            if inv.invariant_type == InvariantType.READ_ONLY:
                read_only_inv = inv
                break
        
        assert read_only_inv is not None
        assert read_only_inv.satisfied
    
    def test_non_authoritative_invariant(self) -> None:
        """Test that non-authoritative invariant is checked."""
        layer = TopologicalInstrumentationLayer()
        
        np.random.seed(42)
        data = np.random.randn(8, 2)
        
        result = layer.observe(data, source_id="non_auth_test")
        
        # Find non-authoritative invariant
        non_auth_inv = None
        for inv in result.invariants:
            if inv.invariant_type == InvariantType.NON_AUTHORITATIVE:
                non_auth_inv = inv
                break
        
        assert non_auth_inv is not None
        assert non_auth_inv.satisfied
    
    def test_merkle_chain_updates(self) -> None:
        """Test that Merkle chain is properly maintained."""
        layer = TopologicalInstrumentationLayer()
        
        initial_root = layer.get_merkle_root()
        
        np.random.seed(42)
        data = np.random.randn(5, 2)
        layer.observe(data, source_id="merkle_test")
        
        # Root should change after observation
        new_root = layer.get_merkle_root()
        assert new_root != initial_root
    
    def test_observation_history(self) -> None:
        """Test observation history tracking."""
        layer = TopologicalInstrumentationLayer()
        
        np.random.seed(42)
        
        for i in range(3):
            data = np.random.randn(5, 2)
            layer.observe(data, source_id=f"hist_{i}")
        
        history = layer.get_observation_history()
        assert len(history) == 3
    
    def test_comprehensive_audit_report(self) -> None:
        """Test comprehensive audit report generation."""
        layer = TopologicalInstrumentationLayer()
        
        np.random.seed(42)
        data = np.random.randn(10, 3)
        layer.observe(data, source_id="audit_test")
        
        report = layer.generate_comprehensive_audit_report()
        
        assert "total_observations" in report
        assert "trust_balance" in report
        assert "trust_invariant_satisfied" in report
        assert report["trust_invariant_satisfied"]
        assert "compliance_assertion" in report


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_compute_persistent_homology(self) -> None:
        """Test compute_persistent_homology function."""
        np.random.seed(42)
        points = np.random.randn(20, 3)
        
        diagram = compute_persistent_homology(points, max_dimension=2)
        
        assert isinstance(diagram, PersistenceDiagram)
        assert diagram.max_dimension == 2
    
    def test_compute_betti_numbers(self) -> None:
        """Test compute_betti_numbers function."""
        np.random.seed(42)
        points = np.random.randn(15, 2)
        
        betti = compute_betti_numbers(points)
        
        assert isinstance(betti, BettiNumbers)
        assert betti.beta_0 >= 1


class TestInvariantPreservation:
    """Tests specifically for invariant preservation."""
    
    def test_all_invariants_have_attestation(self) -> None:
        """Test that all invariant assertions have attestation hashes."""
        layer = TopologicalInstrumentationLayer()
        
        np.random.seed(42)
        data = np.random.randn(10, 2)
        result = layer.observe(data)
        
        for inv in result.invariants:
            attestation = inv.compute_attestation()
            assert len(attestation) == 64  # SHA-256 hex
    
    def test_observation_result_merkle_hash(self) -> None:
        """Test that observation results have Merkle hashes."""
        layer = TopologicalInstrumentationLayer()
        
        np.random.seed(42)
        data = np.random.randn(10, 2)
        result = layer.observe(data)
        
        merkle_hash = result.compute_merkle_hash()
        assert len(merkle_hash) == 64  # SHA-256 hex
    
    def test_audit_report_contains_required_fields(self) -> None:
        """Test that audit reports contain all required fields."""
        layer = TopologicalInstrumentationLayer()
        
        np.random.seed(42)
        data = np.random.randn(10, 2)
        result = layer.observe(data)
        
        report = result.generate_audit_report()
        
        required_fields = [
            "observation_id",
            "status",
            "trust_conserved",
            "all_invariants_satisfied",
            "invariant_assertions",
            "betti_numbers",
            "merkle_hash",
            "attestation_hash",
        ]
        
        for field in required_fields:
            assert field in report, f"Missing field: {field}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
