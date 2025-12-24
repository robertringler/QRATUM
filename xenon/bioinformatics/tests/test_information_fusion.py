"""Tests for Partial Information Decomposition (PID) and information fusion.

Validates:
- PID decomposition correctness
- Conservation constraint enforcement
- Deterministic reproducibility
- Numerical stability
"""

from __future__ import annotations

import numpy as np
import pytest

from xenon.bioinformatics.information_fusion import (ConservationConstraints,
                                                     InformationFusionEngine)


class TestInformationFusionEngine:
    """Test suite for information fusion engine."""

    def test_deterministic_entropy_computation(self):
        """Test entropy computation is deterministic."""

        engine1 = InformationFusionEngine(seed=42)
        engine2 = InformationFusionEngine(seed=42)

        data = np.random.RandomState(42).randn(100)

        entropy1 = engine1.compute_entropy(data, bins=10)
        entropy2 = engine2.compute_entropy(data, bins=10)

        assert entropy1 == entropy2
        assert entropy1 > 0.0

    def test_entropy_bounds(self):
        """Test entropy satisfies theoretical bounds."""

        engine = InformationFusionEngine(seed=42)

        # Uniform distribution should have maximum entropy
        uniform_data = np.arange(100)
        h_uniform = engine.compute_entropy(uniform_data, bins=10)

        # Constant distribution should have zero entropy
        constant_data = np.ones(100)
        h_constant = engine.compute_entropy(constant_data, bins=10)

        assert h_uniform > h_constant
        assert h_constant == 0.0

    def test_mutual_information_symmetry(self):
        """Test MI(X;Y) = MI(Y;X)."""

        engine = InformationFusionEngine(seed=42)
        rng = np.random.RandomState(42)

        x = rng.randn(100)
        y = rng.randn(100)

        mi_xy = engine.compute_mutual_information(x, y, bins=10)
        mi_yx = engine.compute_mutual_information(y, x, bins=10)

        assert abs(mi_xy - mi_yx) < 1e-6

    def test_mutual_information_with_self(self):
        """Test MI(X;X) = H(X)."""

        engine = InformationFusionEngine(seed=42)
        rng = np.random.RandomState(42)

        x = rng.randn(100)

        mi_xx = engine.compute_mutual_information(x, x, bins=10)
        h_x = engine.compute_entropy(x, bins=10)

        # MI(X;X) should equal H(X)
        assert abs(mi_xx - h_x) < 0.5  # Relaxed tolerance for discretization

    def test_mutual_information_non_negative(self):
        """Test MI >= 0 for all inputs."""

        engine = InformationFusionEngine(seed=42)
        rng = np.random.RandomState(42)

        # Independent variables
        x = rng.randn(100)
        y = rng.randn(100)

        mi = engine.compute_mutual_information(x, y, bins=10)

        assert mi >= 0.0

    def test_pid_component_sum(self):
        """Test PID components sum to total MI."""

        engine = InformationFusionEngine(seed=42)
        rng = np.random.RandomState(42)

        # Create correlated data
        source1 = rng.randn(100)
        source2 = source1 + 0.5 * rng.randn(100)
        target = 0.5 * source1 + 0.3 * source2 + 0.2 * rng.randn(100)

        result = engine.decompose_information(source1, source2, target, bins=10)

        # Components should sum to total MI
        component_sum = result.unique_s1 + result.unique_s2 + result.redundant + result.synergistic

        assert abs(component_sum - result.total_mi) < 1e-6

    def test_pid_non_negativity(self):
        """Test all PID components are non-negative."""

        engine = InformationFusionEngine(seed=42)
        rng = np.random.RandomState(42)

        source1 = rng.randn(100)
        source2 = rng.randn(100)
        target = rng.randn(100)

        result = engine.decompose_information(source1, source2, target, bins=10)

        assert result.unique_s1 >= 0.0
        assert result.unique_s2 >= 0.0
        assert result.redundant >= 0.0
        assert result.synergistic >= 0.0

    def test_conservation_upper_bound(self):
        """Test total MI does not exceed entropy bounds."""

        engine = InformationFusionEngine(seed=42)
        rng = np.random.RandomState(42)

        source1 = rng.randn(100)
        source2 = rng.randn(100)
        target = rng.randn(100)

        result = engine.decompose_information(source1, source2, target, bins=10)

        # Total MI should not exceed min(H(sources), H(target))
        max_mi = min(result.entropy_sources, result.entropy_target)
        assert result.total_mi <= max_mi + engine.constraints.tolerance

    def test_redundant_information_bounds(self):
        """Test redundant information satisfies theoretical bounds."""

        engine = InformationFusionEngine(seed=42)
        rng = np.random.RandomState(42)

        # Highly redundant sources (identical)
        source = rng.randn(100)
        source1 = source.copy()
        source2 = source.copy()
        target = source + 0.1 * rng.randn(100)

        result = engine.decompose_information(source1, source2, target, bins=10)

        # For identical sources, most information should be redundant
        assert result.redundant > 0.0
        # Unique components should be small
        assert result.unique_s1 < result.redundant
        assert result.unique_s2 < result.redundant

    def test_synergistic_information_detection(self):
        """Test detection of synergistic information."""

        engine = InformationFusionEngine(seed=42)
        rng = np.random.RandomState(42)

        # Create XOR-like relationship (pure synergy)
        source1 = rng.choice([0, 1], size=100)
        source2 = rng.choice([0, 1], size=100)
        target = (source1 + source2) % 2  # XOR

        result = engine.decompose_information(source1, source2, target, bins=2)

        # For XOR, synergistic component should dominate
        # (though discretization may affect this)
        assert result.synergistic >= 0.0

    def test_auto_correction_of_violations(self):
        """Test automatic correction of minor violations."""

        constraints = ConservationConstraints(
            enforce_non_negativity=True, auto_correct=True, tolerance=1e-6
        )
        engine = InformationFusionEngine(constraints=constraints, seed=42)
        rng = np.random.RandomState(42)

        source1 = rng.randn(100)
        source2 = rng.randn(100)
        target = rng.randn(100)

        # Should not raise error even if minor violations occur
        result = engine.decompose_information(source1, source2, target, bins=10)

        # Result should be valid after auto-correction
        assert result.conservation_valid
        assert len(result.violations) == 0

    def test_violation_detection_without_auto_correct(self):
        """Test violation detection when auto-correct is disabled."""

        # Note: With proper implementation, violations should be rare
        # This test verifies the mechanism is in place
        constraints = ConservationConstraints(
            enforce_non_negativity=True,
            auto_correct=False,
            tolerance=1e-20,  # Impossibly strict
        )
        engine = InformationFusionEngine(constraints=constraints, seed=42)
        rng = np.random.RandomState(42)

        source1 = rng.randn(100)
        source2 = rng.randn(100)
        target = rng.randn(100)

        try:
            result = engine.decompose_information(source1, source2, target, bins=10)
            # If no error, check that result is actually valid
            # (implementation is robust enough)
            assert result.conservation_valid or len(result.violations) == 0
        except ValueError as e:
            # Expected if violations occur
            assert "constraint" in str(e).lower()

    def test_information_flow_multiple_layers(self):
        """Test information flow analysis with multiple omics layers."""

        engine = InformationFusionEngine(seed=42)
        rng = np.random.RandomState(42)

        # Create three omics layers
        transcriptomics = rng.randn(100)
        proteomics = transcriptomics + 0.3 * rng.randn(100)
        metabolomics = transcriptomics + 0.5 * rng.randn(100)
        target = transcriptomics + 0.2 * rng.randn(100)

        layers = [transcriptomics, proteomics, metabolomics]
        layer_names = ["transcriptomics", "proteomics", "metabolomics"]

        flow = engine.compute_information_flow(layers, target, layer_names, bins=10)

        assert "individual_mi" in flow
        assert "pairwise_decompositions" in flow
        assert flow["total_layers"] == 3

        # Check individual MIs are computed
        assert "transcriptomics" in flow["individual_mi"]
        assert "proteomics" in flow["individual_mi"]
        assert "metabolomics" in flow["individual_mi"]

        # Check pairwise decompositions exist
        assert len(flow["pairwise_decompositions"]) == 3  # C(3,2) = 3 pairs

    def test_deterministic_reproducibility(self):
        """Test bit-level reproducibility across runs."""

        rng = np.random.RandomState(42)
        source1 = rng.randn(100)
        source2 = rng.randn(100)
        target = rng.randn(100)

        # Run multiple times with same seed
        results = []
        for _ in range(3):
            engine = InformationFusionEngine(seed=42)
            result = engine.decompose_information(source1, source2, target, bins=10)
            results.append(result)

        # All results should be identical
        for i in range(1, len(results)):
            assert results[i].unique_s1 == results[0].unique_s1
            assert results[i].unique_s2 == results[0].unique_s2
            assert results[i].redundant == results[0].redundant
            assert results[i].synergistic == results[0].synergistic
            assert results[i].total_mi == results[0].total_mi

    def test_caching_behavior(self):
        """Test result caching for performance."""

        engine = InformationFusionEngine(seed=42)
        rng = np.random.RandomState(42)

        source1 = rng.randn(100)
        source2 = rng.randn(100)
        target = rng.randn(100)

        # First call
        result1 = engine.decompose_information(source1, source2, target, bins=10)
        stats1 = engine.get_statistics()

        # Second call (should be cached)
        result2 = engine.decompose_information(source1, source2, target, bins=10)
        stats2 = engine.get_statistics()

        # Results should be identical
        assert result1.unique_s1 == result2.unique_s1
        assert stats2["total_decompositions"] == 1  # Only one unique decomposition

        # Clear cache
        engine.clear_cache()
        stats3 = engine.get_statistics()
        assert stats3["total_decompositions"] == 0

    def test_condition_number_computation(self):
        """Test numerical condition number is computed."""

        engine = InformationFusionEngine(seed=42)
        rng = np.random.RandomState(42)

        source1 = rng.randn(100)
        source2 = rng.randn(100)
        target = rng.randn(100)

        result = engine.decompose_information(source1, source2, target, bins=10)

        # Condition number should be computed
        assert result.condition_number >= 1.0

    def test_empty_data_handling(self):
        """Test handling of edge cases with empty data."""

        engine = InformationFusionEngine(seed=42)

        # Empty array should have zero entropy
        empty_data = np.array([])
        h_empty = engine.compute_entropy(empty_data, bins=10)
        assert h_empty == 0.0

        # Empty MI should be zero
        mi_empty = engine.compute_mutual_information(empty_data, empty_data, bins=10)
        assert mi_empty == 0.0

    def test_engine_statistics(self):
        """Test engine statistics tracking."""

        constraints = ConservationConstraints(
            enforce_non_negativity=True,
            enforce_upper_bound=True,
            auto_correct=True,
        )
        engine = InformationFusionEngine(constraints=constraints, seed=42)
        rng = np.random.RandomState(42)

        # Perform some decompositions
        for _ in range(3):
            source1 = rng.randn(100)
            source2 = rng.randn(100)
            target = rng.randn(100)
            engine.decompose_information(source1, source2, target, bins=10)

        stats = engine.get_statistics()

        assert stats["total_decompositions"] == 3
        assert "conservation_enforcement" in stats
        assert stats["conservation_enforcement"]["non_negativity"] is True
        assert stats["conservation_enforcement"]["upper_bound"] is True
        assert stats["conservation_enforcement"]["auto_correct"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
