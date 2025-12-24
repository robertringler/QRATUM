"""Tests for compression validators."""

import numpy as np

from benchmarks.compression.validators.compression_ratio import (
    aggregate_compression_statistics, compute_compression_metrics)
from benchmarks.compression.validators.fidelity import (
    compute_fidelity, validate_fidelity_bound)


class TestFidelityComputation:
    """Tests for fidelity calculations."""

    def test_identical_states(self):
        """Identical states should have fidelity = 1."""

        state = np.array([1, 0], dtype=complex)
        fidelity = compute_fidelity(state, state)
        assert abs(fidelity - 1.0) < 1e-10

    def test_orthogonal_states(self):
        """Orthogonal states should have fidelity = 0."""

        state1 = np.array([1, 0], dtype=complex)
        state2 = np.array([0, 1], dtype=complex)
        fidelity = compute_fidelity(state1, state2)
        assert abs(fidelity) < 1e-10

    def test_fidelity_range(self):
        """Fidelity should be in range [0, 1]."""

        state1 = np.random.randn(8) + 1j * np.random.randn(8)
        state2 = np.random.randn(8) + 1j * np.random.randn(8)
        state1 /= np.linalg.norm(state1)
        state2 /= np.linalg.norm(state2)

        fidelity = compute_fidelity(state1, state2)
        assert 0.0 <= fidelity <= 1.0


class TestFidelityValidation:
    """Tests for fidelity validation."""

    def test_meets_threshold(self):
        """Fidelity above threshold should pass."""

        assert validate_fidelity_bound(0.996, 0.995)

    def test_below_threshold(self):
        """Fidelity below threshold should fail."""

        assert not validate_fidelity_bound(0.994, 0.995)

    def test_equal_threshold(self):
        """Fidelity equal to threshold should pass."""

        assert validate_fidelity_bound(0.995, 0.995)


class TestCompressionMetrics:
    """Tests for compression ratio computation."""

    def test_basic_ratio(self):
        """Basic compression ratio calculation."""

        metrics = compute_compression_metrics(1024, 64)
        assert abs(metrics["compression_ratio"] - 16.0) < 1e-6
        assert abs(metrics["space_savings"] - 93.75) < 1e-6

    def test_no_compression(self):
        """No compression case."""

        metrics = compute_compression_metrics(100, 100)
        assert abs(metrics["compression_ratio"] - 1.0) < 1e-6
        assert abs(metrics["space_savings"] - 0.0) < 1e-6


class TestAggregateStatistics:
    """Tests for aggregate statistics."""

    def test_basic_stats(self):
        """Basic statistics computation."""

        ratios = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = aggregate_compression_statistics(ratios)
        assert abs(stats["mean"] - 30.0) < 1e-6
        assert abs(stats["median"] - 30.0) < 1e-6
        assert abs(stats["min"] - 10.0) < 1e-6
        assert abs(stats["max"] - 50.0) < 1e-6
