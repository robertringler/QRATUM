"""
Performance Tests for XENON v5

Tests that performance targets are met.
Certificate: QRATUM-HARDENING-20251215-V5
"""

import time

import numpy as np
import pytest

from qratum.bioinformatics.xenon.alignment import QuantumAlignmentEngine
from qratum.core.optimization import PerformanceProfiler


class TestPerformance:
    """Test performance targets for XENON."""

    def test_alignment_performance(self):
        """Test alignment meets performance targets."""
        engine = QuantumAlignmentEngine(seed=42)

        seq1 = "ACGT" * 25  # 100 bases
        seq2 = "ACGT" * 25

        start = time.perf_counter()
        result = engine.align(seq1, seq2)
        elapsed = time.perf_counter() - start

        # Should complete in reasonable time
        assert elapsed < 1.0, f"Alignment took {elapsed:.3f}s, target < 1.0s"
        assert result["score"] > 0, "Alignment should produce valid score"

    def test_profiler_comparison(self):
        """Test performance profiler can compare implementations."""
        profiler = PerformanceProfiler(tolerance=1e-6)

        def baseline_fn(size=100):
            return np.random.randn(size, size).sum()

        def optimized_fn(size=100):
            return np.random.randn(size, size).sum()

        profile = profiler.profile_function(
            baseline_fn, optimized_fn, inputs={"size": 100}, name="matrix_sum", num_runs=3
        )

        assert "speedup" in profile
        assert "equivalence" in profile
        assert profile["baseline"]["mean_time"] > 0
        assert profile["optimized"]["mean_time"] > 0

    def test_information_engine_performance(self):
        """Test information engine meets performance targets."""
        from qratum.bioinformatics.xenon.omics import InformationEngine

        engine = InformationEngine(seed=42)

        # Generate moderate-sized dataset
        data_x = np.random.randn(100, 1)
        data_y = np.random.randn(100, 1)

        start = time.perf_counter()
        result = engine.compute_mutual_information(data_x, data_y)
        elapsed = time.perf_counter() - start

        # Should complete in reasonable time
        assert elapsed < 2.0, f"MI computation took {elapsed:.3f}s, target < 2.0s"
        assert result["mutual_information"] >= 0, "MI should be non-negative"

    def test_neural_symbolic_performance(self):
        """Test neural-symbolic reasoner meets performance targets."""
        from qratum.bioinformatics.xenon.inference import NeuralSymbolicReasoner

        reasoner = NeuralSymbolicReasoner(seed=42, enable_neural=False)

        start = time.perf_counter()
        result = reasoner.reason("test query")
        elapsed = time.perf_counter() - start

        # Should complete quickly
        assert elapsed < 1.0, f"Reasoning took {elapsed:.3f}s, target < 1.0s"
        assert "trace" in result, "Reasoning must produce audit trace"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
