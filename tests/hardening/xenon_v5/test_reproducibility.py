"""

Reproducibility Tests for XENON v5

Tests deterministic behavior across all XENON components.
Certificate: QRATUM-HARDENING-20251215-V5
"""

import numpy as np
import pytest

from qratum.bioinformatics.xenon.alignment import QuantumAlignmentEngine
from qratum.bioinformatics.xenon.omics import InformationEngine
from qratum.core.reproducibility import ReproducibilityManager, get_global_seed


class TestReproducibility:
    """Test reproducibility across XENON components."""

    def test_global_seed(self):
        """Test global seed is locked for production."""

        seed = get_global_seed()
        assert seed == 42, "Global seed must be 42 for production"

    def test_reproducibility_manager_determinism(self):
        """Test reproducibility manager sets up deterministic mode."""

        manager = ReproducibilityManager()
        manager.setup_deterministic_mode()

        status = manager.verify_determinism()
        assert status["seed"] == 42
        assert status["initialized"] is True

    def test_alignment_reproducibility(self):
        """Test alignment produces identical results across runs."""

        seq1 = "ACGTACGT"
        seq2 = "ACGTGCGT"

        results = []
        for _ in range(3):
            engine = QuantumAlignmentEngine(seed=42)
            result = engine.align(seq1, seq2)
            results.append(result["score"])

        # All results should be identical
        assert len(set(results)) == 1, "Alignment scores must be identical across runs"

    def test_information_engine_reproducibility(self):
        """Test information engine produces identical results."""

        data_x = np.random.RandomState(42).randn(100, 1)
        data_y = np.random.RandomState(42).randn(100, 1)

        results = []
        for _ in range(3):
            engine = InformationEngine(seed=42)
            result = engine.compute_mutual_information(data_x, data_y)
            results.append(result["mutual_information"])

        # All results should be very close (within numerical precision)
        assert np.std(results) < 1e-10, "MI estimates must be reproducible"

    def test_cross_platform_determinism(self):
        """Test determinism across different numpy operations."""

        manager = ReproducibilityManager(seed=42)
        manager.setup_deterministic_mode()

        # Generate random arrays
        arr1 = np.random.randn(10, 10)
        arr2 = np.random.randn(10, 10)

        # Reset and regenerate
        manager2 = ReproducibilityManager(seed=42)
        manager2.setup_deterministic_mode()

        arr3 = np.random.randn(10, 10)
        arr4 = np.random.randn(10, 10)

        # Should be identical
        assert np.allclose(arr1, arr3), "Arrays must be identical with same seed"
        assert np.allclose(arr2, arr4), "Arrays must be identical with same seed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
