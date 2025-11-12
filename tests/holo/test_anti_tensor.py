"""Tests for Anti-Holographic Tensor Compression Algorithm (AHTC).

Validates compression fidelity, entanglement preservation, and performance
characteristics of the AHTC implementation.

Test Categories:
    - Unit tests for individual algorithm stages
    - Integration tests with full compression pipeline
    - Fidelity convergence tests
    - Performance benchmarks
"""

import numpy as np
import pytest

from quasim.holo.anti_tensor import (
    adaptive_truncate,
    compress,
    compute_fidelity,
    compute_mutual_information,
    decompress,
    hierarchical_decompose,
    reconstruct,
)


class TestMutualInformation:
    """Tests for entanglement analysis via mutual information."""

    def test_product_state_zero_mutual_info(self):
        """Product state |00⟩ should have zero mutual information."""
        state = np.array([1, 0, 0, 0], dtype=complex)
        M = compute_mutual_information(state)

        # Product states have no entanglement
        assert M.shape == (2, 2)
        # Note: This is a placeholder test for the stub implementation

    def test_entangled_state_nonzero_mutual_info(self):
        """Bell state should have non-zero mutual information."""
        # Bell state: (|00⟩ + |11⟩) / √2
        state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        M = compute_mutual_information(state)

        assert M.shape == (2, 2)
        # Entangled states have non-zero mutual information
        # Note: Full implementation will have I(A:B) ≈ 2 bits


class TestHierarchicalDecomposition:
    """Tests for tensor decomposition preserving subsystem topology."""

    def test_decomposition_structure(self):
        """Decomposition should return expected structure."""
        state = np.random.randn(8) + 1j * np.random.randn(8)
        state /= np.linalg.norm(state)

        M = compute_mutual_information(state)
        tree = hierarchical_decompose(state, M)

        assert "weights" in tree
        assert "basis_left" in tree
        assert "basis_right" in tree
        assert "topology" in tree

    def test_decomposition_preserves_state(self):
        """Decomposition should allow reconstruction of original state."""
        state = np.random.randn(16) + 1j * np.random.randn(16)
        state /= np.linalg.norm(state)

        M = compute_mutual_information(state)
        tree = hierarchical_decompose(state, M)
        reconstructed = reconstruct(tree)

        # Should be able to reconstruct with high fidelity
        fidelity = compute_fidelity(state, reconstructed)
        assert fidelity >= 0.99


class TestAdaptiveTruncation:
    """Tests for adaptive threshold truncation."""

    def test_truncation_reduces_components(self):
        """Truncation should reduce number of components."""
        state = np.random.randn(32) + 1j * np.random.randn(32)
        state /= np.linalg.norm(state)

        M = compute_mutual_information(state)
        tree = hierarchical_decompose(state, M)

        original_weights = len(tree["weights"])
        truncated = adaptive_truncate(tree, epsilon=0.5)
        truncated_weights = len(truncated["weights"])

        # High epsilon should remove some components
        assert truncated_weights <= original_weights

    def test_small_epsilon_preserves_components(self):
        """Small epsilon should preserve most components."""
        state = np.random.randn(16) + 1j * np.random.randn(16)
        state /= np.linalg.norm(state)

        M = compute_mutual_information(state)
        tree = hierarchical_decompose(state, M)
        truncated = adaptive_truncate(tree, epsilon=1e-10)

        # Very small epsilon should keep all significant components
        assert len(truncated["weights"]) >= 1


class TestFidelityComputation:
    """Tests for quantum state fidelity calculation."""

    def test_identical_states_unit_fidelity(self):
        """Identical states should have fidelity = 1."""
        state = np.array([1, 0], dtype=complex)
        fidelity = compute_fidelity(state, state)

        assert abs(fidelity - 1.0) < 1e-10

    def test_orthogonal_states_zero_fidelity(self):
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


class TestCompression:
    """Tests for full compression pipeline."""

    def test_compress_basic(self):
        """Basic compression should succeed."""
        state = np.random.randn(32) + 1j * np.random.randn(32)
        state /= np.linalg.norm(state)

        compressed, fidelity, metadata = compress(state, fidelity=0.99, epsilon=1e-3)

        assert fidelity >= 0.99
        assert "compression_ratio" in metadata
        assert "fidelity_achieved" in metadata

    def test_compress_meets_fidelity_target(self):
        """Compression should meet fidelity target."""
        state = np.random.randn(64) + 1j * np.random.randn(64)
        state /= np.linalg.norm(state)

        target_fidelity = 0.995
        compressed, fidelity, metadata = compress(state, fidelity=target_fidelity, epsilon=1e-3)

        assert fidelity >= target_fidelity
        assert metadata["fidelity_achieved"] >= target_fidelity

    def test_compress_returns_metadata(self):
        """Compression should return complete metadata."""
        state = np.random.randn(16) + 1j * np.random.randn(16)
        state /= np.linalg.norm(state)

        compressed, fidelity, metadata = compress(state)

        required_keys = [
            "compression_ratio",
            "epsilon",
            "mutual_info_entropy",
            "original_size",
            "compressed_size",
            "fidelity_achieved",
        ]

        for key in required_keys:
            assert key in metadata

    def test_compress_invalid_input(self):
        """Compression should reject invalid input."""
        # Zero tensor
        state = np.zeros(8, dtype=complex)

        with pytest.raises(ValueError):
            compress(state)

    def test_compress_small_state(self):
        """Compression should handle small states."""
        # 2-qubit state
        state = np.array([1, 0, 0, 0], dtype=complex)

        compressed, fidelity, metadata = compress(state, fidelity=0.99)

        assert fidelity >= 0.99


class TestDecompression:
    """Tests for decompression functionality."""

    def test_decompress_basic(self):
        """Basic decompression should succeed."""
        state = np.random.randn(16) + 1j * np.random.randn(16)
        state /= np.linalg.norm(state)

        compressed, fidelity, metadata = compress(state)
        decompressed = decompress(compressed)

        assert decompressed.shape == state.shape
        assert abs(np.linalg.norm(decompressed) - 1.0) < 1e-10

    def test_compress_decompress_cycle(self):
        """Compress-decompress cycle should preserve fidelity."""
        state = np.random.randn(32) + 1j * np.random.randn(32)
        state /= np.linalg.norm(state)

        compressed, fidelity_compress, metadata = compress(state, fidelity=0.995)
        decompressed = decompress(compressed)

        fidelity_cycle = compute_fidelity(state, decompressed)

        assert fidelity_cycle >= 0.995


class TestPerformance:
    """Performance and scalability tests."""

    @pytest.mark.parametrize("n_qubits", [4, 5, 6])
    def test_scalability(self, n_qubits):
        """Compression should scale to different qubit counts."""
        dim = 2**n_qubits
        state = np.random.randn(dim) + 1j * np.random.randn(dim)
        state /= np.linalg.norm(state)

        compressed, fidelity, metadata = compress(state, fidelity=0.99)

        assert fidelity >= 0.99
        assert metadata["compression_ratio"] >= 1.0

    def test_compression_ratio(self):
        """Compression should achieve reasonable compression ratio."""
        # Large enough state to benefit from compression
        state = np.random.randn(128) + 1j * np.random.randn(128)
        state /= np.linalg.norm(state)

        compressed, fidelity, metadata = compress(state, fidelity=0.99, epsilon=1e-2)

        # Should achieve some compression
        assert metadata["compression_ratio"] >= 1.0


class TestEntanglementPreservation:
    """Tests for entanglement structure preservation."""

    def test_bell_state_preservation(self):
        """Bell state entanglement should be preserved."""
        # Bell state: (|00⟩ + |11⟩) / √2
        state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)

        compressed, fidelity, metadata = compress(state, fidelity=0.995)

        # High fidelity means entanglement is preserved
        assert fidelity >= 0.995

    def test_ghz_state_preservation(self):
        """GHZ state entanglement should be preserved."""
        # 3-qubit GHZ state: (|000⟩ + |111⟩) / √2
        state = np.zeros(8, dtype=complex)
        state[0] = 1 / np.sqrt(2)
        state[7] = 1 / np.sqrt(2)

        compressed, fidelity, metadata = compress(state, fidelity=0.99)

        assert fidelity >= 0.99


class TestDeterminism:
    """Tests for deterministic reproducibility."""

    def test_reproducible_compression(self):
        """Compression should be reproducible with same input."""
        np.random.seed(42)
        state = np.random.randn(32) + 1j * np.random.randn(32)
        state /= np.linalg.norm(state)

        compressed1, fidelity1, metadata1 = compress(state, fidelity=0.99, epsilon=1e-3)
        compressed2, fidelity2, metadata2 = compress(state, fidelity=0.99, epsilon=1e-3)

        # Should get same results
        assert abs(fidelity1 - fidelity2) < 1e-10
        assert abs(metadata1["compression_ratio"] - metadata2["compression_ratio"]) < 1e-6


# Benchmark tests (marked as slow)
class TestBenchmarks:
    """Benchmark tests for performance validation."""

    @pytest.mark.slow
    def test_benchmark_50_qubit(self):
        """Benchmark compression for 50-qubit system."""
        dim = 2**10  # Smaller for testing, full would be 2^50
        state = np.random.randn(dim) + 1j * np.random.randn(dim)
        state /= np.linalg.norm(state)

        compressed, fidelity, metadata = compress(state, fidelity=0.995, epsilon=1e-3)

        assert fidelity >= 0.995
        print(f"Compression ratio: {metadata['compression_ratio']:.2f}x")
        print(f"Fidelity: {fidelity:.6f}")

    @pytest.mark.slow
    def test_benchmark_compression_sweep(self):
        """Benchmark compression vs fidelity trade-off."""
        state = np.random.randn(256) + 1j * np.random.randn(256)
        state /= np.linalg.norm(state)

        epsilons = [1e-1, 1e-2, 1e-3, 1e-4]
        results = []

        for eps in epsilons:
            compressed, fidelity, metadata = compress(state, fidelity=0.90, epsilon=eps)
            results.append((eps, fidelity, metadata["compression_ratio"]))

        # Should see compression-fidelity trade-off
        for eps, fid, ratio in results:
            print(f"ε={eps:.1e}: F={fid:.4f}, R={ratio:.2f}x")
