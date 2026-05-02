"""Tests for VQE enhancements."""

import pytest

try:
    from quasim.quantum.core import QuantumConfig
    from quasim.quantum.vqe_molecule import MolecularVQE, VQEResult

    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False


@pytest.mark.skipif(not QUANTUM_AVAILABLE, reason="Quantum libraries not available")
class TestVQEResultEnhancements:
    """Test VQEResult dataclass enhancements."""

    def test_vqe_result_with_new_fields(self):
        """Test VQEResult includes new fields."""
        import numpy as np

        result = VQEResult(
            energy=-1.137,
            optimal_params=np.array([0.1, 0.2, 0.3]),
            n_iterations=50,
            n_evaluations=150,
            success=True,
            convergence=True,
            execution_id="test-001",
        )

        assert result.energy == -1.137
        assert result.convergence is True
        assert result.execution_id == "test-001"

    def test_vqe_result_repr(self):
        """Test VQEResult string representation."""
        import numpy as np

        result = VQEResult(
            energy=-1.137,
            optimal_params=np.array([0.1, 0.2]),
            n_iterations=10,
            n_evaluations=30,
            success=True,
            classical_energy=-1.140,
            error_vs_classical=0.003,
        )

        repr_str = repr(result)
        assert "energy=-1.137" in repr_str
        assert "classical_ref=-1.140" in repr_str


@pytest.mark.skipif(not QUANTUM_AVAILABLE, reason="Quantum libraries not available")
class TestMolecularVQEBackendSupport:
    """Test MolecularVQE backend abstraction support."""

    def test_init_with_config(self):
        """Test initialization with QuantumConfig."""
        config = QuantumConfig(backend_type="simulator", seed=42)
        vqe = MolecularVQE(config)

        assert vqe.config.seed == 42
        assert vqe.backend is not None

    def test_compute_molecule_energy_h2(self):
        """Test compute_molecule_energy for H2."""
        config = QuantumConfig(backend_type="simulator", seed=42, shots=100)
        vqe = MolecularVQE(config)

        # This should delegate to compute_h2_energy
        result = vqe.compute_molecule_energy(
            molecule="H2", bond_length=0.735, use_classical_reference=False, max_iterations=5
        )

        assert isinstance(result, VQEResult)
        assert result.energy < 0  # Should be negative for bound state
        assert result.convergence is True

    def test_compute_molecule_energy_lih_not_implemented(self):
        """Test that LiH raises NotImplementedError."""
        config = QuantumConfig(backend_type="simulator", seed=42)
        vqe = MolecularVQE(config)

        with pytest.raises(NotImplementedError, match="Phase 2"):
            vqe.compute_molecule_energy(molecule="LiH", bond_length=1.5)

    def test_compute_molecule_energy_unsupported(self):
        """Test that unsupported molecule raises ValueError."""
        config = QuantumConfig(backend_type="simulator", seed=42)
        vqe = MolecularVQE(config)

        with pytest.raises(ValueError, match="Unsupported molecule"):
            vqe.compute_molecule_energy(molecule="CH4", bond_length=1.0)


@pytest.mark.skipif(not QUANTUM_AVAILABLE, reason="Quantum libraries not available")
class TestVQEBackwardsCompatibility:
    """Test that existing VQE functionality still works."""

    def test_compute_h2_energy_still_works(self):
        """Test that original compute_h2_energy method works."""
        config = QuantumConfig(backend_type="simulator", seed=42, shots=100)
        vqe = MolecularVQE(config)

        result = vqe.compute_h2_energy(
            bond_length=0.735, use_classical_reference=False, max_iterations=5
        )

        assert isinstance(result, VQEResult)
        assert result.success is True
        assert result.n_iterations > 0
