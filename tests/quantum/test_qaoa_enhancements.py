"""Tests for QAOA enhancements."""

import pytest

try:
    from quasim.quantum.core import QuantumConfig
    from quasim.quantum.qaoa_optimization import QAOA, QAOAResult

    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False


@pytest.mark.skipif(not QUANTUM_AVAILABLE, reason="Quantum libraries not available")
class TestQAOAResultEnhancements:
    """Test QAOAResult dataclass enhancements."""

    def test_qaoa_result_with_new_fields(self):
        """Test QAOAResult includes new fields."""
        import numpy as np

        result = QAOAResult(
            solution="1010",
            energy=-2.0,
            optimal_params=np.array([0.5, 1.0, 0.3, 0.7]),
            approximation_ratio=0.9,
            n_iterations=50,
            execution_id="qaoa-001",
        )

        assert result.solution == "1010"
        assert result.execution_id == "qaoa-001"
        assert result.approximation_ratio == 0.9

    def test_qaoa_result_repr(self):
        """Test QAOAResult string representation."""
        import numpy as np

        result = QAOAResult(
            solution="11",
            energy=-1.5,
            optimal_params=np.array([0.1, 0.2]),
            approximation_ratio=0.85,
            n_iterations=30,
            classical_optimal=-1.7,
        )

        repr_str = repr(result)
        assert "solution=11" in repr_str
        assert "approx_ratio=0.8" in repr_str


@pytest.mark.skipif(not QUANTUM_AVAILABLE, reason="Quantum libraries not available")
class TestQAOABackendSupport:
    """Test QAOA backend abstraction support."""

    def test_init_with_config(self):
        """Test initialization with QuantumConfig."""
        config = QuantumConfig(backend_type="simulator", seed=42)
        qaoa = QAOA(config, p_layers=2)

        assert qaoa.config.seed == 42
        assert qaoa.p_layers == 2
        assert qaoa.backend is not None

    def test_p_layers_validation(self):
        """Test that p_layers must be at least 1."""
        config = QuantumConfig(backend_type="simulator", seed=42)

        with pytest.raises(ValueError, match="p_layers must be at least 1"):
            QAOA(config, p_layers=0)


@pytest.mark.skipif(not QUANTUM_AVAILABLE, reason="Quantum libraries not available")
class TestQAOABackwardsCompatibility:
    """Test that existing QAOA functionality still works."""

    def test_solve_maxcut_still_works(self):
        """Test that original solve_maxcut method works."""
        config = QuantumConfig(backend_type="simulator", seed=42, shots=100)
        qaoa = QAOA(config, p_layers=1)

        # Small graph for quick testing
        edges = [(0, 1), (1, 2)]
        result = qaoa.solve_maxcut(
            edges=edges, classical_reference=False, max_iterations=5
        )

        assert isinstance(result, QAOAResult)
        assert len(result.solution) == 3  # 3 nodes
        assert result.success is True

    def test_solve_ising_still_works(self):
        """Test that original solve_ising method works."""
        import numpy as np

        config = QuantumConfig(backend_type="simulator", seed=42, shots=100)
        qaoa = QAOA(config, p_layers=1)

        # Small coupling matrix
        coupling = np.array([[0, -1], [-1, 0]])
        result = qaoa.solve_ising(coupling_matrix=coupling, max_iterations=5)

        assert isinstance(result, QAOAResult)
        assert len(result.solution) == 2  # 2 spins
        assert result.success is True
