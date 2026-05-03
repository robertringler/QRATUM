"""Tests for quantum core module."""

import pytest

# Try to import quantum modules
try:
    from quasim.quantum.core import QISKIT_AVAILABLE, QuantumBackend, QuantumConfig

    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False


@pytest.mark.skipif(not QUANTUM_AVAILABLE, reason="Quantum libraries not available")
class TestQuantumConfig:
    """Test QuantumConfig configuration class."""

    def test_default_config(self):
        """Test default configuration values."""

        config = QuantumConfig()
        assert config.backend_type == "simulator"
        assert config.shots == 1024
        assert config.seed == 42
        assert config.is_simulator

    def test_custom_config(self):
        """Test custom configuration."""

        config = QuantumConfig(backend_type="simulator", shots=2048, seed=123)
        assert config.shots == 2048
        assert config.seed == 123

    def test_low_shots_warning(self):
        """Test warning for low shot count."""

        with pytest.warns(UserWarning, match="shots.*very low"):
            config = QuantumConfig(shots=50)

    def test_ibmq_requires_token(self):
        """Test that IBMQ backend requires token."""

        with pytest.raises(ValueError, match="ibmq_token required"):
            QuantumConfig(backend_type="ibmq")


@pytest.mark.skipif(not QUANTUM_AVAILABLE, reason="Quantum libraries not available")
class TestQuantumBackend:
    """Test QuantumBackend wrapper."""

    def test_simulator_backend(self):
        """Test simulator backend setup."""

        config = QuantumConfig(backend_type="simulator")
        backend = QuantumBackend(config)

        assert backend.backend_name is not None
        assert backend.num_qubits > 0

    def test_backend_execution(self):
        """Test simple circuit execution."""

        from qiskit import QuantumCircuit

        config = QuantumConfig(backend_type="simulator", shots=100)
        backend = QuantumBackend(config)

        # Create simple Bell state circuit
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()

        # Execute
        result = backend.execute(circuit)
        counts = result.get_counts()

        # Should have some results
        assert len(counts) > 0
        # Bell state should have 00 and 11 outcomes
        assert "00" in counts or "11" in counts


def test_import_without_dependencies():
    """Test that module can be imported even without quantum libraries."""

    # This test always runs to verify graceful degradation
    from quasim import quantum

    assert quantum is not None
