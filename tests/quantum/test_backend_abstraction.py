"""Tests for quantum backend abstraction layer."""

import pytest

try:
    from quasim.quantum.core import (
        QISKIT_AVAILABLE,
        AbstractQuantumBackend,
        BackendTypeEnum,
        IBMQBackend,
        QiskitAerBackend,
        QuantumConfig,
        create_backend,
    )

    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False


@pytest.mark.skipif(not QUANTUM_AVAILABLE, reason="Quantum libraries not available")
class TestBackendTypeEnum:
    """Test BackendTypeEnum enumeration."""

    def test_enum_values(self):
        """Test that all expected backend types are defined."""
        assert BackendTypeEnum.SIMULATOR.value == "simulator"
        assert BackendTypeEnum.IBMQ.value == "ibmq"
        assert BackendTypeEnum.CUQUANTUM.value == "cuquantum"


@pytest.mark.skipif(not QUANTUM_AVAILABLE, reason="Quantum libraries not available")
class TestQuantumConfigValidation:
    """Test QuantumConfig validation enhancements."""

    def test_default_config(self):
        """Test default configuration with seed."""
        config = QuantumConfig()
        assert config.backend_type == "simulator"
        assert config.seed == 42
        assert config.shots == 1024

    def test_seed_warning_for_simulator(self):
        """Test warning when seed is None for simulator."""
        with pytest.warns(UserWarning, match="No seed provided"):
            config = QuantumConfig(seed=None)
            assert config.seed is None

    def test_ibmq_requires_token(self):
        """Test that IBMQ backend requires token."""
        with pytest.raises(ValueError, match="ibmq_token required"):
            QuantumConfig(backend_type="ibmq")

    def test_low_shots_warning(self):
        """Test warning for low shot count."""
        with pytest.warns(UserWarning, match="very low"):
            config = QuantumConfig(shots=50)
            assert config.shots == 50


@pytest.mark.skipif(not QUANTUM_AVAILABLE, reason="Quantum libraries not available")
class TestBackendFactory:
    """Test backend factory function."""

    def test_create_simulator_backend(self):
        """Test creating simulator backend."""
        config = QuantumConfig(backend_type="simulator", seed=42)
        backend = create_backend(config)

        assert isinstance(backend, QiskitAerBackend)
        assert isinstance(backend, AbstractQuantumBackend)
        assert backend.config.seed == 42

    def test_create_ibmq_backend_fails_without_token(self):
        """Test that IBMQ backend creation fails without token."""
        config = QuantumConfig(backend_type="simulator")  # Use simulator first
        config.backend_type = "ibmq"  # Change to IBMQ without token

        with pytest.raises(ValueError):
            create_backend(config)

    def test_cuquantum_not_implemented(self):
        """Test that cuQuantum backend raises NotImplementedError."""
        config = QuantumConfig(backend_type="cuquantum", seed=42)

        with pytest.raises(NotImplementedError, match="Phase 2"):
            create_backend(config)

    def test_unknown_backend_type(self):
        """Test that unknown backend type raises ValueError."""
        config = QuantumConfig(backend_type="simulator")
        config.backend_type = "invalid_backend"

        with pytest.raises(ValueError, match="Unknown backend type"):
            create_backend(config)


@pytest.mark.skipif(not QUANTUM_AVAILABLE, reason="Quantum libraries not available")
class TestQiskitAerBackend:
    """Test Qiskit Aer backend implementation."""

    def test_backend_initialization(self):
        """Test backend initializes correctly."""
        config = QuantumConfig(backend_type="simulator", seed=42)
        backend = QiskitAerBackend(config)

        assert backend.config.seed == 42
        assert backend.backend is not None
        assert "aer" in backend.backend_name.lower()

    def test_execute_simple_circuit(self):
        """Test executing a simple quantum circuit."""
        from qiskit import QuantumCircuit

        config = QuantumConfig(backend_type="simulator", shots=100, seed=42)
        backend = QiskitAerBackend(config)

        # Create simple Bell state
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()

        result = backend.execute_circuit(circuit)

        assert "counts" in result
        assert "success" in result
        assert result["success"] is True
        assert len(result["counts"]) > 0

    def test_deterministic_execution(self):
        """Test that same seed produces identical results."""
        from qiskit import QuantumCircuit

        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()

        # First execution
        config1 = QuantumConfig(backend_type="simulator", shots=100, seed=42)
        backend1 = QiskitAerBackend(config1)
        result1 = backend1.execute_circuit(circuit)

        # Second execution with same seed
        config2 = QuantumConfig(backend_type="simulator", shots=100, seed=42)
        backend2 = QiskitAerBackend(config2)
        result2 = backend2.execute_circuit(circuit)

        # Results should be identical
        assert result1["counts"] == result2["counts"]


@pytest.mark.skipif(not QUANTUM_AVAILABLE, reason="Quantum libraries not available")
class TestAbstractBackendInterface:
    """Test abstract backend interface."""

    def test_abstract_methods_required(self):
        """Test that AbstractQuantumBackend requires implementation of abstract methods."""
        config = QuantumConfig()

        # Cannot instantiate abstract class directly
        with pytest.raises(TypeError):
            AbstractQuantumBackend(config)

    def test_backend_name_property(self):
        """Test backend_name property works."""
        config = QuantumConfig(backend_type="simulator", seed=42)
        backend = QiskitAerBackend(config)

        name = backend.backend_name
        assert isinstance(name, str)
        assert len(name) > 0
