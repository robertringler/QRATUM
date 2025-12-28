"""Tests for hybrid quantum backends."""

import pytest

# Try to import hybrid quantum modules
try:
    from quasim.hybrid_quantum.backends import (
        AzureQuantumHybridBackend,
        BackendProvider,
        BraketHybridBackend,
        ExecutionResult,
        ExecutionStatus,
        HybridQuantumBackend,
        HybridQuantumConfig,
        IBMHybridBackend,
        IonQHybridBackend,
        QuantinuumHybridBackend,
    )

    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False

# Try to import qiskit for circuit creation
try:
    from qiskit import QuantumCircuit

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

# Try to import qiskit-aer for simulator
try:
    from qiskit_aer import AerSimulator

    QISKIT_AER_AVAILABLE = True
except ImportError:
    QISKIT_AER_AVAILABLE = False


class TestHybridQuantumConfig:
    """Test HybridQuantumConfig configuration class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HybridQuantumConfig()
        assert config.provider == "simulator"
        assert config.shots == 1024
        assert config.seed == 42
        assert config.require_verification is True
        assert config.enable_rollback is True
        assert config.is_simulator is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = HybridQuantumConfig(
            provider="simulator",
            shots=2048,
            seed=123,
            max_circuit_depth=500,
        )
        assert config.shots == 2048
        assert config.seed == 123
        assert config.max_circuit_depth == 500

    def test_low_shots_warning(self):
        """Test warning for low shot count."""
        with pytest.warns(UserWarning, match="very low"):
            config = HybridQuantumConfig(shots=50)

    def test_real_backend_requires_token(self):
        """Test that real backends require API token."""
        with pytest.raises(ValueError, match="api_token required"):
            HybridQuantumConfig(provider="ibm")

    def test_high_circuit_depth_warning(self):
        """Test warning for high circuit depth."""
        with pytest.warns(UserWarning, match="NISQ practical limits"):
            config = HybridQuantumConfig(max_circuit_depth=6000)


class TestBackendEnums:
    """Test backend enumeration types."""

    def test_backend_provider_values(self):
        """Test BackendProvider enum values."""
        assert BackendProvider.IBM.value == "ibm"
        assert BackendProvider.IONQ.value == "ionq"
        assert BackendProvider.QUANTINUUM.value == "quantinuum"
        assert BackendProvider.AZURE.value == "azure"
        assert BackendProvider.BRAKET.value == "braket"
        assert BackendProvider.SIMULATOR.value == "simulator"

    def test_execution_status_values(self):
        """Test ExecutionStatus enum values."""
        assert ExecutionStatus.PENDING.value == "pending"
        assert ExecutionStatus.COMPLETED.value == "completed"
        assert ExecutionStatus.FAILED.value == "failed"
        assert ExecutionStatus.AWAITING_APPROVAL.value == "awaiting_approval"


class TestExecutionResult:
    """Test ExecutionResult dataclass."""

    def test_execution_result_creation(self):
        """Test creating ExecutionResult."""
        result = ExecutionResult(
            execution_id="test-123",
            counts={"00": 500, "11": 500},
            raw_data={},
            execution_time=1.5,
            success=True,
        )
        assert result.execution_id == "test-123"
        assert result.counts == {"00": 500, "11": 500}
        assert result.success is True
        assert result.timestamp != ""

    def test_execution_result_defaults(self):
        """Test ExecutionResult default values."""
        result = ExecutionResult(
            execution_id="test",
            counts={},
            raw_data={},
            execution_time=0.0,
        )
        assert result.queue_time == 0.0
        assert result.verification_required is True
        assert result.error_message is None


@pytest.mark.skipif(not HYBRID_AVAILABLE, reason="Hybrid quantum module not available")
@pytest.mark.skipif(not QISKIT_AER_AVAILABLE, reason="Qiskit Aer not available")
class TestIBMHybridBackend:
    """Test IBM hybrid backend."""

    def test_simulator_initialization(self):
        """Test IBM backend initializes in simulator mode."""
        config = HybridQuantumConfig(provider="simulator", seed=42)
        backend = IBMHybridBackend(config)

        info = backend.get_backend_info()
        assert info["provider"] == "ibm"
        assert info["is_simulator"] is True

    @pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit not available")
    def test_circuit_execution(self):
        """Test executing circuit on IBM simulator."""
        config = HybridQuantumConfig(provider="simulator", shots=100, seed=42)
        backend = IBMHybridBackend(config)

        # Create Bell state circuit
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()

        result = backend.execute_circuit(circuit)

        assert result.success is True
        assert len(result.counts) > 0
        assert result.provenance_hash != ""

    @pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit not available")
    def test_circuit_validation(self):
        """Test circuit validation."""
        config = HybridQuantumConfig(provider="simulator")
        backend = IBMHybridBackend(config)

        circuit = QuantumCircuit(2)
        circuit.h(0)

        is_valid, message = backend.validate_circuit(circuit)
        assert is_valid is True

    def test_none_circuit_validation(self):
        """Test that None circuit fails validation."""
        config = HybridQuantumConfig(provider="simulator")
        backend = IBMHybridBackend(config)

        is_valid, message = backend.validate_circuit(None)
        assert is_valid is False
        assert "None" in message

    @pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit not available")
    def test_provenance_hash_computation(self):
        """Test provenance hash is deterministic."""
        config = HybridQuantumConfig(provider="simulator", seed=42)
        backend = IBMHybridBackend(config)

        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        hash1 = backend.compute_provenance_hash(circuit, {"shots": 100})
        hash2 = backend.compute_provenance_hash(circuit, {"shots": 100})

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex


@pytest.mark.skipif(not HYBRID_AVAILABLE, reason="Hybrid quantum module not available")
class TestOtherBackends:
    """Test other hybrid backends (stub implementations)."""

    def test_ionq_backend_init(self):
        """Test IonQ backend initializes."""
        config = HybridQuantumConfig(provider="simulator", seed=42)
        backend = IonQHybridBackend(config)

        info = backend.get_backend_info()
        assert info["provider"] == "ionq"
        assert info["connectivity"] == "all-to-all"

    def test_quantinuum_backend_init(self):
        """Test Quantinuum backend initializes."""
        config = HybridQuantumConfig(provider="simulator", seed=42)
        backend = QuantinuumHybridBackend(config)

        info = backend.get_backend_info()
        assert info["provider"] == "quantinuum"
        assert info["fidelity"] == "highest_available"

    def test_azure_backend_init(self):
        """Test Azure Quantum backend initializes."""
        config = HybridQuantumConfig(provider="simulator", seed=42)
        backend = AzureQuantumHybridBackend(config)

        info = backend.get_backend_info()
        assert info["provider"] == "azure"
        assert "ionq" in info["available_providers"]

    def test_braket_backend_init(self):
        """Test AWS Braket backend initializes."""
        config = HybridQuantumConfig(provider="simulator", seed=42)
        backend = BraketHybridBackend(config)

        info = backend.get_backend_info()
        assert info["provider"] == "braket"
        assert "ionq" in info["available_providers"]


@pytest.mark.skipif(not HYBRID_AVAILABLE, reason="Hybrid quantum module not available")
@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit not available")
class TestDeterministicExecution:
    """Test deterministic execution with seeds."""

    def test_same_seed_same_results(self):
        """Test that same seed produces identical results."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()

        # First execution
        config1 = HybridQuantumConfig(provider="simulator", shots=100, seed=42)
        backend1 = IBMHybridBackend(config1)
        result1 = backend1.execute_circuit(circuit)

        # Second execution with same seed
        config2 = HybridQuantumConfig(provider="simulator", shots=100, seed=42)
        backend2 = IBMHybridBackend(config2)
        result2 = backend2.execute_circuit(circuit)

        assert result1.counts == result2.counts
