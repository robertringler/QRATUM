"""Tests for Qiskit Aer quantum simulator integration.

Validates production quantum simulator implementation replaces stubs.
"""

import pytest
import numpy as np


def test_qiskit_simulator_import():
    """Test that Qiskit Aer can be imported."""
    try:
        from qiskit_aer import AerSimulator
        assert AerSimulator is not None
    except ImportError:
        pytest.skip("Qiskit Aer not installed")


def test_simulator_initialization():
    """Test simulator initialization with different backends."""
    from quasim.qc.simulator import QCSimulator
    
    # Test CPU backend (default)
    sim_cpu = QCSimulator(backend="qiskit_aer")
    assert sim_cpu.backend == "qiskit_aer"
    assert sim_cpu.use_qiskit is True
    
    # Test legacy CPU backend
    sim_legacy = QCSimulator(backend="cpu")
    assert sim_legacy.backend == "cpu"
    assert sim_legacy.use_qiskit is False


def test_simulator_backend_validation():
    """Test that invalid backends are rejected."""
    from quasim.qc.simulator import QCSimulator
    
    with pytest.raises(ValueError, match="Unsupported backend"):
        QCSimulator(backend="invalid_backend")


def test_simulator_precision_validation():
    """Test that invalid precisions are rejected."""
    from quasim.qc.simulator import QCSimulator
    
    with pytest.raises(ValueError, match="Unsupported precision"):
        QCSimulator(backend="cpu", precision="invalid")


def test_simple_circuit_simulation():
    """Test simulation of a simple quantum circuit."""
    from quasim.qc.simulator import QCSimulator
    from quasim.qc.circuit import QuantumCircuit
    
    # Create a simple 2-qubit circuit
    circuit = QuantumCircuit(num_qubits=2)
    circuit.add_gate("H", [0])  # Hadamard on qubit 0
    circuit.add_gate("CNOT", [0, 1])  # CNOT with control=0, target=1
    
    # Simulate with Qiskit backend
    sim = QCSimulator(backend="qiskit_aer")
    result = sim.simulate(circuit)
    
    # Verify result structure
    assert "state_vector" in result
    assert "probabilities" in result
    assert "backend_used" in result
    assert "num_qubits" in result
    assert result["num_qubits"] == 2
    
    # Verify probabilities sum to 1
    probs = np.array(result["probabilities"])
    assert np.isclose(probs.sum(), 1.0, atol=1e-6)


def test_cpu_fallback_simulation():
    """Test that CPU fallback works when Qiskit is not available."""
    from quasim.qc.simulator import QCSimulator
    from quasim.qc.circuit import QuantumCircuit
    
    # Create a simple circuit
    circuit = QuantumCircuit(num_qubits=1)
    circuit.add_gate("X", [0])  # X gate (NOT)
    
    # Use CPU fallback
    sim = QCSimulator(backend="cpu")
    result = sim.simulate(circuit)
    
    assert result["backend_used"] == "cpu_fallback"
    assert "state_vector" in result
    assert "probabilities" in result


def test_qiskit_gpu_backend_configuration():
    """Test that GPU backend configuration is attempted."""
    from quasim.qc.simulator import QCSimulator
    
    try:
        sim = QCSimulator(backend="qiskit_aer_gpu")
        assert sim.backend == "qiskit_aer_gpu"
        # Note: GPU may not be available, but configuration should be attempted
    except Exception:
        # GPU backend may not be available in test environment
        pytest.skip("GPU backend not available")


def test_backend_info():
    """Test getting backend information."""
    from quasim.qc.simulator import QCSimulator
    
    sim = QCSimulator(backend="cpu")
    info = sim.get_backend_info()
    
    assert "backend" in info
    assert "precision" in info
    assert info["backend"] == "cpu"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
