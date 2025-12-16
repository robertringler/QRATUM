"""Tests for QRATUM core functionality.

Tests simulator, circuit builder, gates, measurements, and state vectors.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import qratum
from qratum import Circuit, Simulator, StateVector, gates


class TestGates:
    """Test quantum gate definitions."""

    def test_pauli_gates(self):
        """Test Pauli gates are unitary."""
        for gate in [gates.X, gates.Y, gates.Z]:
            identity = gate @ np.conj(gate.T)
            assert np.allclose(identity, np.eye(2))

    def test_hadamard(self):
        """Test Hadamard gate."""
        # H is self-inverse
        result = gates.H @ gates.H
        assert np.allclose(result, np.eye(2))

    def test_cnot(self):
        """Test CNOT gate."""
        # CNOT is self-inverse
        result = gates.CNOT @ gates.CNOT
        assert np.allclose(result, np.eye(4))

    def test_rotation_gates(self):
        """Test rotation gates."""
        # RX(0) should be identity
        rx_zero = gates.RX(0)
        assert np.allclose(rx_zero, np.eye(2))

        # RX(2π) should be -I
        rx_2pi = gates.RX(2 * np.pi)
        assert np.allclose(rx_2pi, -np.eye(2))


class TestCircuit:
    """Test circuit builder."""

    def test_circuit_creation(self):
        """Test creating a circuit."""
        circuit = Circuit(2)
        assert circuit.num_qubits == 2
        assert circuit.gate_count() == 0

    def test_single_qubit_gates(self):
        """Test adding single-qubit gates."""
        circuit = Circuit(1)
        circuit.h(0).x(0).y(0).z(0)
        assert circuit.gate_count() == 4

    def test_two_qubit_gates(self):
        """Test adding two-qubit gates."""
        circuit = Circuit(2)
        circuit.cnot(0, 1).cz(0, 1).swap(0, 1)
        assert circuit.gate_count() == 3

    def test_circuit_depth(self):
        """Test circuit depth calculation."""
        circuit = Circuit(2)
        circuit.h(0)
        circuit.h(1)
        assert circuit.depth() == 1  # Parallel gates

        circuit.cnot(0, 1)
        assert circuit.depth() == 2  # Sequential

    def test_circuit_copy(self):
        """Test circuit copying."""
        circuit1 = Circuit(2)
        circuit1.h(0).cnot(0, 1)

        circuit2 = circuit1.copy()
        assert circuit2.gate_count() == circuit1.gate_count()
        assert circuit2.num_qubits == circuit1.num_qubits


class TestStateVector:
    """Test state vector operations."""

    def test_zero_state(self):
        """Test |0⟩ state creation."""
        state = StateVector.zero_state(2)
        assert state.num_qubits == 2
        assert len(state.data) == 4
        assert np.isclose(state.data[0], 1.0)
        assert np.isclose(np.sum(np.abs(state.data[1:])), 0.0)

    def test_normalization(self):
        """Test state vector normalization."""
        data = np.array([1, 1, 1, 1], dtype=complex)
        state = StateVector(data)
        state.normalize()

        norm = np.linalg.norm(state.data)
        assert np.isclose(norm, 1.0)

    def test_probabilities(self):
        """Test probability calculation."""
        state = StateVector.zero_state(1)
        probs = state.probabilities()
        assert np.isclose(probs[0], 1.0)
        assert np.isclose(probs[1], 0.0)

    def test_inner_product(self):
        """Test inner product."""
        state1 = StateVector.zero_state(1)
        state2 = StateVector.zero_state(1)
        inner = state1.inner_product(state2)
        assert np.isclose(inner, 1.0)


class TestSimulator:
    """Test quantum simulator."""

    def test_simulator_creation(self):
        """Test creating a simulator."""
        sim = Simulator(backend="cpu", precision="fp32", seed=42)
        assert sim.backend == "cpu"
        assert sim.precision == "fp32"
        assert sim.seed == 42

    def test_bell_state_simulation(self):
        """Test simulating Bell state."""
        circuit = Circuit(2)
        circuit.h(0)
        circuit.cnot(0, 1)

        sim = Simulator(backend="cpu", seed=42)
        result = sim.run(circuit, shots=1000)

        counts = result.get_counts()
        assert len(counts) == 2
        assert "00" in counts
        assert "11" in counts

        # Check roughly equal probabilities
        prob_00 = counts["00"] / 1000
        prob_11 = counts["11"] / 1000
        assert 0.4 < prob_00 < 0.6
        assert 0.4 < prob_11 < 0.6

    def test_statevector_simulation(self):
        """Test state vector simulation without measurement."""
        circuit = Circuit(1)
        circuit.h(0)

        sim = Simulator(backend="cpu")
        state = sim.run_statevector(circuit)

        # After Hadamard, should be |+⟩ = (|0⟩ + |1⟩)/√2
        assert np.isclose(abs(state.data[0]), 1 / np.sqrt(2))
        assert np.isclose(abs(state.data[1]), 1 / np.sqrt(2))

    def test_deterministic_simulation(self):
        """Test deterministic simulation with seed."""
        circuit = Circuit(2)
        circuit.h(0)
        circuit.cnot(0, 1)

        sim1 = Simulator(backend="cpu", seed=42)
        result1 = sim1.run(circuit, shots=100)

        sim2 = Simulator(backend="cpu", seed=42)
        result2 = sim2.run(circuit, shots=100)

        # Same seed should give same results
        assert result1.get_counts() == result2.get_counts()


class TestMeasurement:
    """Test measurement operations."""

    def test_measurement_result(self):
        """Test measurement result object."""
        counts = {"00": 500, "11": 500}
        result = qratum.Result(counts, 2)

        assert result.shots == 1000
        assert len(result.get_counts()) == 2

        probs = result.get_probabilities()
        assert np.isclose(probs["00"], 0.5)
        assert np.isclose(probs["11"], 0.5)

    def test_most_frequent(self):
        """Test finding most frequent outcomes."""
        counts = {"00": 100, "01": 300, "10": 200, "11": 400}
        result = qratum.Result(counts, 2)

        most_frequent = result.most_frequent(2)
        assert most_frequent[0] == "11"
        assert most_frequent[1] == "01"


class TestBackwardCompatibility:
    """Test backward compatibility with QuASIM."""

    def test_quasim_import(self):
        """Test importing quasim package."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Should have deprecation warning
            assert len(w) >= 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "qratum" in str(w[0].message).lower()

    def test_quasim_quantum_circuit(self):
        """Test using quantum circuit through quasim."""
        import quasim

        if hasattr(quasim, "QuantumCircuit"):
            circuit = quasim.QuantumCircuit(2)
            circuit.h(0)
            assert circuit.gate_count() == 1


class TestQRATUMMetadata:
    """Test QRATUM metadata and version information."""

    def test_version(self):
        """Test version is 2.0.0."""
        assert qratum.__version__ == "2.0.0"

    def test_legacy_name(self):
        """Test legacy name is QuASIM."""
        assert qratum.__legacy_name__ == "QuASIM"

    def test_branding(self):
        """Test QRATUM branding elements."""
        assert "qratum.io" in qratum.__url__
        assert "QRATUM" in qratum.__github__
        assert qratum.__license__ == "Apache-2.0"


def test_full_workflow():
    """Test complete workflow from circuit to measurement."""
    # Create GHZ state
    circuit = Circuit(3)
    circuit.h(0)
    circuit.cnot(0, 1)
    circuit.cnot(1, 2)

    # Simulate
    sim = Simulator(backend="cpu", seed=42)
    result = sim.run(circuit, shots=1000)

    # Verify
    counts = result.get_counts()
    assert len(counts) == 2
    assert "000" in counts
    assert "111" in counts


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
