"""Minimal, correct state-vector quantum simulator for QRATUM.

This module provides a small but *genuine* quantum-computing core built on exact
numpy linear algebra (no mocked results):

* :data:`gates` - standard quantum gate matrices (Pauli, Hadamard, phase,
  rotations, and two-qubit CNOT/CZ/SWAP).
* :class:`StateVector` - a dense state-vector container.
* :class:`Circuit` - a chainable quantum circuit builder.
* :class:`Simulator` - a CPU state-vector simulator with shot sampling and
  seed-deterministic measurement.
* :class:`Result` - a measurement-counts container.

Gates are applied via tensor contraction over the relevant qubit axes, which is
exact for any qubit indices (not limited to adjacent qubits).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class _Gates:
    """Namespace of standard quantum gate matrices (complex numpy arrays)."""

    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    S = np.array([[1, 0], [0, 1j]], dtype=complex)
    T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

    CNOT = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
    )
    CZ = np.diag([1, 1, 1, -1]).astype(complex)
    SWAP = np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
    )

    @staticmethod
    def RX(theta: float) -> np.ndarray:
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)

    @staticmethod
    def RY(theta: float) -> np.ndarray:
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)

    @staticmethod
    def RZ(theta: float) -> np.ndarray:
        return np.array(
            [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex
        )


gates = _Gates()


class StateVector:
    """Dense quantum state vector over ``num_qubits`` qubits."""

    def __init__(self, data) -> None:
        self.data = np.asarray(data, dtype=complex).reshape(-1)
        if self.data.size == 0 or (self.data.size & (self.data.size - 1)) != 0:
            raise ValueError("State vector length must be a positive power of two")
        self.num_qubits = int(round(np.log2(self.data.size)))

    @classmethod
    def zero_state(cls, num_qubits: int) -> "StateVector":
        """Return the |0...0> computational basis state."""
        data = np.zeros(2**num_qubits, dtype=complex)
        data[0] = 1.0
        return cls(data)

    def normalize(self) -> "StateVector":
        """Normalize the state in place and return self."""
        norm = np.linalg.norm(self.data)
        if norm > 0:
            self.data = self.data / norm
        return self

    def probabilities(self) -> np.ndarray:
        """Measurement probabilities for each computational basis state."""
        return np.abs(self.data) ** 2

    def inner_product(self, other: "StateVector") -> complex:
        """Return <self|other>."""
        return complex(np.vdot(self.data, other.data))


def _apply_gate(
    state: np.ndarray, matrix: np.ndarray, qubits: tuple, num_qubits: int
) -> np.ndarray:
    """Apply ``matrix`` acting on ``qubits`` to a rank-``num_qubits`` state tensor."""
    k = len(qubits)
    gate_tensor = matrix.reshape([2] * (2 * k))
    # Contract the gate's input axes (last k) with the state's qubit axes.
    state = np.tensordot(gate_tensor, state, axes=(list(range(k, 2 * k)), list(qubits)))
    # tensordot puts the k output axes first; move them back to their qubit slots.
    return np.moveaxis(state, list(range(k)), list(qubits))


class Circuit:
    """Chainable quantum circuit builder."""

    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits
        # Each operation is (name, matrix, qubit-tuple).
        self.operations: list[tuple[str, np.ndarray, tuple]] = []

    def gate_count(self) -> int:
        return len(self.operations)

    def _add(self, name: str, matrix: np.ndarray, qubits: tuple) -> "Circuit":
        for q in qubits:
            if not 0 <= q < self.num_qubits:
                raise ValueError(f"Qubit {q} out of range for {self.num_qubits}-qubit circuit")
        self.operations.append((name, matrix, tuple(qubits)))
        return self

    # Single-qubit gates
    def h(self, q: int) -> "Circuit":
        return self._add("h", gates.H, (q,))

    def x(self, q: int) -> "Circuit":
        return self._add("x", gates.X, (q,))

    def y(self, q: int) -> "Circuit":
        return self._add("y", gates.Y, (q,))

    def z(self, q: int) -> "Circuit":
        return self._add("z", gates.Z, (q,))

    def s(self, q: int) -> "Circuit":
        return self._add("s", gates.S, (q,))

    def t(self, q: int) -> "Circuit":
        return self._add("t", gates.T, (q,))

    def rx(self, q: int, theta: float) -> "Circuit":
        return self._add("rx", gates.RX(theta), (q,))

    def ry(self, q: int, theta: float) -> "Circuit":
        return self._add("ry", gates.RY(theta), (q,))

    def rz(self, q: int, theta: float) -> "Circuit":
        return self._add("rz", gates.RZ(theta), (q,))

    # Two-qubit gates
    def cnot(self, control: int, target: int) -> "Circuit":
        return self._add("cnot", gates.CNOT, (control, target))

    cx = cnot

    def cz(self, control: int, target: int) -> "Circuit":
        return self._add("cz", gates.CZ, (control, target))

    def swap(self, a: int, b: int) -> "Circuit":
        return self._add("swap", gates.SWAP, (a, b))

    def depth(self) -> int:
        """Circuit depth, treating gates on disjoint qubits as parallel."""
        times = [0] * self.num_qubits
        for _name, _matrix, qubits in self.operations:
            layer = max(times[q] for q in qubits) + 1
            for q in qubits:
                times[q] = layer
        return max(times) if times else 0

    def copy(self) -> "Circuit":
        clone = Circuit(self.num_qubits)
        clone.operations = list(self.operations)
        return clone


class Result:
    """Container for measurement counts from a simulation run."""

    def __init__(self, counts: dict, num_qubits: int) -> None:
        self.counts = dict(counts)
        self.num_qubits = num_qubits
        self.shots = sum(self.counts.values())

    def get_counts(self) -> dict:
        return dict(self.counts)

    def get_probabilities(self) -> dict:
        if self.shots == 0:
            return {}
        return {k: v / self.shots for k, v in self.counts.items()}

    def most_frequent(self, n: int = 1) -> list:
        ordered = sorted(self.counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return [bitstring for bitstring, _ in ordered[:n]]


class Simulator:
    """CPU state-vector simulator with seed-deterministic measurement."""

    def __init__(
        self,
        backend: str = "cpu",
        precision: str = "fp64",
        seed: Optional[int] = None,
    ) -> None:
        self.backend = backend
        self.precision = precision
        self.seed = seed

    def _evolve(self, circuit: Circuit) -> np.ndarray:
        state = np.zeros([2] * circuit.num_qubits, dtype=complex)
        state.flat[0] = 1.0
        for _name, matrix, qubits in circuit.operations:
            state = _apply_gate(state, matrix, qubits, circuit.num_qubits)
        return state.reshape(-1)

    def run_statevector(self, circuit: Circuit) -> StateVector:
        """Return the final state vector (no measurement)."""
        return StateVector(self._evolve(circuit))

    def run(self, circuit: Circuit, shots: int = 1024) -> Result:
        """Sample ``shots`` measurements of ``circuit`` in the computational basis."""
        amplitudes = self._evolve(circuit)
        probs = np.abs(amplitudes) ** 2
        total = probs.sum()
        if total > 0:
            probs = probs / total
        rng = np.random.default_rng(self.seed)
        n = circuit.num_qubits
        outcomes = rng.choice(len(probs), size=shots, p=probs)
        counts: dict[str, int] = {}
        for outcome in outcomes:
            bitstring = format(int(outcome), f"0{n}b")
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return Result(counts, n)


__all__ = ["gates", "StateVector", "Circuit", "Simulator", "Result"]
