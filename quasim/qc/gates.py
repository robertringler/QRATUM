"""Quantum gate definitions and operations."""

from __future__ import annotations

import math
from typing import Any


class GateSet:
    """Standard quantum gate set for QuASIM.

    Provides common quantum gates including Pauli gates, Hadamard,
    controlled gates, and rotation gates. Gates are defined in a
    backend-agnostic manner and compiled to CUDA/HIP kernels at runtime.
    """

    # Single-qubit gates
    PAULI_X = "X"
    PAULI_Y = "Y"
    PAULI_Z = "Z"
    HADAMARD = "H"
    PHASE = "S"
    T_GATE = "T"

    # Two-qubit gates
    CNOT = "CNOT"
    CZ = "CZ"
    SWAP = "SWAP"

    # Rotation gates
    RX = "RX"
    RY = "RY"
    RZ = "RZ"

    # Three-qubit gates
    TOFFOLI = "TOFFOLI"
    FREDKIN = "FREDKIN"

    @classmethod
    def get_gate_matrix(
        cls, gate_type: str, params: dict[str, Any] | None = None
    ) -> list[list[complex]]:
        """Get the unitary matrix representation of a gate.

        Args:
            gate_type: Type of quantum gate
            params: Gate parameters (e.g., rotation angle for RZ)

        Returns:
            2D list representing the unitary matrix
        """
        params = params or {}

        # Single-qubit Pauli gates
        if gate_type == cls.PAULI_X:
            return [[0, 1], [1, 0]]
        elif gate_type == cls.PAULI_Y:
            return [[0, -1j], [1j, 0]]
        elif gate_type == cls.PAULI_Z:
            return [[1, 0], [0, -1]]

        # Hadamard gate
        elif gate_type == cls.HADAMARD:
            sqrt2 = math.sqrt(2)
            return [[1 / sqrt2, 1 / sqrt2], [1 / sqrt2, -1 / sqrt2]]

        # Phase gate
        elif gate_type == cls.PHASE:
            return [[1, 0], [0, 1j]]

        # T gate
        elif gate_type == cls.T_GATE:
            return [[1, 0], [0, complex(math.cos(math.pi / 4), math.sin(math.pi / 4))]]

        # Rotation gates
        elif gate_type == cls.RZ and "theta" in params:
            theta = params["theta"]
            return [
                [complex(math.cos(-theta / 2), math.sin(-theta / 2)), 0],
                [0, complex(math.cos(theta / 2), math.sin(theta / 2))],
            ]

        # Default identity for unknown gates
        return [[1, 0], [0, 1]]

    @classmethod
    def validate_gate(cls, gate_type: str, num_qubits: int) -> bool:
        """Validate that a gate type is supported and has correct qubit count.

        Args:
            gate_type: Type of quantum gate
            num_qubits: Number of qubits the gate acts on

        Returns:
            True if gate is valid
        """
        single_qubit_gates = {
            cls.PAULI_X,
            cls.PAULI_Y,
            cls.PAULI_Z,
            cls.HADAMARD,
            cls.PHASE,
            cls.T_GATE,
            cls.RX,
            cls.RY,
            cls.RZ,
        }
        two_qubit_gates = {cls.CNOT, cls.CZ, cls.SWAP}
        three_qubit_gates = {cls.TOFFOLI, cls.FREDKIN}

        if gate_type in single_qubit_gates and num_qubits == 1:
            return True
        if gate_type in two_qubit_gates and num_qubits == 2:
            return True
        return bool(gate_type in three_qubit_gates and num_qubits == 3)
