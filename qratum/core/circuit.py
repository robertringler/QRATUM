"""QRATUM quantum circuit builder with fluent API.

Provides a flexible circuit builder for constructing quantum circuits
with support for all standard gates and custom operations.
"""

import numpy as np
from typing import List, Optional, Tuple, Union
from qratum.core import gates


class Circuit:
    """Quantum circuit builder with fluent API.

    Attributes:
        num_qubits: Number of qubits in the circuit
        instructions: List of gate instructions
    """

    def __init__(self, num_qubits: int):
        """Initialize quantum circuit.

        Args:
            num_qubits: Number of qubits
        """
        self.num_qubits = num_qubits
        self.instructions: List[Tuple[str, List[int], Optional[np.ndarray], dict]] = []

    # ========================================================================
    # Single-Qubit Gates
    # ========================================================================

    def h(self, qubit: int) -> "Circuit":
        """Apply Hadamard gate.

        Args:
            qubit: Target qubit index

        Returns:
            Self (for chaining)
        """
        self.instructions.append(("h", [qubit], gates.H, {}))
        return self

    def x(self, qubit: int) -> "Circuit":
        """Apply Pauli-X (NOT) gate.

        Args:
            qubit: Target qubit index

        Returns:
            Self (for chaining)
        """
        self.instructions.append(("x", [qubit], gates.X, {}))
        return self

    def y(self, qubit: int) -> "Circuit":
        """Apply Pauli-Y gate.

        Args:
            qubit: Target qubit index

        Returns:
            Self (for chaining)
        """
        self.instructions.append(("y", [qubit], gates.Y, {}))
        return self

    def z(self, qubit: int) -> "Circuit":
        """Apply Pauli-Z gate.

        Args:
            qubit: Target qubit index

        Returns:
            Self (for chaining)
        """
        self.instructions.append(("z", [qubit], gates.Z, {}))
        return self

    def s(self, qubit: int) -> "Circuit":
        """Apply S (phase) gate.

        Args:
            qubit: Target qubit index

        Returns:
            Self (for chaining)
        """
        self.instructions.append(("s", [qubit], gates.S, {}))
        return self

    def t(self, qubit: int) -> "Circuit":
        """Apply T gate.

        Args:
            qubit: Target qubit index

        Returns:
            Self (for chaining)
        """
        self.instructions.append(("t", [qubit], gates.T, {}))
        return self

    def rx(self, qubit: int, theta: float) -> "Circuit":
        """Apply RX rotation gate.

        Args:
            qubit: Target qubit index
            theta: Rotation angle in radians

        Returns:
            Self (for chaining)
        """
        self.instructions.append(("rx", [qubit], gates.RX(theta), {"theta": theta}))
        return self

    def ry(self, qubit: int, theta: float) -> "Circuit":
        """Apply RY rotation gate.

        Args:
            qubit: Target qubit index
            theta: Rotation angle in radians

        Returns:
            Self (for chaining)
        """
        self.instructions.append(("ry", [qubit], gates.RY(theta), {"theta": theta}))
        return self

    def rz(self, qubit: int, theta: float) -> "Circuit":
        """Apply RZ rotation gate.

        Args:
            qubit: Target qubit index
            theta: Rotation angle in radians

        Returns:
            Self (for chaining)
        """
        self.instructions.append(("rz", [qubit], gates.RZ(theta), {"theta": theta}))
        return self

    # ========================================================================
    # Two-Qubit Gates
    # ========================================================================

    def cnot(self, control: int, target: int) -> "Circuit":
        """Apply CNOT (controlled-NOT) gate.

        Args:
            control: Control qubit index
            target: Target qubit index

        Returns:
            Self (for chaining)
        """
        self.instructions.append(("cnot", [control, target], gates.CNOT, {}))
        return self

    def cx(self, control: int, target: int) -> "Circuit":
        """Apply CX (controlled-X) gate (alias for CNOT).

        Args:
            control: Control qubit index
            target: Target qubit index

        Returns:
            Self (for chaining)
        """
        return self.cnot(control, target)

    def cz(self, control: int, target: int) -> "Circuit":
        """Apply CZ (controlled-Z) gate.

        Args:
            control: Control qubit index
            target: Target qubit index

        Returns:
            Self (for chaining)
        """
        self.instructions.append(("cz", [control, target], gates.CZ, {}))
        return self

    def swap(self, qubit1: int, qubit2: int) -> "Circuit":
        """Apply SWAP gate.

        Args:
            qubit1: First qubit index
            qubit2: Second qubit index

        Returns:
            Self (for chaining)
        """
        self.instructions.append(("swap", [qubit1, qubit2], gates.SWAP, {}))
        return self

    # ========================================================================
    # Multi-Qubit Gates
    # ========================================================================

    def toffoli(self, control1: int, control2: int, target: int) -> "Circuit":
        """Apply Toffoli (CCNOT) gate.

        Args:
            control1: First control qubit index
            control2: Second control qubit index
            target: Target qubit index

        Returns:
            Self (for chaining)
        """
        self.instructions.append(
            ("toffoli", [control1, control2, target], gates.TOFFOLI, {})
        )
        return self

    # ========================================================================
    # Circuit Analysis
    # ========================================================================

    def depth(self) -> int:
        """Calculate circuit depth.

        Returns:
            Circuit depth (number of gate layers)
        """
        if not self.instructions:
            return 0

        # Track when each qubit is last used
        qubit_layers = [0] * self.num_qubits
        current_depth = 0

        for gate_name, qubits, _, _ in self.instructions:
            # Find maximum layer of involved qubits
            max_layer = max(qubit_layers[q] for q in qubits)
            # Update all involved qubits to next layer
            for q in qubits:
                qubit_layers[q] = max_layer + 1
            current_depth = max(current_depth, max_layer + 1)

        return current_depth

    def gate_count(self) -> int:
        """Count total number of gates.

        Returns:
            Total number of gates in circuit
        """
        return len(self.instructions)

    def count_ops(self) -> dict:
        """Count gates by type.

        Returns:
            Dictionary mapping gate names to counts
        """
        counts = {}
        for gate_name, _, _, _ in self.instructions:
            counts[gate_name] = counts.get(gate_name, 0) + 1
        return counts

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def inverse(self) -> "Circuit":
        """Create inverse of this circuit.

        Returns:
            New circuit that is the inverse
        """
        inv_circuit = Circuit(self.num_qubits)
        # Reverse instructions and conjugate gates
        for gate_name, qubits, matrix, params in reversed(self.instructions):
            if matrix is not None:
                inv_matrix = np.conj(matrix.T)
                inv_circuit.instructions.append((gate_name, qubits, inv_matrix, params))
        return inv_circuit

    def copy(self) -> "Circuit":
        """Create a copy of this circuit.

        Returns:
            New circuit with copied instructions
        """
        new_circuit = Circuit(self.num_qubits)
        new_circuit.instructions = self.instructions.copy()
        return new_circuit

    def __repr__(self) -> str:
        """String representation of circuit."""
        return f"Circuit({self.num_qubits} qubits, {len(self.instructions)} gates)"

    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [f"Circuit with {self.num_qubits} qubits:"]
        for i, (gate_name, qubits, _, params) in enumerate(self.instructions):
            qubit_str = ",".join(map(str, qubits))
            param_str = ""
            if params:
                param_str = f" {params}"
            lines.append(f"  {i}: {gate_name}({qubit_str}){param_str}")
        lines.append(f"Depth: {self.depth()}, Gates: {self.gate_count()}")
        return "\n".join(lines)


__all__ = ["Circuit"]
