"""Quantum circuit representation and manipulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QuantumCircuit:
    """Represents a quantum circuit with qubits and gates.

    This class provides the foundation for quantum circuit construction
    and manipulation in QuASIM. It supports both CUDA and HIP backends
    for GPU-accelerated simulation.

    Attributes:
        num_qubits: Number of qubits in the circuit
        gates: List of quantum gates applied to the circuit
        backend: Computation backend ('cuda', 'hip', or 'cpu')
    """

    num_qubits: int
    gates: list[dict[str, Any]] = field(default_factory=list)
    backend: str = "cpu"

    def __post_init__(self) -> None:
        """Validate circuit parameters."""

        if self.num_qubits < 1:
            raise ValueError("Circuit must have at least 1 qubit")
        if self.backend not in ("cuda", "hip", "cpu"):
            raise ValueError(f"Unsupported backend: {self.backend}")

    def add_gate(
        self, gate_type: str, qubits: list[int], params: dict[str, Any] | None = None
    ) -> None:
        """Add a quantum gate to the circuit.

        Args:
            gate_type: Type of gate (e.g., 'H', 'CNOT', 'RZ')
            qubits: List of qubit indices the gate acts on
            params: Optional gate parameters (e.g., rotation angles)
        """

        if any(q < 0 or q >= self.num_qubits for q in qubits):
            raise ValueError(f"Qubit indices must be in range [0, {self.num_qubits})")

        gate = {
            "type": gate_type,
            "qubits": qubits,
            "params": params or {},
        }
        self.gates.append(gate)

    def depth(self) -> int:
        """Calculate the depth of the circuit (critical path length)."""

        # Simplified depth calculation - in production, would analyze gate dependencies
        return len(self.gates)

    def to_dict(self) -> dict[str, Any]:
        """Serialize circuit to dictionary for distributed execution."""

        return {
            "num_qubits": self.num_qubits,
            "gates": self.gates,
            "backend": self.backend,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuantumCircuit:
        """Deserialize circuit from dictionary."""

        circuit = cls(
            num_qubits=data["num_qubits"],
            backend=data.get("backend", "cpu"),
        )
        circuit.gates = data.get("gates", [])
        return circuit
