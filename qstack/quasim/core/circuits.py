"""Minimal circuit abstraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List


@dataclass
class QuantumGate:
    name: str
    operation: Callable[[float], float]

    def apply(self, amplitude: float) -> float:
        return self.operation(amplitude)


@dataclass
class QuantumCircuit:
    gates: List[QuantumGate] = field(default_factory=list)

    def add_gate(self, gate: QuantumGate) -> None:
        self.gates.append(gate)

    def evaluate(self, amplitude: float) -> float:
        result = amplitude
        for gate in self.gates:
            result = gate.apply(result)
        return result
