"""Stubbed deterministic QAOA cost."""

from __future__ import annotations

from ..core.circuits import QuantumCircuit
from ..core.evaluators import evaluate_circuit


def qaoa_cost(circuit: QuantumCircuit, angles: list[float]) -> float:
    return evaluate_circuit(circuit, angles)
