"""Stubbed deterministic QAOA cost."""

from __future__ import annotations

from typing import List

from ..core.circuits import QuantumCircuit
from ..core.evaluators import evaluate_circuit


def qaoa_cost(circuit: QuantumCircuit, angles: List[float]) -> float:
    return evaluate_circuit(circuit, angles)
