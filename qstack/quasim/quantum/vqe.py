"""Stubbed deterministic VQE energy evaluation."""

from __future__ import annotations

from typing import List

from ..core.circuits import QuantumCircuit
from ..core.evaluators import evaluate_circuit


def vqe_energy(circuit: QuantumCircuit, params: List[float]) -> float:
    return evaluate_circuit(circuit, params)
