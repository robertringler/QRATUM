"""Stubbed deterministic VQE energy evaluation."""
from __future__ import annotations


from ..core.circuits import QuantumCircuit
from ..core.evaluators import evaluate_circuit


def vqe_energy(circuit: QuantumCircuit, params: list[float]) -> float:
    return evaluate_circuit(circuit, params)
