"""Evaluators bridging circuits and tensor ops."""

from __future__ import annotations

from typing import List

from .circuits import QuantumCircuit
from .tensor_ops import tensor_contract


def evaluate_circuit(circuit: QuantumCircuit, inputs: List[float]) -> float:
    accum = 0.0
    for value in inputs:
        accum += circuit.evaluate(value)
    return accum / len(inputs) if inputs else 0.0


def evaluate_tensor(tensor: List[List[float]]) -> float:
    return tensor_contract(tensor)
