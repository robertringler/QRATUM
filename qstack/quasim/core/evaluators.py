"""Evaluators bridging circuits and tensor ops."""
from __future__ import annotations


from .tensor_ops import tensor_contract
from .circuits import QuantumCircuit


def evaluate_circuit(circuit: QuantumCircuit, inputs: list[float]) -> float:
    accum = 0.0
    for value in inputs:
        accum += circuit.evaluate(value)
    return accum / len(inputs) if inputs else 0.0


def evaluate_tensor(tensor: list[list[float]]) -> float:
    return tensor_contract(tensor)
