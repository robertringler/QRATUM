"""Pharma domain evaluator."""
from __future__ import annotations


from ..core.tensor_ops import tensor_contract


def evaluate_trial(trial: dict[str, float]) -> float:
    matrix = [[trial.get("dose", 0.0), trial.get("response", 0.0)]]
    return tensor_contract(matrix)
