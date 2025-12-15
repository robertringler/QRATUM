"""Pharma domain evaluator."""

from __future__ import annotations

from typing import Dict

from ..core.tensor_ops import tensor_contract


def evaluate_trial(trial: Dict[str, float]) -> float:
    matrix = [[trial.get("dose", 0.0), trial.get("response", 0.0)]]
    return tensor_contract(matrix)
