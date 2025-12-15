"""Aerospace domain evaluator."""

from __future__ import annotations

from typing import Dict

from ..core.tensor_ops import tensor_contract


def evaluate_flight(profile: Dict[str, float]) -> float:
    matrix = [[profile.get("altitude", 0.0), profile.get("velocity", 0.0)]]
    return tensor_contract(matrix)
