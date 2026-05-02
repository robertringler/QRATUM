"""Aerospace domain evaluator."""

from __future__ import annotations

from ..core.tensor_ops import tensor_contract


def evaluate_flight(profile: dict[str, float]) -> float:
    matrix = [[profile.get("altitude", 0.0), profile.get("velocity", 0.0)]]
    return tensor_contract(matrix)
