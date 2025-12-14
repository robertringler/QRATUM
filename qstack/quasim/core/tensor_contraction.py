"""Deterministic tensor contraction utilities."""

from __future__ import annotations

from typing import Sequence


def contract_tensors(tensors: list[Sequence[float]]) -> list[float]:
    """Performs deterministic pairwise tensor contraction (element-wise multiply and sum)."""
    if not tensors:
        return []
    result = list(tensors[0])
    for tensor in tensors[1:]:
        if len(tensor) != len(result):
            raise ValueError("tensor shapes must match for contraction")
        result = [a * b for a, b in zip(result, tensor)]
    total = sum(result)
    return [total]
