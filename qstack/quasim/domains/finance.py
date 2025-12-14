"""Finance domain evaluator."""
from __future__ import annotations


from ..core.tensor_ops import tensor_contract


def evaluate_portfolio(portfolio: dict[str, float]) -> float:
    matrix = [[portfolio.get("price", 0.0), portfolio.get("volume", 0.0)]]
    return tensor_contract(matrix)
