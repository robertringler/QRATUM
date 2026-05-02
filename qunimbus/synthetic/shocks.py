"""Deterministic shock injectors."""

from __future__ import annotations


def apply_price_shock(prices: dict[str, float], shock: dict[str, float]) -> dict[str, float]:
    updated = dict(prices)
    for symbol, delta in sorted(shock.items()):
        updated[symbol] = updated.get(symbol, 0.0) + delta
    return updated


def liquidity_shock(liquidity: float, reduction: float) -> float:
    return max(liquidity - reduction, 0.0)
