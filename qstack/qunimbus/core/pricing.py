"""Pricing primitives."""
from __future__ import annotations


def price_stream(prices: dict[str, float], weights: dict[str, float]) -> float:
    total = 0.0
    for key, price in prices.items():
        total += price * weights.get(key, 0.0)
    return total
