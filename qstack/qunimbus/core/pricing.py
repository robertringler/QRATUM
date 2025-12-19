"""Pricing primitives."""

from __future__ import annotations

from typing import Dict


def price_stream(prices: Dict[str, float], weights: Dict[str, float]) -> float:
    total = 0.0
    for key, price in prices.items():
        total += price * weights.get(key, 0.0)
    return total
