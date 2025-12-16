"""Macro domain mapping."""

from __future__ import annotations

from typing import Dict

from ..core.pricing import price_stream


def macro_index(indicators: Dict[str, float]) -> float:
    weights = {"gdp": 0.5, "inflation": -0.3, "employment": 0.2}
    return price_stream(indicators, weights)
