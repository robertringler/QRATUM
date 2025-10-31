"""Bandwidth-aware dataframe utilities inspired by RAPIDS."""
from __future__ import annotations

def columnar_sum(table: dict[str, list[float]]) -> dict[str, float]:
    return {name: float(sum(column)) for name, column in table.items()}
