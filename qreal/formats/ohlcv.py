"""Decode OHLCV rows into deterministic dictionaries."""
from __future__ import annotations

from typing import Sequence

FIELDS = ["open", "high", "low", "close", "volume"]


def decode(symbol: str, row: Sequence[float]) -> dict[str, object]:
    if len(row) != len(FIELDS):
        raise ValueError("Expected OHLCV row of length 5")
    return {"symbol": symbol, **{field: float(value) for field, value in zip(FIELDS, row)}}
