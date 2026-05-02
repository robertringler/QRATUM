"""Deterministic market data adapter."""

from __future__ import annotations

from qreal.base_adapter import BaseAdapter
from qreal.normalizers import clamp_numbers, enforce_fields, sort_keys


class MarketAdapter(BaseAdapter):
    def __init__(self, source: str = "market") -> None:
        super().__init__(source=source)
        self.chain.steps.extend(
            [
                enforce_fields(["symbol", "open", "high", "low", "close", "volume"]),
                clamp_numbers(0.0, 10**9),
                sort_keys,
            ]
        )

    def _normalize(self, raw: object, tick: int) -> dict[str, object]:
        if not isinstance(raw, dict):
            raise TypeError("MarketAdapter expects dict input")
        normalized = {**raw, "tick": tick}
        return normalized

    def _to_percept(self, normalized: dict[str, object]) -> dict[str, object]:
        return {
            "kind": "market_bar",
            "symbol": normalized["symbol"],
            "price": normalized["close"],
            "volume": normalized["volume"],
            "tick": normalized["tick"],
        }
