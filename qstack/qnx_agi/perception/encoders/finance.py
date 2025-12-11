"""Market data encoder for finance domain."""
from __future__ import annotations

from ..base import Percept, PerceptionLayer


class FinanceEncoder(PerceptionLayer):
    def process(self, raw: dict[str, float], modality: str = "finance") -> list[Percept]:
        price = float(raw.get("price", 0.0))
        volume = float(raw.get("volume", 0.0))
        percept = Percept(modality=modality, value=raw, features={"price": price, "volume": volume})
        return [percept]
