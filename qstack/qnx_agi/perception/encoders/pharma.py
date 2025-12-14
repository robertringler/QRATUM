"""Lab data encoder for pharma domain."""
from __future__ import annotations

from ..base import Percept, PerceptionLayer


class PharmaEncoder(PerceptionLayer):
    def process(self, raw: dict[str, float], modality: str = "pharma") -> list[Percept]:
        dose = float(raw.get("dose", 0.0))
        response = float(raw.get("response", 0.0))
        percept = Percept(modality=modality, value=raw, features={"dose": dose, "response": response})
        return [percept]
