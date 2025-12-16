"""Deterministic adapter for scientific datasets."""

from __future__ import annotations

from typing import Dict

from qreal.base_adapter import BaseAdapter
from qreal.normalizers import enforce_fields, sort_keys


class ScienceAdapter(BaseAdapter):
    def __init__(self, source: str = "science") -> None:
        super().__init__(source=source)
        self.chain.steps.extend(
            [
                enforce_fields(["experiment", "measurement", "value", "unit"]),
                sort_keys,
            ]
        )

    def _normalize(self, raw: object, tick: int) -> Dict[str, object]:
        if not isinstance(raw, dict):
            raise TypeError("ScienceAdapter expects dict input")
        return {**raw, "tick": tick}

    def _to_percept(self, normalized: Dict[str, object]) -> Dict[str, object]:
        return {
            "kind": "scientific_measurement",
            "experiment": normalized["experiment"],
            "measurement": normalized["measurement"],
            "value": normalized["value"],
            "unit": normalized["unit"],
            "tick": normalized["tick"],
        }
