"""Deterministic energy/grid adapter."""

from __future__ import annotations

from typing import Dict

from qreal.base_adapter import BaseAdapter
from qreal.normalizers import clamp_numbers, enforce_fields, sort_keys


class GridAdapter(BaseAdapter):
    def __init__(self, source: str = "grid") -> None:
        super().__init__(source=source)
        self.chain.steps.extend(
            [
                enforce_fields(["region", "load", "generation", "frequency"]),
                clamp_numbers(-(10**3), 10**6),
                sort_keys,
            ]
        )

    def _normalize(self, raw: object, tick: int) -> Dict[str, object]:
        if not isinstance(raw, dict):
            raise TypeError("GridAdapter expects dict input")
        return {**raw, "tick": tick}

    def _to_percept(self, normalized: Dict[str, object]) -> Dict[str, object]:
        return {
            "kind": "grid_state",
            "region": normalized["region"],
            "load": normalized["load"],
            "generation": normalized["generation"],
            "frequency": normalized["frequency"],
            "tick": normalized["tick"],
        }
