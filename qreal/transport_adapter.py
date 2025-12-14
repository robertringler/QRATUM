"""Deterministic adapter for aviation/rail/road feeds."""

from __future__ import annotations

from typing import Dict

from qreal.base_adapter import BaseAdapter
from qreal.normalizers import clamp_numbers, enforce_fields, sort_keys


class TransportAdapter(BaseAdapter):
    def __init__(self, source: str = "transport") -> None:
        super().__init__(source=source)
        self.chain.steps.extend(
            [
                enforce_fields(["vehicle_id", "mode", "position", "speed", "status"]),
                clamp_numbers(0.0, 10**5),
                sort_keys,
            ]
        )

    def _normalize(self, raw: object, tick: int) -> Dict[str, object]:
        if not isinstance(raw, dict):
            raise TypeError("TransportAdapter expects dict input")
        return {**raw, "tick": tick}

    def _to_percept(self, normalized: Dict[str, object]) -> Dict[str, object]:
        return {
            "kind": "transport_state",
            "vehicle_id": normalized["vehicle_id"],
            "mode": normalized["mode"],
            "position": normalized["position"],
            "speed": normalized["speed"],
            "status": normalized["status"],
            "tick": normalized["tick"],
        }
