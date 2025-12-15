"""Deterministic telemetry adapter for spacecraft/vehicle feeds."""

from __future__ import annotations

from typing import Dict

from qreal.base_adapter import BaseAdapter
from qreal.normalizers import clamp_numbers, enforce_fields, sort_keys


class TelemetryAdapter(BaseAdapter):
    def __init__(self, source: str = "telemetry") -> None:
        super().__init__(source=source)
        self.chain.steps.extend(
            [
                enforce_fields(["vehicle", "position", "velocity", "status"]),
                clamp_numbers(-(10**6), 10**6),
                sort_keys,
            ]
        )

    def _normalize(self, raw: object, tick: int) -> Dict[str, object]:
        if not isinstance(raw, dict):
            raise TypeError("TelemetryAdapter expects dict input")
        normalized = {**raw, "tick": tick}
        return normalized

    def _to_percept(self, normalized: Dict[str, object]) -> Dict[str, object]:
        return {
            "kind": "telemetry_state",
            "vehicle": normalized["vehicle"],
            "position": normalized["position"],
            "velocity": normalized["velocity"],
            "status": normalized["status"],
            "tick": normalized["tick"],
        }
