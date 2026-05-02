"""Telemetry encoder for aerospace domain."""

from __future__ import annotations

from ..base import Percept, PerceptionLayer


class AerospaceEncoder(PerceptionLayer):
    def process(self, raw: dict[str, float], modality: str = "aerospace") -> list[Percept]:
        features = {
            "altitude": float(raw.get("altitude", 0.0)),
            "velocity": float(raw.get("velocity", 0.0)),
        }
        percept = Percept(modality=modality, value=raw, features=features)
        return [percept]
