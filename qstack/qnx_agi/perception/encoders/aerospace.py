"""Telemetry encoder for aerospace domain."""

from __future__ import annotations

from typing import Dict, List

from ..base import Percept, PerceptionLayer


class AerospaceEncoder(PerceptionLayer):
    def process(self, raw: Dict[str, float], modality: str = "aerospace") -> List[Percept]:
        features = {
            "altitude": float(raw.get("altitude", 0.0)),
            "velocity": float(raw.get("velocity", 0.0)),
        }
        percept = Percept(modality=modality, value=raw, features=features)
        return [percept]
