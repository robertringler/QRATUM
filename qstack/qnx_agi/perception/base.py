from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Percept:
    """Atomic observation units emitted by encoders."""

    modality: str
    value: Any
    features: dict[str, float] | None = None


class PerceptionLayer:
    def __init__(self):
        self._listeners: list[str] = []

    def register_listener(self, name: str) -> None:
        if name not in self._listeners:
            self._listeners.append(name)

    def process(self, raw: Any, modality: str = "generic") -> list[Percept]:
        """Convert raw payloads into deterministic percepts."""

        features = self._extract_features(raw)
        percept = Percept(modality=modality, value=raw, features=features)
        return [percept]

    def _extract_features(self, raw: Any) -> dict[str, float]:
        if isinstance(raw, dict):
            return {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))}
        return {"length": float(len(str(raw)))}

    def listeners(self) -> list[str]:
        return list(self._listeners)
