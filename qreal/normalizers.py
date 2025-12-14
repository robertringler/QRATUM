"""Deterministic normalization transforms for external feeds."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class NormalizationChain:
    """Composable set of deterministic normalization steps."""

    steps: list[Callable[[dict[str, object]], dict[str, object]]] = field(default_factory=list)

    def apply(self, payload: dict[str, object]) -> dict[str, object]:
        current = dict(payload)
        for step in self.steps:
            current = step(current)
        return current

    def describe(self) -> list[str]:
        return [getattr(step, "__name__", "anonymous") for step in self.steps]


# Built-in normalizers ----------------------------------------------------------------

def sort_keys(payload: dict[str, object]) -> dict[str, object]:
    return {k: payload[k] for k in sorted(payload.keys())}


def clamp_numbers(min_value: float, max_value: float) -> Callable[[dict[str, object]], dict[str, object]]:
    def _clamp(payload: dict[str, object]) -> dict[str, object]:
        new_payload: dict[str, object] = {}
        for key, value in payload.items():
            if isinstance(value, (int, float)):
                new_payload[key] = max(min_value, min(max_value, float(value)))
            else:
                new_payload[key] = value
        return new_payload

    _clamp.__name__ = f"clamp_{min_value}_{max_value}"
    return _clamp


def rename(mapping: dict[str, str]) -> Callable[[dict[str, object]], dict[str, object]]:
    def _rename(payload: dict[str, object]) -> dict[str, object]:
        return {mapping.get(k, k): v for k, v in payload.items()}

    _rename.__name__ = "rename_fields"
    return _rename


def enforce_fields(required: list[str]) -> Callable[[dict[str, object]], dict[str, object]]:
    def _enforce(payload: dict[str, object]) -> dict[str, object]:
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        return payload

    _enforce.__name__ = "enforce_fields"
    return _enforce
