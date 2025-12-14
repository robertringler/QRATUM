"""Unified telemetry stream for Q-Stack subsystems."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


def _ensure_serializable(data: Mapping[str, Any]) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            serialized[key] = value
        elif isinstance(value, (list, tuple)):
            serialized[key] = [str(item) if not isinstance(item, (str, int, float, bool, type(None))) else item for item in value]
        elif isinstance(value, dict):
            serialized[key] = {str(sub_key): str(sub_val) for sub_key, sub_val in value.items()}
        else:
            serialized[key] = str(value)
    return serialized


@dataclass
class Telemetry:
    """Append-only telemetry collector."""

    entries: list[dict[str, Any]] = field(default_factory=list)

    def record(self, component: str, metrics: Mapping[str, Any], context: Mapping[str, Any] | None = None) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "component": component,
            "metrics": _ensure_serializable(dict(metrics)),
            "context": _ensure_serializable(dict(context)) if context else {},
        }
        self.entries.append(entry)
        return entry

    def as_dict(self) -> dict[str, Any]:
        return {"entries": list(self.entries)}

    def __repr__(self) -> str:  # pragma: no cover - deterministic formatting
        return f"Telemetry(entries={len(self.entries)})"
