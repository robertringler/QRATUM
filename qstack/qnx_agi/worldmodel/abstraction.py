"""Abstraction helpers for world model state grouping."""
from __future__ import annotations

from .base import WorldState


def coarse_grain(state: WorldState, fields: dict[str, str]) -> dict[str, str]:
    return {alias: str(state.facts.get(name, "unknown")) for alias, name in fields.items()}
