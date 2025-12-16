"""Abstraction helpers for world model state grouping."""

from __future__ import annotations

from typing import Dict

from .base import WorldState


def coarse_grain(state: WorldState, fields: Dict[str, str]) -> Dict[str, str]:
    return {alias: str(state.facts.get(name, "unknown")) for alias, name in fields.items()}
