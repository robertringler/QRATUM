"""Deterministic state delta computation."""
from __future__ import annotations

from typing import Any

from .state import QNXState


def compute_delta(prev: QNXState, current: QNXState) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for key, value in current.data.items():
        if prev.read(key) != value:
            delta[key] = {"old": prev.read(key), "new": value}
    for key in prev.data.keys():
        if key not in current.data:
            delta[key] = {"old": prev.read(key), "new": None}
    return delta
