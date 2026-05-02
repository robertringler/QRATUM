"""Adapter from QuASIM evaluators to world model updates."""

from __future__ import annotations

from typing import Any

from ..utils.serialization import deterministic_dumps


def translate_simulation_output(result: dict[str, Any]) -> dict[str, Any]:
    serialized = deterministic_dumps(result)
    return {"simulation_result": serialized, "score": len(serialized)}
