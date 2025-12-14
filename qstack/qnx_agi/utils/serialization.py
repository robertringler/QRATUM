"""Deterministic serialization helpers."""

from __future__ import annotations

import json
from typing import Any


def deterministic_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))
