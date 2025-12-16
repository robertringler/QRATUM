"""Deterministic serialization helpers."""

from __future__ import annotations

import json
from typing import Any, Dict


def deterministic_dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))
