"""Canonical serialization for snapshots."""
from __future__ import annotations

import json
from typing import Any


def canonical_serialize(obj: dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))
