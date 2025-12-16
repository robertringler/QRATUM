"""Deterministic ID generator."""

from __future__ import annotations

import hashlib
from typing import Any


def deterministic_id(namespace: str, value: Any) -> str:
    material = f"{namespace}:{repr(value)}".encode()
    return hashlib.sha256(material).hexdigest()
