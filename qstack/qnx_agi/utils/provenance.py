"""Provenance hashing utilities."""

from __future__ import annotations

import hashlib
from typing import Any, Dict


def hash_payload(payload: Dict[str, Any]) -> str:
    material = repr(sorted(payload.items())).encode("utf-8")
    return hashlib.sha256(material).hexdigest()
