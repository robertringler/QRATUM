"""Integrity rules for scientific datasets."""

from __future__ import annotations

from typing import Dict, List


def check_protocol(snapshot: Dict[str, object], required_fields: List[str]) -> List[str]:
    missing = [field for field in required_fields if field not in snapshot]
    return ["protocol_violation"] if missing else []
