"""Integrity rules for scientific datasets."""
from __future__ import annotations


def check_protocol(snapshot: dict[str, object], required_fields: list[str]) -> list[str]:
    missing = [field for field in required_fields if field not in snapshot]
    return ["protocol_violation"] if missing else []
