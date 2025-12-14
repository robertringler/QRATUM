"""Decode a subset of Common Information Model (CIM) data."""
from __future__ import annotations



def decode(record: dict[str, object]) -> dict[str, object]:
    expected = {"asset", "rating", "status"}
    if not expected.issubset(record.keys()):
        missing = sorted(list(expected - set(record.keys())))
        raise ValueError(f"Missing CIM fields: {missing}")
    return {key: record[key] for key in sorted(record.keys())}
