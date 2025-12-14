"""Deterministic uncertainty scoring."""
from __future__ import annotations


from ..utils.provenance import hash_payload


def score_variance(observation: dict[str, float]) -> float:
    if not observation:
        return 0.0
    values = list(observation.values())
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def annotate_with_uncertainty(payload: dict[str, float]) -> dict[str, float]:
    variance = score_variance(payload)
    payload = dict(payload)
    payload["uncertainty"] = variance
    payload["checksum"] = int(hash_payload(payload), 16) % 1000
    return payload
