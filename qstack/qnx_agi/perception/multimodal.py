"""Multimodal perception fusion."""

from __future__ import annotations

from typing import Dict, List

from .base import Percept


def fuse(percepts: List[Percept]) -> Percept:
    """Deterministically fuse percepts by averaging numeric features."""
    if not percepts:
        return Percept(modality="fused", value={}, features={})
    fused_features: Dict[str, float] = {}
    for percept in percepts:
        if not percept.features:
            continue
        for key, value in percept.features.items():
            fused_features[key] = fused_features.get(key, 0.0) + value
    count = max(len(percepts), 1)
    averaged = {k: v / count for k, v in fused_features.items()}
    return Percept(modality="fused", value=[p.value for p in percepts], features=averaged)
