"""Static deterministic topology model."""

from __future__ import annotations

from typing import Dict, List, Tuple


def ring_topology(nodes: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for i, node in enumerate(sorted(nodes)):
        mapping[node] = sorted(nodes)[(i + 1) % len(nodes)]
    return mapping


def star_topology(center: str, leaves: List[str]) -> List[Tuple[str, str]]:
    return [(center, leaf) for leaf in sorted(leaves)]
