"""Static deterministic topology model."""
from __future__ import annotations


def ring_topology(nodes: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for i, node in enumerate(sorted(nodes)):
        mapping[node] = sorted(nodes)[(i + 1) % len(nodes)]
    return mapping


def star_topology(center: str, leaves: list[str]) -> list[tuple[str, str]]:
    return [(center, leaf) for leaf in sorted(leaves)]
