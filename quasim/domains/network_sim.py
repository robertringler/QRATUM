"""Deterministic network performance simulation."""

from __future__ import annotations

from typing import Dict, Tuple


def simulate_latency(paths: Dict[Tuple[str, str], int], route: Tuple[str, str]) -> int:
    if route not in paths:
        raise KeyError(route)
    hops = paths[route]
    return hops * 2


def path_budget(paths: Dict[Tuple[str, str], int]) -> int:
    return sum(paths.values())
