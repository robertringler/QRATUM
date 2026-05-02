"""Deterministic network performance simulation."""

from __future__ import annotations


def simulate_latency(paths: dict[tuple[str, str], int], route: tuple[str, str]) -> int:
    if route not in paths:
        raise KeyError(route)
    hops = paths[route]
    return hops * 2


def path_budget(paths: dict[tuple[str, str], int]) -> int:
    return sum(paths.values())
