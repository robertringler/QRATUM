"""Deterministic in-memory cache for feed sandboxing."""
from __future__ import annotations



class DeterministicCache:
    def __init__(self) -> None:
        self._store: dict[tuple[str, int], object] = {}

    def set(self, key: str, tick: int, value: object) -> None:
        self._store[(key, tick)] = value

    def get(self, key: str, tick: int) -> object:
        return self._store.get((key, tick))

    def snapshot(self) -> dict[tuple[str, int], object]:
        return dict(sorted(self._store.items(), key=lambda kv: (kv[0][0], kv[0][1])))
