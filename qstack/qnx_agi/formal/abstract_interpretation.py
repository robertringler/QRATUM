"""Abstract interpretation utilities using interval domains."""
from __future__ import annotations

from dataclasses import dataclass

from .interval_arithmetic import Interval


@dataclass(frozen=True)
class AbstractLattice:
    bottom: float = float("-inf")
    top: float = float("inf")

    def join(self, a: Interval, b: Interval) -> Interval:
        return Interval(min(a.low, b.low), max(a.high, b.high))

    def meet(self, a: Interval, b: Interval) -> Interval:
        low = max(a.low, b.low)
        high = min(a.high, b.high)
        if low > high:
            raise ValueError("meet of disjoint intervals is empty")
        return Interval(low, high)

    def widen(self, previous: Interval, current: Interval) -> Interval:
        low = previous.low if current.low >= previous.low else self.bottom
        high = previous.high if current.high <= previous.high else self.top
        return Interval(low, high)


class AbstractState:
    """Tracks abstract intervals for each program variable."""

    def __init__(self, lattice: AbstractLattice | None = None):
        self._lattice = lattice or AbstractLattice()
        self._store: dict[str, Interval] = {}

    def assign(self, name: str, interval: Interval) -> None:
        self._store[name] = interval

    def read(self, name: str) -> Interval:
        iv = self._store.get(name)
        if iv is None:
            raise KeyError(f"abstract variable {name} not found")
        return iv

    def join(self, other: "AbstractState") -> "AbstractState":
        merged = AbstractState(self._lattice)
        keys = set(self._store) | set(other._store)
        for key in keys:
            a = self._store.get(key, Interval(self._lattice.bottom, self._lattice.bottom))
            b = other._store.get(key, Interval(self._lattice.bottom, self._lattice.bottom))
            merged.assign(key, self._lattice.join(a, b))
        return merged

    def apply_guard(self, name: str, constraint: Interval) -> None:
        current = self.read(name)
        self.assign(name, self._lattice.meet(current, constraint))

    def widen(self, previous: "AbstractState") -> "AbstractState":
        widened = AbstractState(self._lattice)
        for key, interval in self._store.items():
            prev = previous._store.get(key, interval)
            widened.assign(key, self._lattice.widen(prev, interval))
        return widened

    def snapshot(self) -> dict[str, Interval]:
        return dict(self._store)
