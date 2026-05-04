"""Action-embedding layer: Action → ΔS displacement mapping.

The displacement of an action depends on the *state* at which it is
applied. To preserve purity, we expose an explicit ``DisplacementCache``
object that callers may pass in; we never use a module-level mutable
cache. ``action_to_displacement`` accepts an optional cache and is pure
otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qagents.control_geometry.sensitivity import (
    DEFAULT_EPS,
    sensitivity_probe,
)
from qagents.mvri.action_space import Action
from qagents.mvri.state import State


@dataclass
class DisplacementCache:
    """Explicit cache for ``action_to_displacement`` results.

    The cache key is ``(state, action, eps)``. Because ``State`` and
    ``Action`` are frozen dataclasses they hash by value, so identical
    inputs collide deterministically.
    """

    _store: dict[tuple[State, Action, float], np.ndarray] = field(default_factory=dict)

    def get(self, key: tuple[State, Action, float]) -> np.ndarray | None:
        cached = self._store.get(key)
        if cached is None:
            return None
        # Return a copy so callers cannot mutate the cached vector.
        return cached.copy()

    def put(self, key: tuple[State, Action, float], value: np.ndarray) -> None:
        self._store[key] = value.copy()

    def __len__(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        self._store.clear()


def action_to_displacement(
    action: Action,
    state: State,
    *,
    eps: float = DEFAULT_EPS,
    cache: DisplacementCache | None = None,
) -> np.ndarray:
    """Map ``action`` at ``state`` to its induced ΔS vector.

    The returned vector lives in the fixed-dimension state-vector space
    (see :func:`state_to_vector`). When a cache is supplied, repeat
    calls return the same value without re-running ``F``.
    """

    if cache is not None:
        key = (state, action, float(eps))
        hit = cache.get(key)
        if hit is not None:
            return hit
        out = sensitivity_probe(state, action, eps)
        cache.put(key, out)
        return out.copy()
    return sensitivity_probe(state, action, eps)


def action_similarity(
    a1: Action,
    a2: Action,
    state: State,
    *,
    eps: float = DEFAULT_EPS,
    cache: DisplacementCache | None = None,
    zero_tol: float = 1e-12,
) -> float:
    """Cosine similarity between displacement vectors of ``a1`` and ``a2``.

    If either action induces a zero (or numerically-zero) displacement
    the function returns ``0.0`` instead of raising — orthogonal-by-
    convention.
    """

    d1 = action_to_displacement(a1, state, eps=eps, cache=cache)
    d2 = action_to_displacement(a2, state, eps=eps, cache=cache)
    n1 = float(np.linalg.norm(d1))
    n2 = float(np.linalg.norm(d2))
    if n1 < zero_tol or n2 < zero_tol:
        return 0.0
    return float(np.dot(d1, d2) / (n1 * n2))


__all__ = [
    "DisplacementCache",
    "action_similarity",
    "action_to_displacement",
]
