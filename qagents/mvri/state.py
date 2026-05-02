"""Typed, immutable state for the Minimal Viable Reality Injector (MVRI).

The state is a directed weighted graph plus per-node activations. All fields
are immutable (``tuple`` / ``frozenset``) so that ``State`` instances are
hashable and safe to pass between pure functions without defensive copying.

Helper predicates (``H``, ``edge_set``, ``is_stable``, ``no_invalid_states``,
``Inv``) are defined here so that every consumer (forward model, constraint
gate, metrics) shares the same definitions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping

# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class StateError(ValueError):
    """Raised when constructing an inconsistent State."""


class ForwardModelError(RuntimeError):
    """Raised by the forward model on unhandled action types."""


# --------------------------------------------------------------------------- #
# State dataclass
# --------------------------------------------------------------------------- #


Edge = tuple[str, str]


@dataclass(frozen=True)
class State:
    """Immutable system state.

    Attributes
    ----------
    nodes:
        Sorted tuple of node identifiers.
    activations:
        Sorted tuple of ``(node_id, activation)`` pairs. Each activation
        must be finite and lie inside ``bounds``.
    edges:
        Frozen set of directed ``(src, dst)`` tuples.
    edge_weights:
        Sorted tuple of ``((src, dst), weight)`` pairs. Every edge in
        ``edges`` must appear here.
    bounds:
        ``(low, high)`` activation bounds.
    initial_edges:
        Frozen set of edges of the *initial* state. This is preserved
        across the trajectory so that the topology-monotonicity
        constraint (no new edges) can be checked.
    step:
        Iteration counter; never affects equality of the underlying
        physical configuration but is logged for auditability.
    """

    nodes: tuple[str, ...]
    activations: tuple[tuple[str, float], ...]
    edges: frozenset[Edge]
    edge_weights: tuple[tuple[Edge, float], ...]
    bounds: tuple[float, float]
    initial_edges: frozenset[Edge]
    step: int = 0
    metadata: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:  # noqa: D401 - dataclass hook
        if len(self.bounds) != 2 or self.bounds[0] >= self.bounds[1]:
            raise StateError(f"invalid bounds: {self.bounds!r}")
        if len(set(self.nodes)) != len(self.nodes):
            raise StateError("duplicate node ids")
        if tuple(sorted(self.nodes)) != self.nodes:
            raise StateError("nodes must be sorted")
        act_keys = tuple(k for k, _ in self.activations)
        if act_keys != self.nodes:
            raise StateError("activations must match nodes (sorted, complete)")
        for n, v in self.activations:
            if not math.isfinite(v):
                raise StateError(f"non-finite activation for {n!r}")
        ew_keys = tuple(k for k, _ in self.edge_weights)
        if set(ew_keys) != set(self.edges):
            raise StateError("edge_weights must match edges")
        if tuple(sorted(ew_keys)) != ew_keys:
            raise StateError("edge_weights must be sorted")
        for e in self.edges:
            if e[0] not in self.nodes or e[1] not in self.nodes:
                raise StateError(f"edge {e!r} references unknown node")
        # initial_edges is allowed to be a (non-strict) superset of edges,
        # since edges may only shrink during a trajectory.
        for w in (w for _, w in self.edge_weights):
            if not math.isfinite(w):
                raise StateError("non-finite edge weight")

    # ------------------------------------------------------------------ #
    # Read helpers
    # ------------------------------------------------------------------ #

    def activation(self, node: str) -> float:
        for n, v in self.activations:
            if n == node:
                return v
        raise KeyError(node)

    def weight(self, edge: Edge) -> float:
        for e, w in self.edge_weights:
            if e == edge:
                return w
        raise KeyError(edge)

    def has_target(self, target: str) -> bool:
        """Targets are either node-ids or ``"src->dst"`` edge specs."""
        if "->" in target:
            return parse_edge(target) in self.edges
        return target in self.nodes


# --------------------------------------------------------------------------- #
# Constructors / utilities
# --------------------------------------------------------------------------- #


def parse_edge(target: str) -> Edge:
    src, _, dst = target.partition("->")
    if not src or not dst:
        raise ValueError(f"invalid edge target: {target!r}")
    return (src, dst)


def format_edge(edge: Edge) -> str:
    return f"{edge[0]}->{edge[1]}"


def make_state(
    activations: Mapping[str, float],
    edge_weights: Mapping[Edge, float],
    bounds: tuple[float, float] = (-1.0, 1.0),
    *,
    step: int = 0,
    initial_edges: Iterable[Edge] | None = None,
    metadata: Mapping[str, str] | None = None,
) -> State:
    """Convenience constructor that handles sorting / freezing."""

    nodes = tuple(sorted(activations))
    acts = tuple(sorted(activations.items()))
    edges = frozenset(edge_weights.keys())
    ew = tuple(sorted(edge_weights.items()))
    init = frozenset(initial_edges) if initial_edges is not None else edges
    meta = tuple(sorted((metadata or {}).items()))
    return State(
        nodes=nodes,
        activations=acts,
        edges=edges,
        edge_weights=ew,
        bounds=bounds,
        initial_edges=init,
        step=step,
        metadata=meta,
    )


def replace_state(
    s: State,
    *,
    activations: Mapping[str, float] | None = None,
    edge_weights: Mapping[Edge, float] | None = None,
    step: int | None = None,
) -> State:
    """Pure update: returns a new State with the given fields replaced."""

    new_acts = (
        tuple(sorted(activations.items()))
        if activations is not None
        else s.activations
    )
    new_ew = (
        tuple(sorted(edge_weights.items()))
        if edge_weights is not None
        else s.edge_weights
    )
    new_edges = (
        frozenset(edge_weights.keys()) if edge_weights is not None else s.edges
    )
    return State(
        nodes=tuple(k for k, _ in new_acts),
        activations=new_acts,
        edges=new_edges,
        edge_weights=new_ew,
        bounds=s.bounds,
        initial_edges=s.initial_edges,
        step=step if step is not None else s.step,
        metadata=s.metadata,
    )


# --------------------------------------------------------------------------- #
# Predicates / observables (pure)
# --------------------------------------------------------------------------- #


def edge_set(s: State) -> frozenset[Edge]:
    return s.edges


def H(s: State) -> float:  # noqa: N802 - mathematical name from spec
    """Shannon entropy (bits) of the normalized |activation| distribution.

    Strictly non-negative; zero when probability mass is concentrated on a
    single node. If all activations are zero the convention is ``H = 0``.
    """

    weights = [abs(v) for _, v in s.activations]
    total = sum(weights)
    if total <= 0.0:
        return 0.0
    h = 0.0
    for w in weights:
        if w <= 0.0:
            continue
        p = w / total
        h -= p * math.log2(p)
    return h


def is_stable(s: State, tol: float = 1e-9) -> bool:
    """Structural stability: bounded activations and bounded weights.

    A state is stable iff every activation is strictly within
    ``bounds`` (with tolerance) and every edge weight is finite.
    """

    lo, hi = s.bounds
    for _, v in s.activations:
        if not math.isfinite(v):
            return False
        if v < lo - tol or v > hi + tol:
            return False
    return all(math.isfinite(w) for _, w in s.edge_weights)


def no_invalid_states(s: State) -> bool:
    """Internal consistency: schema, no NaN, no duplicate edges, bounds sane."""

    try:
        # Re-running validation via a no-op replacement asserts schema.
        for n, v in s.activations:
            if not math.isfinite(v):
                return False
            if n not in s.nodes:
                return False
        for e, w in s.edge_weights:
            if e not in s.edges:
                return False
            if not math.isfinite(w):
                return False
        if s.bounds[0] >= s.bounds[1]:
            return False
    except Exception:  # pragma: no cover - defensive  # noqa: BLE001
        return False
    return True


def Inv(s: State) -> bool:  # noqa: N802 - mathematical name from spec
    """Hard invariants required at every accepted state.

    - ``H(s) ≥ 0``
    - ``edge_set(s) ⊆ initial_edges``
    - All activations within ``bounds``
    - Schema consistency (``no_invalid_states``)
    """

    if not no_invalid_states(s):
        return False
    if H(s) < -1e-12:
        return False
    if not s.edges.issubset(s.initial_edges):
        return False
    return is_stable(s)


__all__ = [
    "Edge",
    "ForwardModelError",
    "H",
    "Inv",
    "State",
    "StateError",
    "edge_set",
    "format_edge",
    "is_stable",
    "make_state",
    "no_invalid_states",
    "parse_edge",
    "replace_state",
]
