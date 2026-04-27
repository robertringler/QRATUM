"""Reachability layer: local R(S), reachability graphs, and costs.

Reachability is approximated as the image of a bounded action set under
the deterministic forward model. The constraint gate is *consulted* but
never bypassed — by default ``local_reachability`` only includes states
that pass ``Φ`` (this is configurable via ``constraint_gated``).

The graph type used here is a dependency-free dataclass; we deliberately
avoid pulling in NetworkX so the layer remains a minimal control-theory
primitive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from qagents.mvri.action_space import EPSILON, Action
from qagents.mvri.constraint_gate import constraint_gate
from qagents.mvri.forward_model import forward_model
from qagents.mvri.state import State

# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ReachEdge:
    """Edge in the reachability graph."""

    src: State
    dst: State
    action: Action
    weight: float  # action norm


@dataclass
class ReachabilityGraph:
    """Directed weighted graph of reachable transitions.

    Nodes are MVRI states (hashable because ``State`` is frozen). Edges
    are :class:`ReachEdge` records carrying the originating action and
    its norm as edge weight.
    """

    nodes: set[State] = field(default_factory=set)
    edges: list[ReachEdge] = field(default_factory=list)

    def add_node(self, s: State) -> None:
        self.nodes.add(s)

    def add_edge(self, edge: ReachEdge) -> None:
        self.nodes.add(edge.src)
        self.nodes.add(edge.dst)
        self.edges.append(edge)

    def successors(self, s: State) -> list[ReachEdge]:
        return [e for e in self.edges if e.src == s]

    def predecessors(self, s: State) -> list[ReachEdge]:
        return [e for e in self.edges if e.dst == s]

    def __len__(self) -> int:
        return len(self.edges)


# --------------------------------------------------------------------------- #
# Action norm
# --------------------------------------------------------------------------- #


def action_norm(a: Action) -> float:
    """Norm of an action used for graph weights and cost minimisation.

    For all current action types the magnitude is one-dimensional, so
    the norm is simply ``|magnitude|``. This function is provided as a
    seam: future multi-dimensional action types should override it.
    """

    return abs(float(a.magnitude))


# --------------------------------------------------------------------------- #
# Reachability primitives
# --------------------------------------------------------------------------- #


def local_reachability(
    state: State,
    actions: Iterable[Action],
    *,
    epsilon: float = EPSILON,
    constraint_gated: bool = True,
) -> set[State]:
    """Approximate ``R(state) = { F(state, a) : ||a|| ≤ ε }``.

    Parameters
    ----------
    state:
        Current MVRI state.
    actions:
        Iterable of candidate actions. Actions whose norm exceeds
        ``epsilon`` are silently filtered out — they are not in the
        bounded set.
    epsilon:
        Norm bound for the action set.
    constraint_gated:
        When ``True`` (default), only states for which the joint
        constraint gate Φ accepts the transition are returned. When
        ``False`` the raw image of ``F`` is returned. **The default is
        gated** — controllability analysis must respect the same Φ as
        the live system.
    """

    out: set[State] = set()
    for a in actions:
        if action_norm(a) > epsilon + 1e-15:
            continue
        candidate = forward_model(state, a)
        if constraint_gated:
            ok, _ = constraint_gate(state, candidate, a)
            if not ok:
                continue
        out.add(candidate)
    return out


def reachability_graph(
    states: Iterable[State],
    transitions: Iterable[tuple[State, Action]],
    *,
    constraint_gated: bool = True,
) -> ReachabilityGraph:
    """Build a directed weighted graph of valid transitions.

    Parameters
    ----------
    states:
        Initial node set; every state will be added as a node.
    transitions:
        Iterable of ``(state, action)`` pairs to evaluate. The image
        ``F(state, action)`` becomes the edge target. When
        ``constraint_gated`` is True (default), transitions rejected by
        Φ are skipped.
    """

    g = ReachabilityGraph()
    for s in states:
        g.add_node(s)
    for s, a in transitions:
        nxt = forward_model(s, a)
        if constraint_gated:
            ok, _ = constraint_gate(s, nxt, a)
            if not ok:
                continue
        g.add_edge(ReachEdge(src=s, dst=nxt, action=a, weight=action_norm(a)))
    return g


def reachability_cost(
    s: State,
    s_prime: State,
    actions: Iterable[Action],
    *,
    constraint_gated: bool = True,
    not_reachable: float = float("inf"),
) -> float:
    """Minimum action norm in ``actions`` that reaches ``s_prime`` from ``s``.

    Returns ``not_reachable`` (default ``+inf``) when no candidate
    action reaches the target. The function is deterministic: given
    identical ``actions`` (in any order) it produces the same answer.
    """

    best: float | None = None
    for a in actions:
        candidate = forward_model(s, a)
        if candidate != s_prime:
            continue
        if constraint_gated:
            ok, _ = constraint_gate(s, candidate, a)
            if not ok:
                continue
        n = action_norm(a)
        if best is None or n < best:
            best = n
    return float(best) if best is not None else float(not_reachable)


__all__ = [
    "ReachEdge",
    "ReachabilityGraph",
    "action_norm",
    "local_reachability",
    "reachability_cost",
    "reachability_graph",
]
