r"""Layer 1 — Local update operator: rewrite rules as physics law analogs.

Each rewrite rule R: N(v) → N'(v) maps a local neighborhood to an updated
neighborhood.  Rules are:
- Strictly local: depend only on radius-1 neighborhood
- Bounded complexity: O(deg(v) * d) per node
- Hybrid: deterministic core + stochastic perturbation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from quasim.ciir.crs.graph import CRSGraph, Node

# ================================================================
# Rewrite primitives
# ================================================================

@dataclass
class RewriteRule:
    """A named, probabilistic rewrite rule.

    Attributes
    ----------
    name : str
        Human-readable rule identifier.
    apply : Callable
        Signature ``(graph, node_id, rng) -> NDArray`` returning the updated
        state vector for *node_id*.
    probability : float
        Probability that this rule fires on any given application.
    """

    name: str
    apply: Callable[[CRSGraph, int, np.random.Generator], NDArray[np.floating]]
    probability: float = 1.0


class RewriteEngine:
    """Registry of rewrite rules applied probabilistically to graph nodes."""

    def __init__(self) -> None:
        self.rules: list[RewriteRule] = []

    def register_rule(self, rule: RewriteRule) -> None:
        """Append *rule* to the engine's rule list."""
        self.rules.append(rule)

    def apply_rules(
        self,
        graph: CRSGraph,
        node_id: int,
        rng: np.random.Generator,
    ) -> Node:
        """Apply all registered rules (in order, probabilistically) to *node_id*.

        Returns a *new* Node with the updated state.
        """
        node = graph.nodes[node_id]
        state = node.state.copy()
        for rule in self.rules:
            if rng.random() < rule.probability:
                state = rule.apply(graph, node_id, rng)
                # Feed the updated state back so subsequent rules see it
                graph.nodes[node_id] = Node(
                    id=node_id,
                    state=state,
                    timestamp=node.timestamp,
                    metadata=node.metadata,
                )
        updated = Node(
            id=node_id,
            state=state,
            timestamp=node.timestamp + 1,
            metadata=node.metadata.copy(),
        )
        return updated


# ================================================================
# Built-in rules
# ================================================================

def diffusion_rule(diffusion_rate: float = 0.1) -> RewriteRule:
    r"""Diffusion: s_v' = (1-α) s_v + α mean(s_n).

    Parameters
    ----------
    diffusion_rate : float
        Mixing coefficient α ∈ [0, 1].
    """

    def _apply(graph: CRSGraph, node_id: int, rng: np.random.Generator) -> NDArray:
        node = graph.nodes[node_id]
        neighbors = graph._adjacency.get(node_id, set())
        if not neighbors:
            return node.state.copy()
        nbr_states = np.array([graph.nodes[n].state for n in neighbors if n in graph.nodes])
        if len(nbr_states) == 0:
            return node.state.copy()
        mean_nbr = nbr_states.mean(axis=0)
        return (1.0 - diffusion_rate) * node.state + diffusion_rate * mean_nbr

    return RewriteRule(name="diffusion", apply=_apply, probability=1.0)


def decay_rule(decay_rate: float = 0.01) -> RewriteRule:
    r"""Exponential decay: s_v' = s_v (1 − δ)."""

    def _apply(graph: CRSGraph, node_id: int, rng: np.random.Generator) -> NDArray:
        return graph.nodes[node_id].state * (1.0 - decay_rate)

    return RewriteRule(name="decay", apply=_apply, probability=1.0)


def interaction_rule(coupling: float = 0.05) -> RewriteRule:
    r"""Neighbour interaction: s_v' = s_v + coupling Σ w_{vn} s_n."""

    def _apply(graph: CRSGraph, node_id: int, rng: np.random.Generator) -> NDArray:
        node = graph.nodes[node_id]
        s = node.state.copy()
        for nbr_id in graph._adjacency.get(node_id, set()):
            if nbr_id not in graph.nodes:
                continue
            edge = graph.edges.get((node_id, nbr_id))
            if edge is None:
                continue
            s = s + coupling * edge.causal_weight * graph.nodes[nbr_id].state
        return s

    return RewriteRule(name="interaction", apply=_apply, probability=1.0)


def stochastic_perturbation_rule(noise_scale: float = 0.01) -> RewriteRule:
    r"""Stochastic kick: s_v' = s_v + σ N(0, 1)."""

    def _apply(graph: CRSGraph, node_id: int, rng: np.random.Generator) -> NDArray:
        node = graph.nodes[node_id]
        return node.state + noise_scale * rng.standard_normal(node.state.shape)

    return RewriteRule(name="stochastic_perturbation", apply=_apply, probability=1.0)


def edge_reweight_rule(learning_rate: float = 0.01) -> RewriteRule:
    r"""Hebbian edge reweight: w_{vn}' = clip(w_{vn} + lr |s_v − s_n|, 0, 1).

    Note: this rule modifies edges as a side-effect and returns the
    original state unchanged.
    """

    def _apply(graph: CRSGraph, node_id: int, rng: np.random.Generator) -> NDArray:
        node = graph.nodes[node_id]
        for nbr_id in list(graph._adjacency.get(node_id, set())):
            key = (node_id, nbr_id)
            edge = graph.edges.get(key)
            if edge is None or nbr_id not in graph.nodes:
                continue
            diff = float(np.linalg.norm(node.state - graph.nodes[nbr_id].state))
            new_w = float(np.clip(edge.causal_weight + learning_rate * diff, 0.0, 1.0))
            edge.causal_weight = new_w
        return node.state.copy()

    return RewriteRule(name="edge_reweight", apply=_apply, probability=1.0)
