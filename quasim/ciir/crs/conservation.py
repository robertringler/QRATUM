r"""Layer 5 — Conservation law engine: detect and verify graph invariants.

Implements Noether-analog invariant detection:
- Total state norm: sum of ||s_v|| over all nodes
- Total causal weight: sum of w_e over all edges
- Node count (topological invariant)
- State centroid (centre of mass analog)
- Graph Laplacian spectrum (spectral invariants)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from quasim.ciir.crs.graph import CRSGraph

# ================================================================
# Invariant primitive
# ================================================================

@dataclass
class Invariant:
    """A named conserved quantity with tolerance.

    Attributes
    ----------
    name : str
        Human-readable name.
    compute : Callable[[CRSGraph], float]
        Function that extracts the scalar invariant from a graph.
    tolerance : float
        Absolute tolerance for violation detection.
    """

    name: str
    compute: Callable[[CRSGraph], float]
    tolerance: float = 0.1


# ================================================================
# Conservation engine
# ================================================================

class ConservationEngine:
    """Registry and checker of conserved quantities.

    Compares invariant values before and after a graph transformation
    and reports violations.
    """

    def __init__(self) -> None:
        self.invariants: list[Invariant] = []

    def register_invariant(self, inv: Invariant) -> None:
        """Add an invariant to the registry."""
        self.invariants.append(inv)

    def check_invariants(
        self,
        before: CRSGraph,
        after: CRSGraph,
    ) -> list[dict]:
        """Compare invariant values between *before* and *after* graphs.

        Returns
        -------
        list[dict]
            One dict per invariant with keys: name, before_value,
            after_value, delta, violated.
        """
        results: list[dict] = []
        for inv in self.invariants:
            val_before = inv.compute(before)
            val_after = inv.compute(after)
            delta = abs(val_after - val_before)
            results.append(
                {
                    "name": inv.name,
                    "before_value": val_before,
                    "after_value": val_after,
                    "delta": delta,
                    "violated": delta > inv.tolerance,
                }
            )
        return results


# ================================================================
# Built-in invariants
# ================================================================

def total_state_norm_invariant(tol: float = 0.1) -> Invariant:
    r"""Σ_v ||s_v||₂ — total state norm."""

    def _compute(graph: CRSGraph) -> float:
        return float(sum(np.linalg.norm(n.state) for n in graph.nodes.values()))

    return Invariant(name="total_state_norm", compute=_compute, tolerance=tol)


def total_causal_weight_invariant(tol: float = 0.1) -> Invariant:
    r"""Σ_e w_e — total causal weight."""

    def _compute(graph: CRSGraph) -> float:
        return float(sum(e.causal_weight for e in graph.edges.values()))

    return Invariant(name="total_causal_weight", compute=_compute, tolerance=tol)


def node_count_invariant() -> Invariant:
    """|V| — node count (exact topological invariant)."""

    def _compute(graph: CRSGraph) -> float:
        return float(graph.node_count)

    return Invariant(name="node_count", compute=_compute, tolerance=0.0)


def state_centroid_invariant(tol: float = 0.05) -> Invariant:
    r"""||mean(s_v)|| — norm of the state centroid."""

    def _compute(graph: CRSGraph) -> float:
        if not graph.nodes:
            return 0.0
        states = np.array([n.state for n in graph.nodes.values()])
        centroid = states.mean(axis=0)
        return float(np.linalg.norm(centroid))

    return Invariant(name="state_centroid", compute=_compute, tolerance=tol)


def graph_energy_invariant(tol: float = 0.1) -> Invariant:
    r"""Σ_e w_e ||s_{src} − s_{tgt}||² — elastic energy analog."""

    def _compute(graph: CRSGraph) -> float:
        energy = 0.0
        for edge in graph.edges.values():
            src = graph.nodes.get(edge.source)
            tgt = graph.nodes.get(edge.target)
            if src is None or tgt is None:
                continue
            diff_sq = float(np.sum((src.state - tgt.state) ** 2))
            energy += edge.causal_weight * diff_sq
        return energy

    return Invariant(name="graph_energy", compute=_compute, tolerance=tol)
