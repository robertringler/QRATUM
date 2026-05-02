r"""Layer 2 — Global evolution engine: causally consistent graph updates.

Evolves G_t → G_{t+1} via local rewrite rules applied in causal order.
No global synchronisation: uses topological ordering for deterministic
results while supporting asynchronous-compatible semantics.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from quasim.ciir.crs.graph import CRSGraph, Node
from quasim.ciir.crs.rewrite import RewriteEngine


# ================================================================
# CRS Evolution Engine
# ================================================================

class CRSEngine:
    """Global evolution engine for a CRS graph.

    Applies a :class:`RewriteEngine` to every node in topological
    (causal) order, collects per-step metrics, and maintains a run
    history for analysis.

    Parameters
    ----------
    graph : CRSGraph
        Initial graph state G_0.
    rewrite_engine : RewriteEngine
        Set of local rewrite rules.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        graph: CRSGraph,
        rewrite_engine: RewriteEngine,
        seed: int = 42,
    ) -> None:
        self.graph = graph
        self.rewrite_engine = rewrite_engine
        self.step_count: int = 0
        self.history: list[dict] = []
        self.rng: np.random.Generator = np.random.default_rng(seed)

    # ---- public API -------------------------------------------------

    def step(self) -> dict:
        """Evolve the graph by one discrete time step.

        Returns
        -------
        dict
            Metrics for this step (step, n_nodes, n_edges, …).
        """
        order = self._compute_update_order()
        updated_nodes: dict[int, Node] = {}
        for nid in order:
            if nid not in self.graph.nodes:
                continue
            updated = self.rewrite_engine.apply_rules(self.graph, nid, self.rng)
            updated_nodes[nid] = updated

        # Commit all updates atomically
        for nid, node in updated_nodes.items():
            self.graph.nodes[nid] = node

        self.step_count += 1
        metrics = self._compute_metrics()
        self.history.append(metrics)
        return metrics

    def evolve(self, n_steps: int) -> list[dict]:
        """Run *n_steps* evolution steps.

        Returns
        -------
        list[dict]
            Per-step metrics.
        """
        results: list[dict] = []
        for _ in range(n_steps):
            results.append(self.step())
        return results

    # ---- internal ---------------------------------------------------

    def _compute_update_order(self) -> list[int]:
        """Topological sort of nodes; falls back to id-order if cyclic."""
        in_degree: dict[int, int] = {nid: 0 for nid in self.graph.nodes}
        for nid in self.graph.nodes:
            for parent in self.graph._reverse_adj.get(nid, set()):
                if parent in self.graph.nodes:
                    in_degree[nid] = in_degree.get(nid, 0) + 1

        queue: deque[int] = deque(
            nid for nid, deg in in_degree.items() if deg == 0
        )
        order: list[int] = []
        while queue:
            nid = queue.popleft()
            order.append(nid)
            for child in self.graph._adjacency.get(nid, set()):
                if child in in_degree:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)

        # Fallback: append remaining nodes (cycle members) in id order
        if len(order) < len(self.graph.nodes):
            remaining = sorted(set(self.graph.nodes) - set(order))
            order.extend(remaining)

        return order

    def _compute_metrics(self) -> dict:
        """Collect per-step summary metrics."""
        norms: list[float] = []
        for node in self.graph.nodes.values():
            norms.append(float(np.linalg.norm(node.state)))

        total_weight = sum(e.causal_weight for e in self.graph.edges.values())

        # Shannon entropy estimate from binned state norms
        if norms:
            norm_arr = np.array(norms)
            hist, _ = np.histogram(norm_arr, bins=max(10, len(norms) // 5 + 1))
            probs = hist / hist.sum() if hist.sum() > 0 else hist
            probs = probs[probs > 0]
            entropy = -float(np.sum(probs * np.log(probs))) if len(probs) > 0 else 0.0
        else:
            entropy = 0.0

        return {
            "step": self.step_count,
            "n_nodes": self.graph.node_count,
            "n_edges": self.graph.edge_count,
            "mean_state_norm": float(np.mean(norms)) if norms else 0.0,
            "max_state_norm": float(np.max(norms)) if norms else 0.0,
            "total_causal_weight": total_weight,
            "entropy_estimate": entropy,
        }
