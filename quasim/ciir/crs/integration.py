r"""Layer 7 — Integration with CIIR / QuASIM / QRATUM.

Maps:
- CIIR distortion → rewrite nonlinearity (constraint-induced noise in updates)
- QuASIM → forward simulation engine of CRSGraph
- QRATUM → recursive self-modifying update kernel

Provides :class:`QRATUM_CRS_Core` as the unified interface.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quasim.ciir.crs.branching import BranchingEngine
from quasim.ciir.crs.conservation import (
    ConservationEngine,
    graph_energy_invariant,
    node_count_invariant,
    total_causal_weight_invariant,
    total_state_norm_invariant,
)
from quasim.ciir.crs.evolution import CRSEngine
from quasim.ciir.crs.graph import CRSGraph, Edge, Node
from quasim.ciir.crs.spacetime import (
    curvature_proxy,
    graph_dimension_estimate,
)

# ================================================================
# Density-matrix bridge
# ================================================================


def graph_to_density_matrix(graph: CRSGraph) -> NDArray[np.floating]:
    r"""Construct a density matrix from graph state vectors.

    ρ = (1/N) Σ_v |s_v⟩⟨s_v|

    Parameters
    ----------
    graph : CRSGraph

    Returns
    -------
    NDArray
        Shape (d, d) positive-semidefinite density matrix.
    """
    if not graph.nodes:
        return np.eye(1)

    states = [n.state for n in graph.nodes.values()]
    d = states[0].shape[0]
    rho = np.zeros((d, d))
    for s in states:
        rho += np.outer(s, s)
    rho /= len(states)

    # Ensure trace normalisation
    tr = np.trace(rho)
    if tr > 1e-30:
        rho /= tr
    return rho


def density_matrix_to_graph(
    rho: NDArray[np.floating],
    n_nodes: int,
) -> CRSGraph:
    r"""Reverse map: eigendecompose ρ and create nodes from eigenvectors.

    Parameters
    ----------
    rho : NDArray
        Density matrix (d × d).
    n_nodes : int
        Number of nodes to create (uses the top-*n_nodes* eigenvectors).

    Returns
    -------
    CRSGraph
    """
    eigvals, eigvecs = np.linalg.eigh(rho)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    g = CRSGraph()
    k = min(n_nodes, len(eigvals))
    for i in range(k):
        lam = max(eigvals[i], 0.0)
        state = eigvecs[:, i] * np.sqrt(lam)
        g.add_node(Node(id=i, state=state))

    # Remaining nodes get near-zero states
    d = rho.shape[0]
    for i in range(k, n_nodes):
        g.add_node(Node(id=i, state=np.zeros(d)))

    # Connect all pairs with weight proportional to state overlap
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            si = g.nodes[i].state
            sj = g.nodes[j].state
            ni = np.linalg.norm(si)
            nj = np.linalg.norm(sj)
            if ni > 1e-12 and nj > 1e-12:
                overlap = float(np.abs(np.dot(si, sj)) / (ni * nj))
            else:
                overlap = 0.0
            if overlap > 0.01:
                g.add_edge(Edge(source=i, target=j, causal_weight=overlap))
                g.add_edge(Edge(source=j, target=i, causal_weight=overlap))

    return g


# ================================================================
# CIIR distortion bridge
# ================================================================


def apply_ciir_distortion(
    graph: CRSGraph,
    kappa_min: float = 1.0,
    diameter: float = 1.0,
) -> CRSGraph:
    r"""Apply CIIR contraction to all node states.

    s_v' = s_v · exp(−κ_min · D)

    Corresponds to the interface map's distortion bound (CIIR T8.4).

    Parameters
    ----------
    graph : CRSGraph
    kappa_min : float
        Minimum sectional curvature bound.
    diameter : float
        Manifold diameter.

    Returns
    -------
    CRSGraph
        New graph with contracted states (deep copy).
    """
    factor = float(np.exp(-kappa_min * diameter))
    g = graph.copy()
    for node in g.nodes.values():
        node.state = node.state * factor
    return g


# ================================================================
# QRATUM_CRS_Core — unified interface
# ================================================================


class QRATUM_CRS_Core:
    """Unified CRS interface integrating evolution, conservation, and branching.

    Parameters
    ----------
    engine : CRSEngine
        Graph evolution engine.
    conservation : ConservationEngine | None
        Conservation law checker (auto-created with defaults if *None*).
    branching : BranchingEngine | None
        Optional branching engine for quantum-like superposition.
    distortion_kappa : float
        CIIR curvature parameter κ_min.
    distortion_diameter : float
        CIIR manifold diameter D.
    """

    def __init__(
        self,
        engine: CRSEngine,
        conservation: ConservationEngine | None = None,
        branching: BranchingEngine | None = None,
        distortion_kappa: float = 0.0,
        distortion_diameter: float = 1.0,
    ) -> None:
        self.engine = engine
        self.branching = branching
        self.distortion_kappa = distortion_kappa
        self.distortion_diameter = distortion_diameter

        if conservation is None:
            conservation = ConservationEngine()
            conservation.register_invariant(total_state_norm_invariant())
            conservation.register_invariant(total_causal_weight_invariant())
            conservation.register_invariant(node_count_invariant())
            conservation.register_invariant(graph_energy_invariant())
        self.conservation = conservation

    # ---- single step ------------------------------------------------

    def step(self) -> dict:
        """Run one evolution step + distortion + conservation check.

        Returns
        -------
        dict
            Combined metrics including conservation violations.
        """
        before = self.engine.graph.copy()

        metrics = self.engine.step()

        # Apply CIIR distortion
        if self.distortion_kappa > 0.0:
            self.engine.graph = apply_ciir_distortion(
                self.engine.graph,
                kappa_min=self.distortion_kappa,
                diameter=self.distortion_diameter,
            )

        conservation_report = self.conservation.check_invariants(before, self.engine.graph)
        metrics["conservation"] = conservation_report
        metrics["conservation_violations"] = sum(1 for r in conservation_report if r["violated"])
        return metrics

    # ---- full experiment --------------------------------------------

    def run_experiment(self, n_steps: int) -> dict:
        """Run *n_steps* and aggregate time-series metrics.

        Returns
        -------
        dict
            Keys map to lists of per-step values.
        """
        time_series: dict[str, list] = {
            "step": [],
            "n_nodes": [],
            "n_edges": [],
            "mean_state_norm": [],
            "max_state_norm": [],
            "total_causal_weight": [],
            "entropy_estimate": [],
            "conservation_violations": [],
        }

        for _ in range(n_steps):
            m = self.step()
            for key in time_series:
                time_series[key].append(m.get(key))
        return time_series

    # ---- accessors --------------------------------------------------

    def get_density_matrix(self) -> NDArray[np.floating]:
        """Return the current graph's density-matrix representation."""
        return graph_to_density_matrix(self.engine.graph)

    def get_emergent_metrics(self) -> dict:
        """Return emergent spacetime metrics.

        Returns
        -------
        dict
            dimension_estimate, mean_curvature, n_nodes, n_edges.
        """
        g = self.engine.graph
        dim = graph_dimension_estimate(g)
        ids = sorted(g.nodes)
        sample = ids[:: max(1, len(ids) // 10)]
        curvatures = [curvature_proxy(g, nid) for nid in sample]
        mean_curv = float(np.mean(curvatures)) if curvatures else 0.0
        return {
            "dimension_estimate": dim,
            "mean_curvature": mean_curv,
            "n_nodes": g.node_count,
            "n_edges": g.edge_count,
        }
