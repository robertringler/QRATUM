r"""Layer 6 — Observer subsystem: measurement model with information loss.

An observer is a subgraph that can partially observe the full graph.
Measurement projects the full state onto the observer's accessible subspace,
with information loss modelled by the observer's limited connectivity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from quasim.ciir.crs.graph import CRSGraph

# ================================================================
# Observer primitive
# ================================================================

@dataclass
class Observer:
    """Partial-access observer within a CRS graph.

    Attributes
    ----------
    id : int
        Observer identifier.
    accessible_nodes : set[int]
        Node ids this observer can perceive.
    memory : NDArray[np.floating]
        Internal memory vector (aggregated observation state).
    resolution : float
        Observation fidelity ∈ (0, 1].  1 = perfect observation.
    """

    id: int
    accessible_nodes: set[int] = field(default_factory=set)
    memory: NDArray[np.floating] = field(
        default_factory=lambda: np.zeros(1)
    )
    resolution: float = 1.0


# ================================================================
# Observation functions
# ================================================================

def observe(
    observer: Observer,
    graph: CRSGraph,
    rng: np.random.Generator | None = None,
) -> dict[int, NDArray[np.floating]]:
    r"""Partial measurement of the graph through the observer.

    For each accessible node present in *graph*:

        observed_state = resolution * true_state + (1 − resolution) * noise

    Parameters
    ----------
    observer : Observer
    graph : CRSGraph
    rng : numpy.random.Generator, optional

    Returns
    -------
    dict[int, NDArray]
        Mapping node_id → observed state vector.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    observations: dict[int, NDArray[np.floating]] = {}
    for nid in observer.accessible_nodes:
        node = graph.nodes.get(nid)
        if node is None:
            continue
        noise = rng.standard_normal(node.state.shape)
        obs = observer.resolution * node.state + (1.0 - observer.resolution) * noise
        observations[nid] = obs
    return observations


def measurement_entropy(
    observer: Observer,
    graph: CRSGraph,
    rng: np.random.Generator | None = None,
    n_bins: int = 20,
) -> float:
    r"""Shannon entropy of the binned observed state norms.

    Parameters
    ----------
    observer : Observer
    graph : CRSGraph
    rng : numpy.random.Generator, optional
    n_bins : int
        Number of histogram bins.

    Returns
    -------
    float
        Shannon entropy in nats.
    """
    obs = observe(observer, graph, rng=rng)
    if not obs:
        return 0.0
    norms = np.array([float(np.linalg.norm(v)) for v in obs.values()])
    if norms.std() < 1e-12:
        return 0.0
    hist, _ = np.histogram(norms, bins=n_bins)
    probs = hist / hist.sum()
    probs = probs[probs > 0]
    return -float(np.sum(probs * np.log(probs)))


# ================================================================
# Decoherence and partial trace
# ================================================================

def decohere(
    graph: CRSGraph,
    observer: Observer,
    decoherence_rate: float = 0.1,
    rng: np.random.Generator | None = None,
) -> CRSGraph:
    r"""Measurement back-action: decohere observed nodes.

    For each node in the observer's accessible set, mix the state
    toward the component-wise mean (simulating off-diagonal coherence
    loss):

        s_v' = (1 − γ) s_v + γ mean(s_v)

    Parameters
    ----------
    graph : CRSGraph
    observer : Observer
    decoherence_rate : float
        γ ∈ [0, 1].
    rng : numpy.random.Generator, optional

    Returns
    -------
    CRSGraph
        New graph with decohered states (deep copy).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    g = graph.copy()
    for nid in observer.accessible_nodes:
        node = g.nodes.get(nid)
        if node is None:
            continue
        uniform = np.full_like(node.state, node.state.mean())
        new_state = (1.0 - decoherence_rate) * node.state + decoherence_rate * uniform
        node.state = new_state
    return g


def partial_trace(graph: CRSGraph, keep_nodes: set[int]) -> CRSGraph:
    """Extract the induced subgraph of *keep_nodes*.

    Equivalent to tracing out the complement in a density-matrix
    formulation.

    Parameters
    ----------
    graph : CRSGraph
    keep_nodes : set[int]
        Node ids to retain.

    Returns
    -------
    CRSGraph
    """
    return graph.subgraph(keep_nodes)
