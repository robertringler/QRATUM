r"""Layer 3 — Emergent spacetime: reconstruct metric structure from graph topology.

Computes:
- Graph geodesic distances (shortest path via Dijkstra with 1/w weights)
- Causal depth (longest path from any root)
- Information propagation delay
- Emergent metric tensor approximation (from local distance structure)
- Curvature proxy (from graph density gradients)
"""

from __future__ import annotations

import heapq
from collections import deque

import numpy as np
from numpy.typing import NDArray

from quasim.ciir.crs.graph import CRSGraph

# ================================================================
# Geodesic distances
# ================================================================

def geodesic_distance(graph: CRSGraph, a: int, b: int) -> float:
    r"""Shortest-path distance from *a* to *b* using 1/w as edge length.

    Parameters
    ----------
    graph : CRSGraph
    a, b : int
        Source and target node ids.

    Returns
    -------
    float
        Geodesic distance, or ``inf`` if no path exists.
    """
    if a not in graph.nodes or b not in graph.nodes:
        return float("inf")
    if a == b:
        return 0.0

    dist: dict[int, float] = {a: 0.0}
    heap: list[tuple[float, int]] = [(0.0, a)]

    while heap:
        d, u = heapq.heappop(heap)
        if u == b:
            return d
        if d > dist.get(u, float("inf")):
            continue
        for v in graph._adjacency.get(u, set()):
            edge = graph.edges.get((u, v))
            if edge is None:
                continue
            w = edge.causal_weight
            edge_len = 1.0 / w if w > 1e-12 else 1e12
            nd = d + edge_len
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))

    return float("inf")


def distance_matrix(graph: CRSGraph) -> NDArray[np.floating]:
    r"""Pairwise geodesic distance matrix.

    Returns
    -------
    NDArray
        Shape (N, N) where N = |V|.  Entry (i, j) is the geodesic
        distance from the i-th node (in sorted-id order) to the j-th.
    """
    ids = sorted(graph.nodes)
    n = len(ids)
    D = np.full((n, n), np.inf)
    id_to_idx = {nid: idx for idx, nid in enumerate(ids)}

    for i, src in enumerate(ids):
        # Dijkstra from src
        dist: dict[int, float] = {src: 0.0}
        heap: list[tuple[float, int]] = [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, float("inf")):
                continue
            for v in graph._adjacency.get(u, set()):
                edge = graph.edges.get((u, v))
                if edge is None:
                    continue
                w = edge.causal_weight
                edge_len = 1.0 / w if w > 1e-12 else 1e12
                nd = d + edge_len
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        for nid, dd in dist.items():
            if nid in id_to_idx:
                D[i, id_to_idx[nid]] = dd

    return D


# ================================================================
# Causal depth
# ================================================================

def causal_depth(graph: CRSGraph) -> dict[int, int]:
    """Longest path from any root to each node (causal depth).

    Roots are nodes with no incoming edges within the graph.

    Returns
    -------
    dict[int, int]
        Mapping node_id → depth.
    """
    depth: dict[int, int] = dict.fromkeys(graph.nodes, 0)
    in_degree: dict[int, int] = dict.fromkeys(graph.nodes, 0)
    for nid in graph.nodes:
        for parent in graph._reverse_adj.get(nid, set()):
            if parent in graph.nodes:
                in_degree[nid] = in_degree.get(nid, 0) + 1

    queue: deque[int] = deque(
        nid for nid, deg in in_degree.items() if deg == 0
    )
    visited = 0
    while queue:
        u = queue.popleft()
        visited += 1
        for v in graph._adjacency.get(u, set()):
            if v not in graph.nodes:
                continue
            new_d = depth[u] + 1
            if new_d > depth[v]:
                depth[v] = new_d
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    # Handle cycles: assign max observed depth + 1
    if visited < len(graph.nodes):
        max_d = max(depth.values()) if depth else 0
        for nid in graph.nodes:
            if in_degree.get(nid, 0) > 0:
                depth[nid] = max_d + 1

    return depth


# ================================================================
# Emergent metric tensor
# ================================================================

def emergent_metric_tensor(
    graph: CRSGraph,
    node_id: int,
    d_embed: int = 3,
) -> NDArray[np.floating]:
    r"""Local metric approximation via MDS-like construction.

    Uses distances from *node_id* to its radius-2 neighbourhood to
    build a (d_embed × d_embed) positive-semidefinite metric tensor.

    Parameters
    ----------
    graph : CRSGraph
    node_id : int
    d_embed : int
        Embedding dimension for the local metric.

    Returns
    -------
    NDArray
        Shape (d_embed, d_embed).
    """
    nbrs = graph.get_neighborhood(node_id, radius=2) - {node_id}
    if len(nbrs) < d_embed:
        return np.eye(d_embed)

    nbr_list = sorted(nbrs)
    n = len(nbr_list)
    dists = np.zeros(n)
    for idx, nid in enumerate(nbr_list):
        dists[idx] = geodesic_distance(graph, node_id, nid)
    dists = np.where(np.isinf(dists), 1e6, dists)

    # Gram matrix from distance vector: G_ij = 0.5(d_0i^2 + d_0j^2 - d_ij^2)
    D_pair = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = geodesic_distance(graph, nbr_list[i], nbr_list[j])
            d = d if np.isfinite(d) else 1e6
            D_pair[i, j] = d
            D_pair[j, i] = d

    G = 0.5 * (
        dists[:, None] ** 2
        + dists[None, :] ** 2
        - D_pair ** 2
    )

    # Extract top-k eigenvalues → metric tensor
    eigvals, eigvecs = np.linalg.eigh(G)
    top_k = min(d_embed, len(eigvals))
    idx = np.argsort(eigvals)[::-1][:top_k]
    lam = np.maximum(eigvals[idx], 0.0)

    metric = np.zeros((d_embed, d_embed))
    for k_idx in range(top_k):
        metric[k_idx, k_idx] = lam[k_idx] if lam[k_idx] > 1e-12 else 1e-12

    return metric


# ================================================================
# Curvature proxy
# ================================================================

def curvature_proxy(graph: CRSGraph, node_id: int) -> float:
    r"""Estimate curvature from ball-volume scaling.

    Compares neighbour counts at radius 1 vs radius 2.
    Positive → sphere-like (fewer r=2 neighbours than expected).
    Negative → hyperbolic-like (more r=2 neighbours than expected).

    Returns
    -------
    float
        Curvature proxy value.
    """
    ball_1 = graph.get_neighborhood(node_id, radius=1)
    ball_2 = graph.get_neighborhood(node_id, radius=2)
    n1 = max(len(ball_1) - 1, 1)  # exclude center
    n2 = max(len(ball_2) - 1, 1)

    # In flat d-dim space, ball volume ~ r^d, so n2/n1 ~ 2^d for d=2 → ~4
    # Curvature proxy: deviation from flat expectation (ratio ~4 for 2-D)
    expected_ratio = 4.0
    actual_ratio = n2 / n1
    return expected_ratio - actual_ratio


# ================================================================
# Dimension estimate
# ================================================================

def graph_dimension_estimate(graph: CRSGraph) -> float:
    r"""Estimate effective dimension from ball-volume scaling.

    Samples several nodes and fits log(|B(r)|) ∝ d · log(r).

    Returns
    -------
    float
        Estimated graph dimension.
    """
    ids = sorted(graph.nodes)
    if len(ids) < 3:
        return 1.0

    # Sample up to 20 nodes
    step = max(1, len(ids) // 20)
    sample_ids = ids[::step]

    dims: list[float] = []
    for nid in sample_ids:
        radii = []
        volumes = []
        for r in range(1, 5):
            ball = graph.get_neighborhood(nid, radius=r)
            vol = len(ball)
            if vol > 1:
                radii.append(r)
                volumes.append(vol)
        if len(radii) >= 2:
            log_r = np.log(np.array(radii, dtype=float))
            log_v = np.log(np.array(volumes, dtype=float))
            # Linear fit: log_v = d * log_r + c
            if np.std(log_r) > 1e-12:
                slope = float(np.polyfit(log_r, log_v, 1)[0])
                dims.append(slope)

    return float(np.mean(dims)) if dims else 1.0
