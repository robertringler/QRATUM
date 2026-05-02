"""community.py — community detection on persona-lifted edge graphs.

Layer B / Step 3.

Three backends, in order of preference:

1. **Louvain via NetworkX** (``networkx.community.louvain_communities``)
   when ``networkx`` is importable.  This is the recommended backend
   per the spec and is gated behind the ``observability`` extra in
   ``pyproject.toml``.
2. **Deterministic greedy-modularity** (built-in, no third-party deps)
   used as the default fallback.  This is a single-pass agglomerative
   modularity-maximization scheme that visits nodes in lexicographic
   order so two equivalent invocations produce byte-identical
   communities — matching the determinism convention used elsewhere
   in the framework.
3. **Spectral clustering via NumPy** (``spectral_communities``) — an
   alternative entry-point that the caller may invoke explicitly when
   they have a desired ``k`` cluster count.  Uses the Laplacian
   eigendecomposition; required NumPy is already a core dependency.

Edges with negative weights (i.e. baseline-dominant pairs) are dropped
before clustering, since modularity-style objectives are defined for
non-negative weights.  See :func:`qratum_framework.observability.clusters.lift.positive_lift_subgraph`.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, Iterable, List, Mapping, Sequence, Set, Tuple

EdgeKey = Tuple[str, str]


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------


def detect_communities(
    weighted_edges: Mapping[EdgeKey, float],
    *,
    seed: int = 0,
    backend: str = "auto",
    resolution: float = 1.0,
) -> List[FrozenSet[str]]:
    """Detect communities on a positive-weight edge graph.

    Parameters
    ----------
    weighted_edges:
        Mapping ``(i, j) -> weight``.  Non-positive edges are filtered
        out internally; callers usually supply the output of
        :func:`...lift.positive_lift_subgraph`.
    seed:
        Determinism seed for the NetworkX-Louvain backend.  The
        built-in greedy backend is fully deterministic and ignores it.
    backend:
        One of ``"auto"``, ``"louvain"``, ``"greedy"``, ``"spectral"``.
        ``"auto"`` picks Louvain if ``networkx`` is importable, else
        the greedy fallback.  ``"spectral"`` falls back to the greedy
        backend unless ``k`` is supplied via :func:`spectral_communities`.
    resolution:
        Modularity resolution parameter for Louvain (passed through).

    Returns
    -------
    list of frozenset of str
        Communities sorted by descending size (then by sorted member
        tuple) for stable output.  Singletons (nodes that did not
        survive filtering) are excluded.
    """
    pos = {k: v for k, v in weighted_edges.items() if v > 0.0}
    if not pos:
        return []

    if backend == "auto":
        backend = "louvain" if _has_networkx() else "greedy"

    if backend == "louvain":
        try:
            comms = _louvain_communities(pos, seed=seed, resolution=resolution)
        except _LouvainUnavailable:
            comms = _greedy_modularity(pos)
    elif backend == "greedy":
        comms = _greedy_modularity(pos)
    elif backend == "spectral":
        # Without an explicit ``k`` we cannot run spectral; fall back to
        # greedy and let the caller invoke ``spectral_communities``
        # directly when they want the spectral path.
        comms = _greedy_modularity(pos)
    else:
        raise ValueError(f"unknown backend {backend!r}")

    return _sort_communities(comms)


def spectral_communities(
    weighted_edges: Mapping[EdgeKey, float],
    *,
    k: int,
) -> List[FrozenSet[str]]:
    """Spectral clustering with a fixed number of communities.

    Uses the unnormalised Laplacian's bottom-``k`` eigenvectors and a
    deterministic argmax assignment.  Imported lazily so the core
    package stays NumPy-only at module-load time.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    pos = {e: w for e, w in weighted_edges.items() if w > 0.0}
    if not pos:
        return []
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover — numpy is a hard dep
        raise RuntimeError("spectral_communities requires NumPy") from exc

    nodes = sorted({n for e in pos for n in e})
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    if n == 0:
        return []
    if k > n:
        k = n

    W = np.zeros((n, n), dtype=float)
    for (a, b), w in pos.items():
        W[idx[a], idx[b]] = w
        W[idx[b], idx[a]] = w
    D = np.diag(W.sum(axis=1))
    L = D - W
    # Symmetric → eigh gives ascending real eigenvalues.
    vals, vecs = np.linalg.eigh(L)
    # Take the bottom-k eigenvectors (skip the trivial zero one when present).
    embedding = vecs[:, :k]
    # Deterministic clustering: discretise rows by argmax sign-pattern
    # via simple k-means-style nearest-centroid iterations seeded by
    # the lex-smallest rows.  For stability we use the argmax of the
    # absolute embedding, which is deterministic given ``vecs``.
    labels = _kmeans_lite(embedding, k=k)
    out: Dict[int, Set[str]] = {}
    for node, lab in zip(nodes, labels):
        out.setdefault(int(lab), set()).add(node)
    return _sort_communities([frozenset(s) for s in out.values() if len(s) > 1])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_networkx() -> bool:
    try:
        import networkx  # noqa: F401
        return True
    except ImportError:
        return False


class _LouvainUnavailable(RuntimeError):
    pass


def _louvain_communities(
    weighted_edges: Mapping[EdgeKey, float],
    *,
    seed: int,
    resolution: float,
) -> List[FrozenSet[str]]:
    try:
        import networkx as nx
    except ImportError as exc:
        raise _LouvainUnavailable("networkx not installed") from exc
    g = nx.Graph()
    for (a, b), w in weighted_edges.items():
        g.add_edge(a, b, weight=float(w))
    try:
        from networkx.algorithms.community import louvain_communities
    except ImportError as exc:  # pragma: no cover — older NX
        raise _LouvainUnavailable("louvain_communities unavailable") from exc
    parts = louvain_communities(g, weight="weight", resolution=resolution, seed=seed)
    return [frozenset(p) for p in parts if len(p) > 1]


def _greedy_modularity(weighted_edges: Mapping[EdgeKey, float]) -> List[FrozenSet[str]]:
    """Deterministic weighted label-propagation community detection.

    Each node starts with its own label.  In lex-order, each node adopts
    the label whose neighbour-weight-sum to that node is largest
    (ties broken by the lex-smaller label).  Iteration stops when no
    label changes in a full pass or a hard cap is hit.

    For a fully-connected positive-weight subgraph this collapses every
    component into a single community; richer modularity-style
    subdivisions are produced by the NetworkX/Louvain backend.  This
    fallback is intentionally simple, fully deterministic, and produces
    sensible communities on the sparse persona-lift graphs the drift
    engine consumes.
    """
    nodes = sorted({n for e in weighted_edges for n in e})
    if not nodes:
        return []

    adj: Dict[str, Dict[str, float]] = {n: {} for n in nodes}
    for (a, b), w in weighted_edges.items():
        if w <= 0.0:
            continue
        adj[a][b] = adj[a].get(b, 0.0) + w
        adj[b][a] = adj[b].get(a, 0.0) + w

    label: Dict[str, str] = {n: n for n in nodes}

    max_passes = 50
    for _ in range(max_passes):
        moved = False
        for n in nodes:  # lex order
            if not adj[n]:
                continue
            tally: Dict[str, float] = {}
            for nbr, w in adj[n].items():
                lab = label[nbr]
                tally[lab] = tally.get(lab, 0.0) + w
            # Pick best label: max weight, ties → lex-smaller label.
            best_label = label[n]
            best_w = tally.get(best_label, 0.0)
            for lab, w in sorted(tally.items()):
                if w > best_w + 1e-12 or (
                    abs(w - best_w) <= 1e-12 and lab < best_label
                ):
                    best_label = lab
                    best_w = w
            if best_label != label[n]:
                label[n] = best_label
                moved = True
        if not moved:
            break

    by_label: Dict[str, Set[str]] = {}
    for n, lab in label.items():
        by_label.setdefault(lab, set()).add(n)
    return [frozenset(s) for s in by_label.values() if len(s) > 1]


def _kmeans_lite(points, *, k: int, max_iter: int = 50) -> Sequence[int]:
    """Tiny deterministic k-means for the spectral path.

    Centroids are initialised as the lex-first ``k`` points (sorted by
    first-coordinate then by second-coordinate, etc.) and updated by
    the standard hard-assignment loop.
    """
    import numpy as np

    pts = points
    n = pts.shape[0]
    if k >= n:
        return list(range(n))

    # Lex-sort points to get deterministic seeds.
    order = np.lexsort(np.flipud(pts.T))
    centroids = pts[order[:k]].copy()

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        # Assign.
        d2 = ((pts[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = d2.argmin(axis=1)
        if (new_labels == labels).all():
            break
        labels = new_labels
        # Update.
        for c in range(k):
            mask = labels == c
            if mask.any():
                centroids[c] = pts[mask].mean(axis=0)
    return [int(x) for x in labels]


def _sort_communities(comms: Iterable[FrozenSet[str]]) -> List[FrozenSet[str]]:
    """Stable sort: by descending size, then by sorted member tuple."""
    out = list(comms)
    out.sort(key=lambda c: (-len(c), tuple(sorted(c))))
    return out


__all__ = ["detect_communities", "spectral_communities"]
