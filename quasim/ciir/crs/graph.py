r"""Layer 0 — Primitive substrate: causal graph with typed nodes and weighted edges.

Implements the CRS graph G = (V, E) where:
- V = {Node_i} with state tensors s_i ∈ R^d
- E = {(i,j,w)} with causal weights w_ij ∈ [0,1]

Formal state space: S = (V × R^d) × (V×V × [0,1])
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

# ================================================================
# Primitives
# ================================================================

@dataclass(frozen=False)
class Node:
    """Graph vertex carrying a real-valued state tensor.

    Attributes
    ----------
    id : int
        Unique node identifier.
    state : NDArray[np.floating]
        State vector s ∈ R^d.
    timestamp : int
        Last-update logical clock value.
    metadata : dict
        Arbitrary key-value annotations.
    """

    id: int
    state: NDArray[np.floating]
    timestamp: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=False)
class Edge:
    """Directed causal link between two nodes.

    Attributes
    ----------
    source : int
        Origin node id.
    target : int
        Destination node id.
    causal_weight : float
        Weight w ∈ [0, 1] encoding causal influence strength.
    """

    source: int
    target: int
    causal_weight: float = 1.0

    def __post_init__(self) -> None:
        self.causal_weight = float(np.clip(self.causal_weight, 0.0, 1.0))


# ================================================================
# CRS Graph
# ================================================================

class CRSGraph:
    """Directed weighted graph implementing the CRS causal substrate.

    Maintains forward and reverse adjacency lists for efficient
    neighborhood and causal-cone queries.
    """

    def __init__(self) -> None:
        self.nodes: dict[int, Node] = {}
        self.edges: dict[tuple[int, int], Edge] = {}
        self._adjacency: dict[int, set[int]] = {}
        self._reverse_adj: dict[int, set[int]] = {}

    # ---- mutators ---------------------------------------------------

    def add_node(self, node: Node) -> None:
        """Insert *node* into the graph (overwrites if id exists)."""
        self.nodes[node.id] = node
        self._adjacency.setdefault(node.id, set())
        self._reverse_adj.setdefault(node.id, set())

    def remove_node(self, node_id: int) -> None:
        """Remove a node and all incident edges."""
        if node_id not in self.nodes:
            return
        for nbr in list(self._adjacency.get(node_id, set())):
            self.remove_edge(node_id, nbr)
        for nbr in list(self._reverse_adj.get(node_id, set())):
            self.remove_edge(nbr, node_id)
        del self.nodes[node_id]
        self._adjacency.pop(node_id, None)
        self._reverse_adj.pop(node_id, None)

    def add_edge(self, edge: Edge) -> None:
        """Insert a directed edge (overwrites if key exists)."""
        self.edges[(edge.source, edge.target)] = edge
        self._adjacency.setdefault(edge.source, set()).add(edge.target)
        self._reverse_adj.setdefault(edge.target, set()).add(edge.source)

    def remove_edge(self, src: int, tgt: int) -> None:
        """Remove the directed edge src → tgt."""
        key = (src, tgt)
        if key not in self.edges:
            return
        del self.edges[key]
        self._adjacency.get(src, set()).discard(tgt)
        self._reverse_adj.get(tgt, set()).discard(src)

    # ---- queries ----------------------------------------------------

    def get_neighborhood(self, node_id: int, radius: int = 1) -> set[int]:
        """Return node ids reachable within *radius* forward hops."""
        visited: set[int] = {node_id}
        frontier: set[int] = {node_id}
        for _ in range(radius):
            next_frontier: set[int] = set()
            for nid in frontier:
                for nbr in self._adjacency.get(nid, set()):
                    if nbr not in visited:
                        visited.add(nbr)
                        next_frontier.add(nbr)
            frontier = next_frontier
            if not frontier:
                break
        return visited

    def get_causal_cone(self, node_id: int) -> set[int]:
        """Return all ancestors of *node_id* via reverse edges (past light-cone)."""
        visited: set[int] = set()
        stack: list[int] = [node_id]
        while stack:
            nid = stack.pop()
            if nid in visited:
                continue
            visited.add(nid)
            for parent in self._reverse_adj.get(nid, set()):
                if parent not in visited:
                    stack.append(parent)
        return visited

    def subgraph(self, node_ids: set[int]) -> CRSGraph:
        """Extract an induced subgraph containing only *node_ids*."""
        sg = CRSGraph()
        for nid in node_ids:
            if nid in self.nodes:
                sg.add_node(copy.deepcopy(self.nodes[nid]))
        for (src, tgt), edge in self.edges.items():
            if src in node_ids and tgt in node_ids:
                sg.add_edge(copy.deepcopy(edge))
        return sg

    # ---- properties -------------------------------------------------

    @property
    def node_count(self) -> int:
        """Number of nodes |V|."""
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        """Number of directed edges |E|."""
        return len(self.edges)

    def density(self) -> float:
        """Edge density |E| / (|V|(|V|-1))."""
        n = self.node_count
        max_edges = n * (n - 1)
        if max_edges == 0:
            return 0.0
        return self.edge_count / max_edges

    def copy(self) -> CRSGraph:
        """Return a deep copy of this graph."""
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return f"CRSGraph(nodes={self.node_count}, edges={self.edge_count})"


# ================================================================
# Graph generators
# ================================================================

def random_graph(
    n_nodes: int,
    d_state: int,
    p_edge: float = 0.1,
    seed: int = 42,
) -> CRSGraph:
    r"""Create an Erdős–Rényi random graph G(n, p).

    Parameters
    ----------
    n_nodes : int
        Number of vertices.
    d_state : int
        Dimensionality of each node's state vector.
    p_edge : float
        Independent edge probability.
    seed : int
        Random seed.

    Returns
    -------
    CRSGraph
    """
    rng = np.random.default_rng(seed)
    g = CRSGraph()
    for i in range(n_nodes):
        g.add_node(Node(id=i, state=rng.standard_normal(d_state)))
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j and rng.random() < p_edge:
                w = float(rng.random())
                g.add_edge(Edge(source=i, target=j, causal_weight=w))
    return g


def lattice_graph(
    rows: int,
    cols: int,
    d_state: int,
    seed: int = 42,
) -> CRSGraph:
    """Create a 2-D lattice with bidirectional nearest-neighbour edges.

    Parameters
    ----------
    rows, cols : int
        Grid dimensions.
    d_state : int
        State-vector dimensionality.
    seed : int
        Random seed for initial states.

    Returns
    -------
    CRSGraph
    """
    rng = np.random.default_rng(seed)
    g = CRSGraph()

    def _idx(r: int, c: int) -> int:
        return r * cols + c

    for r in range(rows):
        for c in range(cols):
            nid = _idx(r, c)
            g.add_node(Node(id=nid, state=rng.standard_normal(d_state)))

    for r in range(rows):
        for c in range(cols):
            src = _idx(r, c)
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    tgt = _idx(nr, nc)
                    g.add_edge(Edge(source=src, target=tgt, causal_weight=1.0))
    return g
