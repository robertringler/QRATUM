"""QRATUM-native graph data structures for large-scale graph algorithms.

This module provides efficient graph representations optimized for QRATUM's
computational stack, including support for directed edges, weighted edges,
and hierarchical graph structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class QGraph:
    """QRATUM-native graph structure optimized for shortest path algorithms.

    Features:
    - Directed edges with non-negative weights
    - Memory-efficient adjacency list representation
    - Fast neighbor and edge iteration
    - Support for hierarchical graph contraction

    Attributes:
        num_nodes: Total number of nodes in the graph
        edges: Adjacency list mapping node_id -> [(neighbor_id, weight), ...]
        directed: Whether the graph is directed (default: True)
    """

    num_nodes: int
    edges: dict[int, list[tuple[int, float]]] = field(default_factory=dict)
    directed: bool = True

    def __post_init__(self) -> None:
        """Initialize empty adjacency lists for all nodes."""
        for i in range(self.num_nodes):
            if i not in self.edges:
                self.edges[i] = []

    def add_edge(self, src: int, dst: int, weight: float = 1.0) -> None:
        """Add a directed edge with non-negative weight.

        Args:
            src: Source node id
            dst: Destination node id
            weight: Edge weight (must be non-negative)

        Raises:
            ValueError: If weight is negative or nodes are out of range
        """
        if weight < 0:
            raise ValueError(f"Edge weight must be non-negative, got {weight}")
        if not (0 <= src < self.num_nodes and 0 <= dst < self.num_nodes):
            raise ValueError(f"Node indices must be in range [0, {self.num_nodes})")

        # Add edge from src to dst
        if src not in self.edges:
            self.edges[src] = []
        self.edges[src].append((dst, weight))

        # If undirected, add reverse edge
        if not self.directed:
            if dst not in self.edges:
                self.edges[dst] = []
            self.edges[dst].append((src, weight))

    def neighbors(self, node: int) -> Iterator[tuple[int, float]]:
        """Iterate over neighbors of a node with edge weights.

        Args:
            node: Node id

        Yields:
            Tuples of (neighbor_id, edge_weight)
        """
        if node not in self.edges:
            return
        yield from self.edges[node]

    def degree(self, node: int) -> int:
        """Get out-degree of a node.

        Args:
            node: Node id

        Returns:
            Number of outgoing edges
        """
        return len(self.edges.get(node, []))

    def edge_count(self) -> int:
        """Get total number of edges in the graph.

        Returns:
            Total edge count
        """
        return sum(len(neighbors) for neighbors in self.edges.values())

    @classmethod
    def from_edge_list(
        cls, num_nodes: int, edge_list: list[tuple[int, int, float]], directed: bool = True
    ) -> QGraph:
        """Create graph from edge list.

        Args:
            num_nodes: Total number of nodes
            edge_list: List of (src, dst, weight) tuples
            directed: Whether graph is directed

        Returns:
            QGraph instance
        """
        graph = cls(num_nodes=num_nodes, directed=directed)
        for src, dst, weight in edge_list:
            graph.add_edge(src, dst, weight)
        return graph

    @classmethod
    def random_graph(
        cls,
        num_nodes: int,
        edge_probability: float,
        seed: int = 42,
        directed: bool = True,
        max_weight: float = 10.0,
    ) -> QGraph:
        """Generate random graph with specified edge probability.

        Args:
            num_nodes: Number of nodes
            edge_probability: Probability of edge between any two nodes
            seed: Random seed for reproducibility
            directed: Whether graph is directed
            max_weight: Maximum edge weight

        Returns:
            Randomly generated QGraph
        """
        import numpy as np

        np.random.seed(seed)
        graph = cls(num_nodes=num_nodes, directed=directed)

        # Generate edges based on probability
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and np.random.random() < edge_probability:
                    weight = np.random.uniform(1.0, max_weight)
                    graph.add_edge(i, j, weight)

        return graph


@dataclass
class HierarchicalGraph:
    """Multi-level hierarchical graph structure for graph contraction.

    Supports efficient lookup of supernodes and contracted edges for
    hierarchical shortest path algorithms.

    Attributes:
        levels: List of QGraph objects, from finest to coarsest
        node_mappings: Mapping from level i node to level i+1 supernode
    """

    levels: list[QGraph] = field(default_factory=list)
    node_mappings: list[dict[int, int]] = field(default_factory=list)

    def add_level(self, graph: QGraph, mapping: dict[int, int] | None = None) -> None:
        """Add a hierarchical level.

        Args:
            graph: Graph at this level
            mapping: Node to supernode mapping (None for base level)
        """
        self.levels.append(graph)
        if mapping is not None:
            self.node_mappings.append(mapping)

    def get_supernode(self, level: int, node: int) -> int:
        """Get supernode id for a node at a given level.

        Args:
            level: Source level (0 = base graph)
            node: Node id at source level

        Returns:
            Supernode id at coarsest level
        """
        current_node = node
        for l in range(level, len(self.node_mappings)):
            current_node = self.node_mappings[l].get(current_node, current_node)
        return current_node

    @property
    def num_levels(self) -> int:
        """Get number of hierarchical levels."""
        return len(self.levels)

    @classmethod
    def from_contraction(
        cls, base_graph: QGraph, num_levels: int = 3, contraction_factor: float = 0.5
    ) -> HierarchicalGraph:
        """Create hierarchical graph by repeated contraction.

        Args:
            base_graph: Base graph at finest level
            num_levels: Number of hierarchical levels to create
            contraction_factor: Fraction of nodes to keep at each level

        Returns:
            HierarchicalGraph with multiple levels
        """

        hierarchy = cls()
        hierarchy.add_level(base_graph, mapping=None)

        current_graph = base_graph
        for _ in range(num_levels - 1):
            # Contract graph by clustering nodes
            num_supernodes = max(1, int(current_graph.num_nodes * contraction_factor))

            # Simple random clustering for now
            # Production: Use graph partitioning algorithms for better quality:
            # - METIS (multilevel k-way partitioning): pip install metis
            # - KaHIP (Karlsruhe High Quality Partitioning): pip install kahip
            # - Scotch (Static Mapping, Graph Partitioning): available in most package managers
            # - NetworkX spectral clustering: nx.spectral_clustering(G, n_clusters)
            # These provide better balanced cuts and preserve graph structure
            mapping = {}
            for node in range(current_graph.num_nodes):
                supernode = node % num_supernodes
                mapping[node] = supernode

            # Build contracted graph
            contracted_graph = QGraph(num_nodes=num_supernodes, directed=current_graph.directed)
            edge_weights: dict[tuple[int, int], float] = {}

            for src in range(current_graph.num_nodes):
                super_src = mapping[src]
                for dst, weight in current_graph.neighbors(src):
                    super_dst = mapping[dst]
                    if super_src != super_dst:  # No self-loops
                        key = (super_src, super_dst)
                        # Keep minimum weight edge between supernodes
                        # Rationale: Preserves shortest path lower bounds in contracted graph
                        # Alternative strategies (configurable in production):
                        # - Maximum weight: Preserves upper bounds
                        # - Average weight: Balances between lower/upper bounds
                        # - Sum of weights: For flow/capacity applications
                        if key not in edge_weights or weight < edge_weights[key]:
                            edge_weights[key] = weight

            for (super_src, super_dst), weight in edge_weights.items():
                contracted_graph.add_edge(super_src, super_dst, weight)

            hierarchy.add_level(contracted_graph, mapping)
            current_graph = contracted_graph

        return hierarchy


__all__ = ["QGraph", "HierarchicalGraph"]
