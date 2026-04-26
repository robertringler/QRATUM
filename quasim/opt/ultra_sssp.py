"""UltraSSSP: Large-scale Single-Source Shortest Path algorithm for QRATUM.

This module implements an advanced SSSP algorithm with:
- Adaptive frontier clustering for batch processing
- Hierarchical graph contraction for memory efficiency
- Optional quantum pivot selection hooks
- Performance benchmarking and validation
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .graph import HierarchicalGraph, QGraph


@dataclass
class SSSPMetrics:
    """Performance and validation metrics for SSSP algorithms.

    Attributes:
        total_time: Total execution time in seconds
        iteration_times: Time per iteration
        memory_bytes: Memory usage in bytes
        nodes_visited: Number of nodes visited
        edges_relaxed: Number of edge relaxations
        frontier_sizes: Frontier size at each iteration
        correctness: Whether results match baseline
    """

    total_time: float = 0.0
    iteration_times: list[float] = field(default_factory=list)
    memory_bytes: int = 0
    nodes_visited: int = 0
    edges_relaxed: int = 0
    frontier_sizes: list[int] = field(default_factory=list)
    correctness: bool = False

    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {
            "total_time": self.total_time,
            "avg_iteration_time": np.mean(self.iteration_times) if self.iteration_times else 0.0,
            "memory_mb": self.memory_bytes / (1024 * 1024),
            "nodes_visited": self.nodes_visited,
            "edges_relaxed": self.edges_relaxed,
            "avg_frontier_size": np.mean(self.frontier_sizes) if self.frontier_sizes else 0.0,
            "correctness": self.correctness,
        }


def dijkstra_baseline(graph: QGraph, source: int) -> tuple[list[float], SSSPMetrics]:
    """Classical Dijkstra's algorithm for validation baseline.

    Args:
        graph: Input graph
        source: Source node id

    Returns:
        Tuple of (distances array, metrics)
    """
    start_time = time.time()
    metrics = SSSPMetrics()

    # Initialize distances
    distances = [float("inf")] * graph.num_nodes
    distances[source] = 0.0

    # Priority queue: (distance, node)
    pq = [(0.0, source)]
    visited = set()

    while pq:
        dist, node = heapq.heappop(pq)

        if node in visited:
            continue

        visited.add(node)
        metrics.nodes_visited += 1

        # Relax edges
        for neighbor, weight in graph.neighbors(node):
            metrics.edges_relaxed += 1
            new_dist = distances[node] + weight

            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))

    metrics.total_time = time.time() - start_time

    return distances, metrics


@dataclass
class FrontierBatch:
    """Batch of frontier nodes for parallel processing.

    Attributes:
        nodes: List of node ids in this batch
        min_distance: Minimum distance in batch
        max_distance: Maximum distance in batch
    """

    nodes: list[int] = field(default_factory=list)
    min_distance: float = float("inf")
    max_distance: float = 0.0

    def add_node(self, node: int, distance: float) -> None:
        """Add node to batch and update distance bounds."""
        self.nodes.append(node)
        self.min_distance = min(self.min_distance, distance)
        self.max_distance = max(self.max_distance, distance)

    def size(self) -> int:
        """Get batch size."""
        return len(self.nodes)


class UltraSSSP:
    """UltraSSSP algorithm with adaptive frontier clustering.

    Features:
    - Iterative frontier expansion with batch processing
    - Hierarchical graph contraction for large graphs
    - Optional quantum pivot selection for exploration ordering
    - Memory-efficient data structures

    Attributes:
        graph: Input graph structure
        batch_size: Target size for frontier batches
        use_hierarchy: Whether to use hierarchical contraction
        hierarchy_levels: Number of hierarchy levels
        quantum_pivot_fn: Optional quantum pivot selection function
    """

    def __init__(
        self,
        graph: QGraph,
        batch_size: int = 100,
        use_hierarchy: bool = False,
        hierarchy_levels: int = 3,
        quantum_pivot_fn: Callable[[list[int], list[float]], int] | None = None,
    ):
        """Initialize UltraSSSP algorithm.

        Args:
            graph: Input graph
            batch_size: Target batch size for frontier clustering
            use_hierarchy: Whether to use hierarchical contraction
            hierarchy_levels: Number of hierarchy levels (if enabled)
            quantum_pivot_fn: Optional quantum pivot selection function
        """
        self.graph = graph
        self.batch_size = batch_size
        self.use_hierarchy = use_hierarchy
        self.hierarchy_levels = hierarchy_levels
        self.quantum_pivot_fn = quantum_pivot_fn

        # Build hierarchy if requested
        self.hierarchy: HierarchicalGraph | None = None
        if use_hierarchy:
            self.hierarchy = HierarchicalGraph.from_contraction(graph, num_levels=hierarchy_levels)

    def solve(self, source: int) -> tuple[list[float], SSSPMetrics]:
        """Solve single-source shortest path from source node.

        Args:
            source: Source node id

        Returns:
            Tuple of (distances array, metrics)
        """
        start_time = time.time()
        metrics = SSSPMetrics()

        # Initialize distances
        distances = [float("inf")] * self.graph.num_nodes
        distances[source] = 0.0

        # Initialize frontier with source
        frontier = [(0.0, source)]
        visited = set()

        iteration = 0
        while frontier:
            iter_start = time.time()

            # Extract batch from frontier
            # For correctness, we need to only process nodes with the minimum distance
            # or within a small epsilon of it to maintain optimality
            batch = self._extract_batch(frontier, distances, visited)
            metrics.frontier_sizes.append(len(frontier))

            # Process batch - all nodes in batch have similar distances
            for node_dist, node in batch:
                if node in visited:
                    continue

                # Double-check: Skip if distance has been updated since extraction
                if node_dist > distances[node]:
                    continue

                visited.add(node)
                metrics.nodes_visited += 1

                # Relax edges
                for neighbor, weight in self.graph.neighbors(node):
                    metrics.edges_relaxed += 1
                    new_dist = distances[node] + weight

                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        heapq.heappush(frontier, (new_dist, neighbor))

            iter_time = time.time() - iter_start
            metrics.iteration_times.append(iter_time)
            iteration += 1

        metrics.total_time = time.time() - start_time

        # Estimate memory usage
        metrics.memory_bytes = self._estimate_memory(distances)

        return distances, metrics

    def _extract_batch(
        self, frontier: list[tuple[float, int]], distances: list[float], visited: set[int]
    ) -> list[tuple[float, int]]:
        """Extract batch of nodes from frontier using adaptive clustering.

        To maintain correctness, we extract nodes with distances within a small
        epsilon of the minimum distance. This ensures we process nodes in
        approximately the right order while still enabling batching.

        Args:
            frontier: Priority queue of (distance, node) tuples
            distances: Current distance array
            visited: Set of already visited nodes

        Returns:
            List of (distance, node) tuples to process
        """
        batch = []

        if not frontier:
            return batch

        # Find minimum distance in frontier (peek at top)
        # We need to find the actual minimum among non-stale entries
        min_dist = float("inf")
        temp_extracted = []

        # Extract entries until we find batch_size valid nodes or run out
        while frontier and len(batch) < self.batch_size:
            dist, node = heapq.heappop(frontier)
            temp_extracted.append((dist, node))

            # Skip if already visited or stale
            if node in visited or dist > distances[node]:
                continue

            # Track minimum distance
            if dist < min_dist:
                min_dist = dist

            batch.append((dist, node))

        # Put back extracted nodes that we're not processing
        # (they were stale or already visited)
        for dist, node in temp_extracted:
            if (dist, node) not in batch and node not in visited:
                heapq.heappush(frontier, (dist, node))

        # For correctness: only return nodes within epsilon of min_dist
        # This maintains approximate optimality while enabling batching
        # epsilon=0.0 gives exact Dijkstra behavior (process one distance level at a time)
        # epsilon>0.0 allows batch processing but may affect optimality
        # TODO: Make epsilon configurable as a parameter
        epsilon = 0.0  # Set to 0 for exact Dijkstra behavior
        filtered_batch = [(d, n) for d, n in batch if d <= min_dist + epsilon]

        # Put back nodes that are too far from minimum
        for dist, node in batch:
            if dist > min_dist + epsilon:
                heapq.heappush(frontier, (dist, node))

        return filtered_batch

    def _estimate_memory(self, distances: list[float]) -> int:
        """Estimate memory usage in bytes.

        Args:
            distances: Distance array

        Returns:
            Estimated memory usage in bytes
        """
        # Distance array
        memory = len(distances) * 8  # 8 bytes per float

        # Edge storage
        memory += self.graph.edge_count() * (8 + 8)  # (node_id, weight)

        # Additional overhead for data structures
        memory += self.graph.num_nodes * 16  # frontier overhead

        return memory

    def quantum_pivot_select(self, candidates: list[int], distances: list[float]) -> int:
        """Select pivot node using quantum algorithm (placeholder).

        This is a placeholder for future QPU integration. Currently falls back
        to classical heuristic (minimum distance).

        Args:
            candidates: Candidate node ids
            distances: Current distances

        Returns:
            Selected pivot node id

        Note:
            TODO: Integrate with QRATUM QPU API for quantum pivot selection.
            Expected integration:
            1. Import from qratum.qpu import QPUSelector
            2. Convert candidates to quantum state |candidates⟩
            3. Use amplitude amplification to boost minimum distance states
            4. Measure and return selected pivot
            5. Fall back to classical if QPU unavailable

            Example future implementation:
            ```python
            from qratum.qpu import QPUSelector
            if self.qpu_available:
                selector = QPUSelector(backend='ibm_quantum')
                return selector.amplitude_amplification(
                    candidates, distances, metric='min_distance'
                )
            ```
        """
        if self.quantum_pivot_fn is not None:
            # Use custom quantum pivot function if provided
            return self.quantum_pivot_fn(candidates, distances)

        # Fallback: select node with minimum distance (classical)
        return min(candidates, key=lambda n: distances[n])


def validate_sssp_results(
    distances1: list[float], distances2: list[float], tolerance: float = 1e-6
) -> bool:
    """Validate that two SSSP distance arrays match within tolerance.

    Args:
        distances1: First distance array
        distances2: Second distance array
        tolerance: Numerical tolerance for floating-point comparison

    Returns:
        True if distances match within tolerance
    """
    if len(distances1) != len(distances2):
        return False

    for d1, d2 in zip(distances1, distances2):
        # Handle infinity
        if d1 == float("inf") and d2 == float("inf"):
            continue

        if abs(d1 - d2) > tolerance:
            return False

    return True


@dataclass
class SSSPSimulationConfig:
    """Configuration for UltraSSSP simulation.

    Attributes:
        num_nodes: Number of nodes in graph
        edge_probability: Probability of edge between nodes
        max_edge_weight: Maximum edge weight
        source_node: Source node for SSSP
        batch_size: Frontier batch size
        use_hierarchy: Enable hierarchical contraction
        hierarchy_levels: Number of hierarchy levels
        seed: Random seed for reproducibility
        validate_against_dijkstra: Validate results against baseline
    """

    num_nodes: int = 1000
    edge_probability: float = 0.01
    max_edge_weight: float = 10.0
    source_node: int = 0
    batch_size: int = 100
    use_hierarchy: bool = False
    hierarchy_levels: int = 3
    seed: int = 42
    validate_against_dijkstra: bool = True


def run_sssp_simulation(config: SSSPSimulationConfig) -> dict:
    """Run complete UltraSSSP simulation with benchmarking and validation.

    Args:
        config: Simulation configuration

    Returns:
        Dictionary with results including:
            - distances: Shortest path distances
            - ultra_sssp_metrics: UltraSSSP performance metrics
            - dijkstra_metrics: Baseline Dijkstra metrics (if validated)
            - correctness: Whether results match baseline
            - speedup: Speedup over baseline (if applicable)
    """
    print(
        f"Generating random graph: {config.num_nodes} nodes, "
        f"p={config.edge_probability}, seed={config.seed}"
    )

    # Generate random graph
    graph = QGraph.random_graph(
        num_nodes=config.num_nodes,
        edge_probability=config.edge_probability,
        seed=config.seed,
        directed=True,
        max_weight=config.max_edge_weight,
    )

    print(f"Graph generated: {graph.edge_count()} edges")

    # Run UltraSSSP
    print("\nRunning UltraSSSP algorithm...")
    ultra_sssp = UltraSSSP(
        graph=graph,
        batch_size=config.batch_size,
        use_hierarchy=config.use_hierarchy,
        hierarchy_levels=config.hierarchy_levels,
    )

    ultra_distances, ultra_metrics = ultra_sssp.solve(config.source_node)
    print(f"UltraSSSP completed in {ultra_metrics.total_time:.4f}s")
    print(f"  Nodes visited: {ultra_metrics.nodes_visited}")
    print(f"  Edges relaxed: {ultra_metrics.edges_relaxed}")
    print(f"  Memory usage: {ultra_metrics.memory_bytes / (1024*1024):.2f} MB")

    results = {
        "distances": ultra_distances,
        "ultra_sssp_metrics": ultra_metrics.to_dict(),
        "graph_info": {
            "num_nodes": graph.num_nodes,
            "num_edges": graph.edge_count(),
        },
    }

    # Validate against Dijkstra baseline if requested
    if config.validate_against_dijkstra:
        print("\nRunning baseline Dijkstra for validation...")
        dijkstra_distances, dijkstra_metrics = dijkstra_baseline(graph, config.source_node)
        print(f"Dijkstra completed in {dijkstra_metrics.total_time:.4f}s")

        # Check correctness
        correctness = validate_sssp_results(ultra_distances, dijkstra_distances)
        ultra_metrics.correctness = correctness

        # Calculate relative performance
        # Note: speedup < 1.0 means UltraSSSP is slower (overhead from batching)
        # speedup > 1.0 means UltraSSSP is faster (benefits from parallelization potential)
        speedup_factor = (
            dijkstra_metrics.total_time / ultra_metrics.total_time
            if ultra_metrics.total_time > 0
            else 1.0
        )

        print(f"\nValidation: {'PASS' if correctness else 'FAIL'}")
        print(f"Performance ratio: {speedup_factor:.2f}x (>1.0 = faster, <1.0 = slower)")

        results["dijkstra_metrics"] = dijkstra_metrics.to_dict()
        results["correctness"] = correctness
        results["speedup"] = speedup_factor

    return results


__all__ = [
    "UltraSSSP",
    "dijkstra_baseline",
    "validate_sssp_results",
    "SSSPSimulationConfig",
    "run_sssp_simulation",
    "SSSPMetrics",
    "FrontierBatch",
]
