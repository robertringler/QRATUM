"""Topological memory graph optimizer using GNN-inspired algorithms."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MemoryNode:
    """Represents a tensor or buffer in memory."""

    node_id: str
    size_bytes: int
    access_frequency: int = 0
    neighbors: set[str] = field(default_factory=set)

    def to_dict(self) -> dict:
        """Convert to dictionary."""

        return {
            "node_id": self.node_id,
            "size_bytes": self.size_bytes,
            "access_frequency": self.access_frequency,
            "neighbors": list(self.neighbors),
        }


@dataclass
class MemoryLayout:
    """Optimized memory layout."""

    layout_id: str
    node_order: list[str]
    total_path_length: float = 0.0
    cache_miss_rate: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""

        return {
            "layout_id": self.layout_id,
            "node_order": self.node_order,
            "total_path_length": self.total_path_length,
            "cache_miss_rate": self.cache_miss_rate,
        }


class MemoryGraphOptimizer:
    """

    Optimizer that represents memory allocation as a dynamic graph
    and uses GNN-inspired algorithms to predict optimal layouts.
    """

    def __init__(self):
        self.nodes: dict[str, MemoryNode] = {}
        self.layouts: dict[str, MemoryLayout] = {}

    def add_node(self, node_id: str, size_bytes: int, access_frequency: int = 1) -> None:
        """Add a memory node to the graph."""

        if node_id not in self.nodes:
            self.nodes[node_id] = MemoryNode(
                node_id=node_id,
                size_bytes=size_bytes,
                access_frequency=access_frequency,
            )
        else:
            # Update frequency if node exists
            self.nodes[node_id].access_frequency += access_frequency

    def add_edge(self, node1: str, node2: str) -> None:
        """Add an edge (access relationship) between two nodes."""

        if node1 in self.nodes and node2 in self.nodes:
            self.nodes[node1].neighbors.add(node2)
            self.nodes[node2].neighbors.add(node1)

    def compute_node_features(self, node_id: str) -> dict[str, float]:
        """

        Compute GNN-style node features.
        Aggregates information from neighbors.
        """

        if node_id not in self.nodes:
            return {}

        node = self.nodes[node_id]

        # Aggregate neighbor information
        neighbor_avg_size = 0.0
        neighbor_avg_freq = 0.0

        if node.neighbors:
            neighbor_sizes = [self.nodes[n].size_bytes for n in node.neighbors if n in self.nodes]
            neighbor_freqs = [
                self.nodes[n].access_frequency for n in node.neighbors if n in self.nodes
            ]

            if neighbor_sizes:
                neighbor_avg_size = sum(neighbor_sizes) / len(neighbor_sizes)
            if neighbor_freqs:
                neighbor_avg_freq = sum(neighbor_freqs) / len(neighbor_freqs)

        return {
            "size_bytes": float(node.size_bytes),
            "access_frequency": float(node.access_frequency),
            "degree": float(len(node.neighbors)),
            "neighbor_avg_size": neighbor_avg_size,
            "neighbor_avg_freq": neighbor_avg_freq,
        }

    def compute_path_length(self, node_order: list[str]) -> float:
        """

        Compute total path length for a given node ordering.
        Lower is better (nodes accessed together are closer).
        """

        if len(node_order) < 2:
            return 0.0

        total_length = 0.0

        # Build position map
        position = {node_id: i for i, node_id in enumerate(node_order)}

        # For each edge, compute distance in the layout
        for node_id in node_order:
            if node_id not in self.nodes:
                continue

            node = self.nodes[node_id]
            for neighbor_id in node.neighbors:
                if neighbor_id in position:
                    # Distance weighted by access frequency
                    distance = abs(position[node_id] - position[neighbor_id])
                    weight = node.access_frequency + self.nodes[neighbor_id].access_frequency
                    total_length += distance * weight

        return total_length / 2.0  # Divide by 2 to avoid double counting edges

    def estimate_cache_miss_rate(self, node_order: list[str]) -> float:
        """

        Estimate cache miss rate for a given layout.
        Simplified model based on access patterns.
        """

        if not node_order:
            return 1.0

        # Simulate cache with limited capacity
        cache_lines = 64  # Simplified cache size
        cache = []
        misses = 0
        accesses = 0

        # Simulate sequential access pattern
        for node_id in node_order:
            if node_id not in self.nodes:
                continue

            node = self.nodes[node_id]

            for _ in range(node.access_frequency):
                accesses += 1

                if node_id not in cache:
                    misses += 1
                    cache.append(node_id)

                    # Evict if cache full
                    if len(cache) > cache_lines:
                        cache.pop(0)

        return misses / accesses if accesses > 0 else 1.0

    def optimize_layout(self, layout_id: str) -> MemoryLayout:
        """

        Optimize memory layout using greedy heuristic.
        Places frequently accessed and connected nodes close together.
        """

        if not self.nodes:
            return MemoryLayout(layout_id=layout_id, node_order=[])

        # Score nodes by importance (frequency + degree)
        node_scores = {}
        for node_id, _node in self.nodes.items():
            features = self.compute_node_features(node_id)
            score = (
                features["access_frequency"] * 0.5
                + features["degree"] * 0.3
                + features["neighbor_avg_freq"] * 0.2
            )
            node_scores[node_id] = score

        # Sort nodes by score (descending)
        sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)

        # Build layout using greedy placement
        node_order = []
        placed = set()

        # Start with highest-scored node
        if sorted_nodes:
            first_node = sorted_nodes[0][0]
            node_order.append(first_node)
            placed.add(first_node)

            # Greedily add neighbors
            while len(placed) < len(self.nodes):
                best_next = None
                best_score = -1

                # Find best unplaced neighbor of any placed node
                for placed_node in placed:
                    for neighbor in self.nodes[placed_node].neighbors:
                        if neighbor not in placed and neighbor in node_scores:
                            if node_scores[neighbor] > best_score:
                                best_score = node_scores[neighbor]
                                best_next = neighbor

                # If no neighbors found, pick highest-scored unplaced node
                if best_next is None:
                    for node_id, score in sorted_nodes:
                        if node_id not in placed:
                            best_next = node_id
                            break

                if best_next:
                    node_order.append(best_next)
                    placed.add(best_next)
                else:
                    break

        # Compute metrics for the layout
        path_length = self.compute_path_length(node_order)
        cache_miss_rate = self.estimate_cache_miss_rate(node_order)

        layout = MemoryLayout(
            layout_id=layout_id,
            node_order=node_order,
            total_path_length=path_length,
            cache_miss_rate=cache_miss_rate,
        )

        self.layouts[layout_id] = layout
        return layout

    def save_layout(self, layout_id: str, output_dir: str = "memgraph") -> Path:
        """Save memory layout to disk."""

        if layout_id not in self.layouts:
            raise ValueError(f"No layout found for {layout_id}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        layout_file = output_path / f"{layout_id}.json"
        layout = self.layouts[layout_id]

        # Include node information
        data = layout.to_dict()
        data["nodes"] = {
            node_id: self.nodes[node_id].to_dict()
            for node_id in layout.node_order
            if node_id in self.nodes
        }

        with open(layout_file, "w") as f:
            json.dump(data, f, indent=2)

        return layout_file
