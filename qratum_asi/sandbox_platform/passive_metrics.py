"""Passive Metrics Collection for Sandbox Observability.

Implements passive read-only metrics collection including latency,
throughput, entropy, and topological indices without impacting performance.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from qradle.merkle import MerkleChain


class MetricType(Enum):
    """Type of metric."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class LatencyMetrics:
    """Latency metrics for sandbox operations.

    Attributes:
        metric_id: Unique metric identifier
        operation: Operation being measured
        min_ms: Minimum latency
        max_ms: Maximum latency
        avg_ms: Average latency
        p50_ms: 50th percentile latency
        p95_ms: 95th percentile latency
        p99_ms: 99th percentile latency
        sample_count: Number of samples
    """

    metric_id: str
    operation: str
    min_ms: float = float("inf")
    max_ms: float = 0.0
    avg_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    sample_count: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize latency metrics."""
        return {
            "metric_id": self.metric_id,
            "operation": self.operation,
            "min_ms": self.min_ms if self.min_ms != float("inf") else 0,
            "max_ms": self.max_ms,
            "avg_ms": self.avg_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "sample_count": self.sample_count,
            "timestamp": self.timestamp,
        }


@dataclass
class ThroughputMetrics:
    """Throughput metrics for sandbox operations.

    Attributes:
        metric_id: Unique metric identifier
        operation: Operation being measured
        ops_per_second: Operations per second
        bytes_per_second: Bytes processed per second
        peak_ops_per_second: Peak OPS observed
        total_operations: Total operations processed
        window_seconds: Measurement window
    """

    metric_id: str
    operation: str
    ops_per_second: float = 0.0
    bytes_per_second: float = 0.0
    peak_ops_per_second: float = 0.0
    total_operations: int = 0
    window_seconds: float = 60.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize throughput metrics."""
        return {
            "metric_id": self.metric_id,
            "operation": self.operation,
            "ops_per_second": self.ops_per_second,
            "bytes_per_second": self.bytes_per_second,
            "peak_ops_per_second": self.peak_ops_per_second,
            "total_operations": self.total_operations,
            "window_seconds": self.window_seconds,
            "timestamp": self.timestamp,
        }


@dataclass
class EntropyMetrics:
    """Entropy metrics for sandbox state.

    Attributes:
        metric_id: Unique metric identifier
        source: Source of entropy measurement
        shannon_entropy: Shannon entropy of state
        normalized_entropy: Normalized entropy (0-1)
        state_complexity: Estimated state complexity
        information_content: Information content in bits
    """

    metric_id: str
    source: str
    shannon_entropy: float = 0.0
    normalized_entropy: float = 0.0
    state_complexity: float = 0.0
    information_content: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize entropy metrics."""
        return {
            "metric_id": self.metric_id,
            "source": self.source,
            "shannon_entropy": self.shannon_entropy,
            "normalized_entropy": self.normalized_entropy,
            "state_complexity": self.state_complexity,
            "information_content": self.information_content,
            "timestamp": self.timestamp,
        }


@dataclass
class TopologicalIndices:
    """Topological indices for sandbox structure.

    Attributes:
        metric_id: Unique metric identifier
        source: Source of topology measurement
        node_count: Number of nodes in structure
        edge_count: Number of edges/connections
        density: Graph density
        clustering_coefficient: Clustering coefficient
        diameter: Graph diameter
        connectivity: Connectivity measure
    """

    metric_id: str
    source: str
    node_count: int = 0
    edge_count: int = 0
    density: float = 0.0
    clustering_coefficient: float = 0.0
    diameter: int = 0
    connectivity: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize topological indices."""
        return {
            "metric_id": self.metric_id,
            "source": self.source,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "density": self.density,
            "clustering_coefficient": self.clustering_coefficient,
            "diameter": self.diameter,
            "connectivity": self.connectivity,
            "timestamp": self.timestamp,
        }


class PassiveMetricsCollector:
    """Passive metrics collector for sandbox observability.

    Provides:
    - Read-only metrics collection
    - No impact on production paths
    - Automatic aggregation and windowing
    - Multiple metric types (latency, throughput, entropy, topology)
    """

    def __init__(
        self,
        collector_id: str = "passive_metrics",
        window_seconds: float = 60.0,
        max_samples: int = 10000,
        merkle_chain: MerkleChain | None = None,
    ):
        """Initialize passive metrics collector.

        Args:
            collector_id: Unique collector identifier
            window_seconds: Metrics window in seconds
            max_samples: Maximum samples to retain
            merkle_chain: Merkle chain for audit trail
        """
        self.collector_id = collector_id
        self.window_seconds = window_seconds
        self.max_samples = max_samples
        self.merkle_chain = merkle_chain or MerkleChain()

        # Latency samples per operation
        self._latency_samples: dict[str, deque[tuple[float, float]]] = {}

        # Throughput tracking
        self._throughput_counters: dict[str, list[tuple[float, int]]] = {}
        self._throughput_bytes: dict[str, int] = {}

        # Entropy measurements
        self._entropy_history: deque[EntropyMetrics] = deque(maxlen=1000)

        # Topology measurements
        self._topology_history: deque[TopologicalIndices] = deque(maxlen=1000)

        # Counters
        self._metric_counter = 0
        self._lock = threading.RLock()

        # Log initialization
        self.merkle_chain.add_event(
            "passive_metrics_collector_initialized",
            {
                "collector_id": collector_id,
                "window_seconds": window_seconds,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    def record_latency(
        self,
        operation: str,
        latency_ms: float,
    ) -> None:
        """Record a latency sample.

        Args:
            operation: Operation being measured
            latency_ms: Latency in milliseconds
        """
        with self._lock:
            if operation not in self._latency_samples:
                self._latency_samples[operation] = deque(maxlen=self.max_samples)

            timestamp = time.time()
            self._latency_samples[operation].append((timestamp, latency_ms))

            # Clean old samples
            self._clean_old_samples(operation)

    def record_operation(
        self,
        operation: str,
        count: int = 1,
        bytes_processed: int = 0,
    ) -> None:
        """Record an operation for throughput tracking.

        Args:
            operation: Operation being measured
            count: Number of operations
            bytes_processed: Bytes processed
        """
        with self._lock:
            if operation not in self._throughput_counters:
                self._throughput_counters[operation] = []
                self._throughput_bytes[operation] = 0

            timestamp = time.time()
            self._throughput_counters[operation].append((timestamp, count))
            self._throughput_bytes[operation] += bytes_processed

            # Clean old counters
            self._clean_old_counters(operation)

    def record_entropy(
        self,
        source: str,
        state_data: dict[str, Any] | list[Any] | str,
    ) -> EntropyMetrics:
        """Record entropy measurement.

        Args:
            source: Source of measurement
            state_data: Data to measure entropy of

        Returns:
            EntropyMetrics for the measurement
        """
        self._metric_counter += 1
        metric_id = f"entropy_{self.collector_id}_{self._metric_counter:08d}"

        # Compute entropy
        entropy_value, normalized, complexity, info_content = self._compute_entropy(state_data)

        metrics = EntropyMetrics(
            metric_id=metric_id,
            source=source,
            shannon_entropy=entropy_value,
            normalized_entropy=normalized,
            state_complexity=complexity,
            information_content=info_content,
        )

        with self._lock:
            self._entropy_history.append(metrics)

        return metrics

    def record_topology(
        self,
        source: str,
        nodes: list[str],
        edges: list[tuple[str, str]],
    ) -> TopologicalIndices:
        """Record topology measurement.

        Args:
            source: Source of measurement
            nodes: List of node identifiers
            edges: List of (source, target) edges

        Returns:
            TopologicalIndices for the measurement
        """
        self._metric_counter += 1
        metric_id = f"topology_{self.collector_id}_{self._metric_counter:08d}"

        # Compute topological indices
        indices = self._compute_topology_indices(nodes, edges)

        metrics = TopologicalIndices(
            metric_id=metric_id,
            source=source,
            node_count=len(nodes),
            edge_count=len(edges),
            density=indices["density"],
            clustering_coefficient=indices["clustering"],
            diameter=indices["diameter"],
            connectivity=indices["connectivity"],
        )

        with self._lock:
            self._topology_history.append(metrics)

        return metrics

    def get_latency_metrics(self, operation: str) -> LatencyMetrics:
        """Get latency metrics for an operation.

        Args:
            operation: Operation to get metrics for

        Returns:
            LatencyMetrics for the operation
        """
        self._metric_counter += 1
        metric_id = f"latency_{self.collector_id}_{self._metric_counter:08d}"

        with self._lock:
            if operation not in self._latency_samples:
                return LatencyMetrics(
                    metric_id=metric_id,
                    operation=operation,
                )

            samples = [s[1] for s in self._latency_samples[operation]]
            if not samples:
                return LatencyMetrics(
                    metric_id=metric_id,
                    operation=operation,
                )

            samples_sorted = sorted(samples)
            count = len(samples)

            return LatencyMetrics(
                metric_id=metric_id,
                operation=operation,
                min_ms=min(samples),
                max_ms=max(samples),
                avg_ms=sum(samples) / count,
                p50_ms=samples_sorted[int(count * 0.5)],
                p95_ms=samples_sorted[int(count * 0.95)] if count >= 20 else samples_sorted[-1],
                p99_ms=samples_sorted[int(count * 0.99)] if count >= 100 else samples_sorted[-1],
                sample_count=count,
            )

    def get_throughput_metrics(self, operation: str) -> ThroughputMetrics:
        """Get throughput metrics for an operation.

        Args:
            operation: Operation to get metrics for

        Returns:
            ThroughputMetrics for the operation
        """
        self._metric_counter += 1
        metric_id = f"throughput_{self.collector_id}_{self._metric_counter:08d}"

        with self._lock:
            if operation not in self._throughput_counters:
                return ThroughputMetrics(
                    metric_id=metric_id,
                    operation=operation,
                    window_seconds=self.window_seconds,
                )

            counters = self._throughput_counters[operation]
            if not counters:
                return ThroughputMetrics(
                    metric_id=metric_id,
                    operation=operation,
                    window_seconds=self.window_seconds,
                )

            current_time = time.time()
            window_start = current_time - self.window_seconds

            # Sum operations in window
            total_ops = sum(c[1] for c in counters if c[0] >= window_start)
            total_bytes = self._throughput_bytes.get(operation, 0)

            ops_per_second = total_ops / self.window_seconds if self.window_seconds > 0 else 0
            bytes_per_second = total_bytes / self.window_seconds if self.window_seconds > 0 else 0

            return ThroughputMetrics(
                metric_id=metric_id,
                operation=operation,
                ops_per_second=ops_per_second,
                bytes_per_second=bytes_per_second,
                peak_ops_per_second=ops_per_second,  # Simplified
                total_operations=sum(c[1] for c in counters),
                window_seconds=self.window_seconds,
            )

    def _clean_old_samples(self, operation: str) -> None:
        """Clean samples outside the window."""
        cutoff = time.time() - self.window_seconds
        samples = self._latency_samples[operation]
        while samples and samples[0][0] < cutoff:
            samples.popleft()

    def _clean_old_counters(self, operation: str) -> None:
        """Clean counters outside the window."""
        cutoff = time.time() - self.window_seconds
        counters = self._throughput_counters[operation]
        self._throughput_counters[operation] = [c for c in counters if c[0] >= cutoff]

    def _compute_entropy(
        self,
        data: dict[str, Any] | list[Any] | str,
    ) -> tuple[float, float, float, float]:
        """Compute entropy metrics for data."""
        # Convert to string for analysis
        if isinstance(data, dict):
            data_str = str(data)
        elif isinstance(data, list):
            data_str = str(data)
        else:
            data_str = str(data)

        # Compute character frequency
        freq: dict[str, int] = {}
        for char in data_str:
            freq[char] = freq.get(char, 0) + 1

        total = len(data_str)
        if total == 0:
            return 0.0, 0.0, 0.0, 0.0

        # Shannon entropy
        entropy = 0.0
        for count in freq.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        # Normalized entropy (max is log2 of unique chars)
        max_entropy = math.log2(len(freq)) if len(freq) > 1 else 1
        normalized = entropy / max_entropy if max_entropy > 0 else 0

        # State complexity (approximated by unique elements / total)
        complexity = len(freq) / total

        # Information content in bits
        info_content = entropy * total

        return entropy, normalized, complexity, info_content

    def _compute_topology_indices(
        self,
        nodes: list[str],
        edges: list[tuple[str, str]],
    ) -> dict[str, float]:
        """Compute topological indices for a graph."""
        n = len(nodes)
        m = len(edges)

        if n == 0:
            return {"density": 0, "clustering": 0, "diameter": 0, "connectivity": 0}

        # Density
        max_edges = n * (n - 1) / 2 if n > 1 else 1
        density = m / max_edges if max_edges > 0 else 0

        # Build adjacency for clustering
        adj: dict[str, set[str]] = {node: set() for node in nodes}
        for src, dst in edges:
            if src in adj and dst in adj:
                adj[src].add(dst)
                adj[dst].add(src)

        # Clustering coefficient (simplified)
        clustering = 0.0
        for node in nodes:
            neighbors = adj[node]
            k = len(neighbors)
            if k >= 2:
                # Count edges between neighbors
                neighbor_edges = sum(
                    1 for n1 in neighbors for n2 in neighbors
                    if n1 < n2 and n2 in adj[n1]
                )
                max_neighbor_edges = k * (k - 1) / 2
                if max_neighbor_edges > 0:
                    clustering += neighbor_edges / max_neighbor_edges
        clustering = clustering / n if n > 0 else 0

        # Diameter (simplified - just estimate)
        diameter = int(math.log2(n + 1)) if n > 0 else 0

        # Connectivity (fraction of nodes with edges)
        connected = sum(1 for node in nodes if adj[node])
        connectivity = connected / n if n > 0 else 0

        return {
            "density": density,
            "clustering": clustering,
            "diameter": diameter,
            "connectivity": connectivity,
        }

    def get_all_metrics(self) -> dict[str, Any]:
        """Get all collected metrics.

        Returns:
            Dictionary of all metrics
        """
        with self._lock:
            latency_metrics = {
                op: self.get_latency_metrics(op).to_dict()
                for op in self._latency_samples
            }
            throughput_metrics = {
                op: self.get_throughput_metrics(op).to_dict()
                for op in self._throughput_counters
            }

            return {
                "collector_id": self.collector_id,
                "window_seconds": self.window_seconds,
                "latency": latency_metrics,
                "throughput": throughput_metrics,
                "entropy_history_size": len(self._entropy_history),
                "topology_history_size": len(self._topology_history),
                "latest_entropy": (
                    self._entropy_history[-1].to_dict()
                    if self._entropy_history
                    else None
                ),
                "latest_topology": (
                    self._topology_history[-1].to_dict()
                    if self._topology_history
                    else None
                ),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    def get_collector_stats(self) -> dict[str, Any]:
        """Get collector statistics."""
        return {
            "collector_id": self.collector_id,
            "window_seconds": self.window_seconds,
            "max_samples": self.max_samples,
            "operations_tracked": len(self._latency_samples),
            "total_metrics_generated": self._metric_counter,
            "entropy_measurements": len(self._entropy_history),
            "topology_measurements": len(self._topology_history),
        }
