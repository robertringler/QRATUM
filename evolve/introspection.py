"""Runtime introspection agent for kernel metrics collection."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class KernelMetrics:
    """Metrics collected from kernel execution."""

    kernel_id: str
    timestamp: float = field(default_factory=time.time)
    warp_divergence: float = 0.0
    cache_misses: int = 0
    latency_ms: float = 0.0
    memory_bandwidth_gbps: float = 0.0
    compute_utilization: float = 0.0
    tile_size: int = 0
    warp_count: int = 0
    unroll_factor: int = 1
    async_depth: int = 1
    precision: str = "fp32"
    energy_joules: float = 0.0

    def to_dict(self) -> dict:
        """Convert metrics to dictionary for serialization."""
        return {
            "kernel_id": self.kernel_id,
            "timestamp": self.timestamp,
            "warp_divergence": self.warp_divergence,
            "cache_misses": self.cache_misses,
            "latency_ms": self.latency_ms,
            "memory_bandwidth_gbps": self.memory_bandwidth_gbps,
            "compute_utilization": self.compute_utilization,
            "tile_size": self.tile_size,
            "warp_count": self.warp_count,
            "unroll_factor": self.unroll_factor,
            "async_depth": self.async_depth,
            "precision": self.precision,
            "energy_joules": self.energy_joules,
        }


class IntrospectionAgent:
    """Agent that logs kernel execution metrics for RL optimization."""

    def __init__(self, log_dir: str = "evolve/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history: List[KernelMetrics] = []
        self._session_id = int(time.time() * 1000)

    def record_metrics(self, metrics: KernelMetrics) -> None:
        """Record kernel execution metrics."""
        self.metrics_history.append(metrics)

    def flush_to_disk(self) -> Path:
        """Write accumulated metrics to disk."""
        output_path = self.log_dir / f"metrics_{self._session_id}.json"
        data = [m.to_dict() for m in self.metrics_history]
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        return output_path

    def get_recent_metrics(self, n: int = 100) -> List[KernelMetrics]:
        """Get the n most recent metrics."""
        return self.metrics_history[-n:]

    def compute_statistics(self) -> Dict[str, float]:
        """Compute aggregate statistics from collected metrics."""
        if not self.metrics_history:
            return {}

        latencies = [m.latency_ms for m in self.metrics_history]
        energies = [m.energy_joules for m in self.metrics_history]

        return {
            "avg_latency_ms": sum(latencies) / len(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "total_energy_j": sum(energies),
            "avg_energy_j": sum(energies) / len(energies) if energies else 0.0,
            "total_kernels": len(self.metrics_history),
        }
