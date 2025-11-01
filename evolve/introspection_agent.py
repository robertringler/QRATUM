"""Runtime introspection agent for monitoring kernel performance metrics.

This module provides instrumentation hooks to log warp divergence, cache misses,
latency, and other performance metrics during kernel execution.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class PerformanceMetrics:
    """Performance metrics captured during kernel execution."""
    
    kernel_id: str
    timestamp: float
    latency_ms: float
    warp_divergence_pct: float = 0.0
    cache_miss_rate: float = 0.0
    memory_bandwidth_gb_s: float = 0.0
    compute_utilization_pct: float = 0.0
    energy_consumption_j: float = 0.0
    temperature_c: float = 0.0
    additional_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> PerformanceMetrics:
        """Create metrics from dictionary."""
        return cls(**data)


class IntrospectionAgent:
    """Agent for monitoring and logging kernel performance metrics."""
    
    def __init__(self, log_dir: Optional[Path] = None):
        """Initialize the introspection agent.
        
        Args:
            log_dir: Directory to store performance logs
        """
        self.log_dir = log_dir or Path(__file__).parent.parent / "profiles" / "introspection"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._active_traces: Dict[str, float] = {}
        self._metrics_buffer: List[PerformanceMetrics] = []
        
    def start_trace(self, kernel_id: str) -> None:
        """Start tracing a kernel execution.
        
        Args:
            kernel_id: Unique identifier for the kernel
        """
        self._active_traces[kernel_id] = time.perf_counter()
        
    def end_trace(self, kernel_id: str, **kwargs) -> PerformanceMetrics:
        """End tracing and record metrics.
        
        Args:
            kernel_id: Unique identifier for the kernel
            **kwargs: Additional performance metrics to record
            
        Returns:
            PerformanceMetrics instance with captured data
        """
        if kernel_id not in self._active_traces:
            raise ValueError(f"No active trace for kernel: {kernel_id}")
        
        start_time = self._active_traces.pop(kernel_id)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        
        # Create metrics object
        metrics = PerformanceMetrics(
            kernel_id=kernel_id,
            timestamp=time.time(),
            latency_ms=latency_ms,
            warp_divergence_pct=kwargs.get("warp_divergence_pct", 0.0),
            cache_miss_rate=kwargs.get("cache_miss_rate", 0.0),
            memory_bandwidth_gb_s=kwargs.get("memory_bandwidth_gb_s", 0.0),
            compute_utilization_pct=kwargs.get("compute_utilization_pct", 0.0),
            energy_consumption_j=kwargs.get("energy_consumption_j", 0.0),
            temperature_c=kwargs.get("temperature_c", 0.0),
            additional_metrics={k: v for k, v in kwargs.items() 
                              if k not in PerformanceMetrics.__dataclass_fields__}
        )
        
        self._metrics_buffer.append(metrics)
        return metrics
    
    def flush_logs(self, filename: Optional[str] = None) -> Path:
        """Write buffered metrics to disk.
        
        Args:
            filename: Optional filename for the log file
            
        Returns:
            Path to the written log file
        """
        if not self._metrics_buffer:
            return self.log_dir / "empty.json"
        
        if filename is None:
            timestamp = int(time.time())
            filename = f"metrics_{timestamp}.json"
        
        log_path = self.log_dir / filename
        
        metrics_data = {
            "timestamp": time.time(),
            "count": len(self._metrics_buffer),
            "metrics": [m.to_dict() for m in self._metrics_buffer]
        }
        
        with open(log_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        self._metrics_buffer.clear()
        return log_path
    
    def get_summary_statistics(self) -> Dict[str, float]:
        """Compute summary statistics from buffered metrics.
        
        Returns:
            Dictionary of aggregated statistics
        """
        if not self._metrics_buffer:
            return {}
        
        latencies = [m.latency_ms for m in self._metrics_buffer]
        divergences = [m.warp_divergence_pct for m in self._metrics_buffer]
        cache_misses = [m.cache_miss_rate for m in self._metrics_buffer]
        
        return {
            "avg_latency_ms": sum(latencies) / len(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "avg_warp_divergence_pct": sum(divergences) / len(divergences) if divergences else 0.0,
            "avg_cache_miss_rate": sum(cache_misses) / len(cache_misses) if cache_misses else 0.0,
            "total_samples": len(self._metrics_buffer)
        }
    
    def load_metrics_from_file(self, log_path: Path) -> List[PerformanceMetrics]:
        """Load metrics from a log file.
        
        Args:
            log_path: Path to the metrics log file
            
        Returns:
            List of PerformanceMetrics instances
        """
        if not log_path.exists():
            return []
        
        with open(log_path, 'r') as f:
            data = json.load(f)
        
        return [PerformanceMetrics.from_dict(m) for m in data.get("metrics", [])]


def simulate_kernel_execution(agent: IntrospectionAgent, kernel_id: str, 
                              workload_complexity: float = 1.0) -> PerformanceMetrics:
    """Simulate a kernel execution with synthetic metrics.
    
    Args:
        agent: IntrospectionAgent instance
        kernel_id: Identifier for the kernel
        workload_complexity: Complexity factor for simulation
        
    Returns:
        PerformanceMetrics captured during simulation
    """
    import random
    
    agent.start_trace(kernel_id)
    
    # Simulate some work
    time.sleep(0.001 * workload_complexity)
    
    # Generate synthetic metrics
    metrics = agent.end_trace(
        kernel_id,
        warp_divergence_pct=random.uniform(5.0, 25.0),
        cache_miss_rate=random.uniform(0.01, 0.15),
        memory_bandwidth_gb_s=random.uniform(100.0, 800.0),
        compute_utilization_pct=random.uniform(60.0, 95.0),
        energy_consumption_j=random.uniform(0.1, 2.0),
        temperature_c=random.uniform(45.0, 75.0)
    )
    
    return metrics


if __name__ == "__main__":
    # Demonstrate introspection agent usage
    print("Introspection Agent Demo")
    print("=" * 60)
    
    agent = IntrospectionAgent()
    
    # Simulate multiple kernel executions
    for i in range(5):
        kernel_id = f"test_kernel_{i}"
        metrics = simulate_kernel_execution(agent, kernel_id, workload_complexity=1.0 + i * 0.5)
        print(f"\nKernel: {kernel_id}")
        print(f"  Latency:         {metrics.latency_ms:.3f} ms")
        print(f"  Warp Divergence: {metrics.warp_divergence_pct:.2f}%")
        print(f"  Cache Miss Rate: {metrics.cache_miss_rate:.3f}")
    
    # Display summary statistics
    summary = agent.get_summary_statistics()
    print("\n" + "=" * 60)
    print("Summary Statistics:")
    for key, value in summary.items():
        print(f"  {key}: {value:.3f}")
    
    # Flush logs
    log_path = agent.flush_logs()
    print(f"\nMetrics saved to: {log_path}")
