"""Extended performance instrumentation for XENON Bioinformatics.

Provides comprehensive metrics collection:
- Memory usage tracking
- GPU utilization monitoring
- Throughput measurements
- Integration with logging framework
"""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot.
    
    Attributes:
        timestamp: Measurement timestamp
        operation: Operation being measured
        duration_ms: Operation duration in milliseconds
        memory_mb: Memory usage in MB
        gpu_util_percent: GPU utilization percentage (if available)
        throughput_ops_per_sec: Operations per second
        metadata: Additional context
    """
    
    timestamp: float = field(default_factory=time.time)
    operation: str = ""
    duration_ms: float = 0.0
    memory_mb: float = 0.0
    gpu_util_percent: float = 0.0
    throughput_ops_per_sec: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceInstrument:
    """Performance instrumentation and monitoring.
    
    Tracks execution metrics for performance analysis and optimization.
    """
    
    def __init__(self, enable_gpu: bool = True):
        """Initialize performance instrument.
        
        Args:
            enable_gpu: Whether to monitor GPU metrics
        """
        self.enable_gpu = enable_gpu
        self.metrics_history: List[PerformanceMetrics] = []
        
        # Try to import GPU monitoring libraries
        self.gpu_available = False
        if enable_gpu:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.gpu_available = True
            except (ImportError, Exception):
                warnings.warn(
                    "GPU monitoring not available (pynvml not found)",
                    UserWarning
                )
    
    def start_operation(self, operation: str) -> int:
        """Start tracking an operation.
        
        Args:
            operation: Operation name
            
        Returns:
            Operation ID for ending
        """
        metrics = PerformanceMetrics(
            operation=operation,
            timestamp=time.time(),
        )
        self.metrics_history.append(metrics)
        return len(self.metrics_history) - 1
    
    def end_operation(self, operation_id: int, metadata: Optional[Dict] = None) -> PerformanceMetrics:
        """End tracking an operation.
        
        Args:
            operation_id: ID from start_operation
            metadata: Additional context
            
        Returns:
            Final metrics
        """
        if operation_id >= len(self.metrics_history):
            raise ValueError(f"Invalid operation ID: {operation_id}")
        
        metrics = self.metrics_history[operation_id]
        end_time = time.time()
        metrics.duration_ms = (end_time - metrics.timestamp) * 1000
        
        # Memory usage
        try:
            import psutil
            process = psutil.Process(os.getpid())
            metrics.memory_mb = process.memory_info().rss / 1024 / 1024
        except ImportError:
            pass
        
        # GPU utilization
        if self.gpu_available:
            try:
                import pynvml
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                metrics.gpu_util_percent = util.gpu
            except Exception:
                pass
        
        if metadata:
            metrics.metadata = metadata
        
        return metrics
    
    def get_summary(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get performance summary.
        
        Args:
            operation: Filter by operation name
            
        Returns:
            Summary statistics
        """
        filtered = self.metrics_history
        if operation:
            filtered = [m for m in self.metrics_history if m.operation == operation]
        
        if not filtered:
            return {}
        
        durations = [m.duration_ms for m in filtered if m.duration_ms > 0]
        memories = [m.memory_mb for m in filtered if m.memory_mb > 0]
        
        summary = {
            "count": len(filtered),
            "total_duration_ms": sum(durations),
            "mean_duration_ms": np.mean(durations) if durations else 0,
            "std_duration_ms": np.std(durations) if durations else 0,
            "min_duration_ms": min(durations) if durations else 0,
            "max_duration_ms": max(durations) if durations else 0,
            "mean_memory_mb": np.mean(memories) if memories else 0,
            "max_memory_mb": max(memories) if memories else 0,
        }
        
        return summary
    
    def export_metrics(self, output_path: str) -> None:
        """Export metrics to JSON.
        
        Args:
            output_path: Output file path
        """
        import json
        
        data = {
            "metrics": [
                {
                    "timestamp": m.timestamp,
                    "operation": m.operation,
                    "duration_ms": m.duration_ms,
                    "memory_mb": m.memory_mb,
                    "gpu_util_percent": m.gpu_util_percent,
                    "throughput_ops_per_sec": m.throughput_ops_per_sec,
                    "metadata": m.metadata,
                }
                for m in self.metrics_history
            ],
            "summary": self.get_summary(),
        }
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
