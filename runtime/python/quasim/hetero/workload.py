"""Workload characterization for heterogeneous scheduling."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class WorkloadType(Enum):
    """Types of computational workloads."""
    DENSE_LINEAR_ALGEBRA = "dense_linear_algebra"
    SPARSE_LINEAR_ALGEBRA = "sparse_linear_algebra"
    FFT = "fft"
    QUANTUM_SIMULATION = "quantum_simulation"
    TENSOR_CONTRACTION = "tensor_contraction"
    MEMORY_BOUND = "memory_bound"
    COMPUTE_BOUND = "compute_bound"


@dataclass
class Workload:
    """Represents a computational workload."""
    name: str
    workload_type: WorkloadType
    size_gflops: float
    memory_footprint_gb: float
    arithmetic_intensity: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}
            
    @property
    def is_memory_bound(self) -> bool:
        """Check if workload is memory-bound."""
        # Arithmetic intensity < 10 typically indicates memory-bound
        return self.arithmetic_intensity < 10.0
        
    @property
    def is_compute_bound(self) -> bool:
        """Check if workload is compute-bound."""
        return self.arithmetic_intensity >= 10.0
        
    def estimate_cpu_suitability(self) -> float:
        """Estimate how suitable this workload is for CPU (0-1)."""
        # Small workloads or memory-bound workloads suit CPU better
        size_factor = min(1.0, 100.0 / max(self.size_gflops, 1.0))
        memory_factor = 1.0 if self.is_memory_bound else 0.3
        return (size_factor + memory_factor) / 2.0
        
    def estimate_gpu_suitability(self) -> float:
        """Estimate how suitable this workload is for GPU (0-1)."""
        # Large compute-bound workloads suit GPU better
        size_factor = min(1.0, self.size_gflops / 100.0)
        compute_factor = 1.0 if self.is_compute_bound else 0.4
        return (size_factor + compute_factor) / 2.0
        
    def estimate_tpu_suitability(self) -> float:
        """Estimate how suitable this workload is for TPU (0-1)."""
        # Matrix operations and tensor contractions suit TPU
        if self.workload_type in (
            WorkloadType.DENSE_LINEAR_ALGEBRA,
            WorkloadType.TENSOR_CONTRACTION,
        ):
            return 0.9
        return 0.3
        
    def get_optimal_backend_hint(self) -> str:
        """Get hint for optimal backend selection."""
        cpu_score = self.estimate_cpu_suitability()
        gpu_score = self.estimate_gpu_suitability()
        tpu_score = self.estimate_tpu_suitability()
        
        scores = {"cpu": cpu_score, "gpu": gpu_score, "tpu": tpu_score}
        return max(scores, key=scores.get)
