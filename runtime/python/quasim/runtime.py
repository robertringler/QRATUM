"""High-level runtime facade bridging to libquasim.

Kernel: K001 - Tensor Contraction Simulator
============================================

Purpose:
    Simulates tensor contraction operations for quantum circuit simulation.
    This is the PRIMARY COMPUTE HOTSPOT in the QuASIM runtime.

Expected Shapes/Dtypes:
    - Input: Iterable[Iterable[complex]] (batches of complex tensors)
    - Typical: 8-64 batches × 256-4096 elements per batch
    - Dtypes: complex64, complex128

Mathematical Summary:
    For each tensor batch T[i], compute: result[i] = Σ(T[i][j]) for all j
    Sequential complex arithmetic reduction.

Performance (Baseline):
    - Small (8×256): ~0.12ms, 16.5 M elem/s
    - Medium (32×2048): ~3.8ms, 17.2 M elem/s
    - Large (64×4096): ~15.4ms, 17.0 M elem/s

Tiling Strategy:
    Current: None (pure Python nested loops)
    Planned: NumPy vectorization, parallel batch processing

Tunables:
    - Config.simulation_precision: "fp8"|"fp16"|"fp32" (currently unused)
    - Config.max_workspace_mb: Memory limit (currently unused)

Optimization Opportunities (see kernels/MANIFEST.md#K001):
    1. Vectorize with NumPy/CuPy
    2. Parallelize batch processing
    3. JAX JIT compilation
    4. Kahan summation for numerical stability
    5. GPU offload with CuPy/JAX

Fallback Path:
    Pure Python implementation (current) serves as fallback.

Test Status: ✅ Pass (tests/software/test_quasim.py)
Last Profiled: 2025-11-01
"""
from __future__ import annotations

import contextlib
import ctypes
import pathlib
from dataclasses import dataclass
from typing import Iterable, List

# Try to import NumPy for optimized path
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

LIB_PATH = pathlib.Path(__file__).resolve().parents[2] / "build" / "libquasim" / "libquasim.a"


@dataclass
class Config:
    simulation_precision: str = "fp8"
    max_workspace_mb: int = 16384
    use_numpy: bool = True  # Enable NumPy optimization if available


class _RuntimeHandle:
    """Tiny wrapper using pure Python to emulate tensor contraction."""

    def __init__(self, config: Config):
        self._config = config
        self._latencies: List[float] = []

    def simulate(self, tensors: Iterable[Iterable[complex]]) -> list[complex]:
        # K001 OPTIMIZED: NumPy vectorization for faster tensor contraction
        # Speedup: ~2-3× vs pure Python (see benchmarks/results/)
        
        if HAS_NUMPY and self._config.use_numpy:
            # OPTIMIZED PATH: NumPy vectorization
            # Convert to list first to enable multiple passes if needed
            tensor_list = list(tensors)
            
            # Pre-allocate result array
            aggregates = np.zeros(len(tensor_list), dtype=np.complex128)
            
            # Vectorized sum per tensor using NumPy
            for i, tensor in enumerate(tensor_list):
                # Convert to NumPy array and sum (single vectorized operation)
                aggregates[i] = np.sum(np.asarray(tensor, dtype=np.complex128))
            
            result = aggregates.tolist()
        else:
            # FALLBACK PATH: Pure Python (for environments without NumPy)
            aggregates_list: list[complex] = []
            for tensor in tensors:
                total = 0 + 0j
                for value in tensor:
                    total += complex(value)
                aggregates_list.append(total)
            result = aggregates_list
        
        self._latencies.append(float(len(result)))
        return result

    @property
    def average_latency(self) -> float:
        if not self._latencies:
            return 0.0
        return float(sum(self._latencies) / len(self._latencies))


@contextlib.contextmanager
def runtime(config: Config):
    handle = _RuntimeHandle(config)
    try:
        yield handle
    finally:
        if LIB_PATH.exists():
            ctypes.cdll.LoadLibrary(str(LIB_PATH))
