"""Bandwidth-aware dataframe utilities inspired by RAPIDS.

Kernel: K003 - Columnar Sum Aggregation
========================================

Purpose:
    RAPIDS-inspired columnar aggregation. Computes column-wise sums over
    dataframe-like structures (dict of lists).

Expected Shapes/Dtypes:
    - Input: dict[str, list[float]] (column name → values)
    - Typical: 10-100 columns × 1,000-50,000 rows
    - Dtypes: float32, float64

Mathematical Summary:
    For each column C[col]: result[col] = Σ(C[col][i]) for all i
    Independent per-column reduction (embarrassingly parallel).

Performance (Baseline):
    - Small (10×1K): ~0.08ms, 128.5 M elem/s
    - Medium (50×10K): ~3.9ms, 128.4 M elem/s
    - Large (100×50K): ~40.7ms, 122.7 M elem/s

Tiling Strategy:
    Current: None (sequential dict comprehension)
    Planned: Parallel column processing

Optimization Opportunities (see kernels/MANIFEST.md#K003):
    1. Use NumPy sum operations
    2. Parallel column processing (ThreadPoolExecutor)
    3. GPU offload with CuPy
    4. SIMD via Numba JIT

Test Status: ✅ Pass (via benchmark harness)
Last Profiled: 2025-11-01
"""
from __future__ import annotations

def columnar_sum(table: dict[str, list[float]]) -> dict[str, float]:
    # TODO(K003): Columnar aggregation optimization opportunity
    # See kernels/MANIFEST.md#K003 - Use NumPy, parallel processing, or GPU offload
    return {name: float(sum(column)) for name, column in table.items()}
