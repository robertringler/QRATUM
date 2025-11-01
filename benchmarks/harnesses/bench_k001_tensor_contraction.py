"""Microbenchmark for K001: Tensor Contraction Simulator.

This benchmark measures the performance of the core tensor contraction
simulation kernel in runtime/python/quasim/runtime.py.

See kernels/MANIFEST.md#K001 for kernel details.
"""
from __future__ import annotations

import random
import statistics
import time
from typing import List

import pytest

from quasim.runtime import Config, runtime


def _setup_workload(batches: int, dimension: int) -> list[list[complex]]:
    """Generate deterministic test workload."""
    random.seed(42)  # Ensure reproducibility
    workload = []
    for batch in range(batches):
        scale = float(batch + 1)
        step = 1.0 / float(dimension - 1) if dimension > 1 else 0.0
        tensor = [complex(idx * step * scale, -idx * step * scale) for idx in range(dimension)]
        workload.append(tensor)
    return workload


def benchmark_tensor_contraction(batches: int, dimension: int, repeat: int = 10) -> dict[str, float]:
    """Run tensor contraction benchmark with specified parameters.
    
    Args:
        batches: Number of tensor batches to process
        dimension: Size of each tensor (number of complex elements)
        repeat: Number of iterations for statistical stability
        
    Returns:
        Dictionary with timing statistics and throughput metrics
    """
    config = Config(simulation_precision="fp8", max_workspace_mb=64)
    workload = _setup_workload(batches, dimension)
    
    # Warmup run (not timed)
    with runtime(config) as handle:
        _ = handle.simulate(workload)
    
    # Timed runs
    timings: List[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        with runtime(config) as handle:
            result = handle.simulate(workload)
        end = time.perf_counter()
        timings.append(end - start)
        
        # Sanity check
        assert len(result) == batches, f"Expected {batches} results, got {len(result)}"
    
    total_elements = batches * dimension
    median_time = statistics.median(timings)
    
    return {
        "min_s": min(timings),
        "median_s": median_time,
        "p95_s": statistics.quantiles(timings, n=20)[18],  # 95th percentile
        "max_s": max(timings),
        "mean_s": statistics.fmean(timings),
        "stdev_s": statistics.stdev(timings) if len(timings) > 1 else 0.0,
        "elements_per_sec": total_elements / median_time,
        "batches": float(batches),
        "dimension": float(dimension),
        "repeat": float(repeat),
    }


# Pytest benchmark fixtures
@pytest.mark.benchmark
def test_bench_k001_small():
    """Benchmark small workload (8 batches × 256 elements)."""
    result = benchmark_tensor_contraction(batches=8, dimension=256, repeat=10)
    print(f"\n[K001-SMALL] Median: {result['median_s']*1000:.3f}ms, "
          f"Throughput: {result['elements_per_sec']/1e6:.2f}M elem/s")
    assert result['median_s'] > 0, "Benchmark must complete"


@pytest.mark.benchmark
def test_bench_k001_medium():
    """Benchmark medium workload (32 batches × 2048 elements)."""
    result = benchmark_tensor_contraction(batches=32, dimension=2048, repeat=5)
    print(f"\n[K001-MEDIUM] Median: {result['median_s']*1000:.3f}ms, "
          f"Throughput: {result['elements_per_sec']/1e6:.2f}M elem/s")
    assert result['median_s'] > 0, "Benchmark must complete"


@pytest.mark.benchmark
def test_bench_k001_large():
    """Benchmark large workload (64 batches × 4096 elements)."""
    result = benchmark_tensor_contraction(batches=64, dimension=4096, repeat=3)
    print(f"\n[K001-LARGE] Median: {result['median_s']*1000:.3f}ms, "
          f"Throughput: {result['elements_per_sec']/1e6:.2f}M elem/s")
    assert result['median_s'] > 0, "Benchmark must complete"


if __name__ == "__main__":
    print("=" * 70)
    print("K001: Tensor Contraction Benchmark Suite")
    print("=" * 70)
    
    for name, batches, dimension in [
        ("SMALL", 8, 256),
        ("MEDIUM", 32, 2048),
        ("LARGE", 64, 4096),
    ]:
        result = benchmark_tensor_contraction(batches, dimension)
        print(f"\n{name} ({batches} batches × {dimension} elements)")
        print(f"  Min:        {result['min_s']*1000:7.3f} ms")
        print(f"  Median:     {result['median_s']*1000:7.3f} ms")
        print(f"  P95:        {result['p95_s']*1000:7.3f} ms")
        print(f"  Max:        {result['max_s']*1000:7.3f} ms")
        print(f"  Throughput: {result['elements_per_sec']/1e6:7.2f} M elem/s")
