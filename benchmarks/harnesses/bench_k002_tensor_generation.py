"""Microbenchmark for K002: Tensor Generation.

This benchmark measures the performance of deterministic tensor generation
used in benchmark workloads (benchmarks/quasim_bench.py).

See kernels/MANIFEST.md#K002 for kernel details.
"""
from __future__ import annotations

import statistics
import time
from typing import List

import pytest


def generate_tensor(rank: int, dimension: int) -> List[complex]:
    """Generate a deterministic tensor (current implementation)."""
    if dimension <= 1:
        scale = 0.0
        step = 0.0
    else:
        scale = float(rank + 1)
        step = 1.0 / float(dimension - 1)
    return [complex(idx * step * scale, -idx * step * scale) for idx in range(dimension)]


def benchmark_tensor_generation(batches: int, dimension: int, repeat: int = 50) -> dict[str, float]:
    """Run tensor generation benchmark.
    
    Args:
        batches: Number of tensors to generate
        dimension: Size of each tensor
        repeat: Number of iterations for statistical stability
        
    Returns:
        Dictionary with timing statistics and throughput metrics
    """
    # Warmup
    for batch in range(min(3, batches)):
        _ = generate_tensor(batch, dimension)
    
    # Timed runs
    timings: List[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        for batch in range(batches):
            _ = generate_tensor(batch, dimension)
        end = time.perf_counter()
        timings.append(end - start)
    
    total_elements = batches * dimension
    median_time = statistics.median(timings)
    
    return {
        "min_s": min(timings),
        "median_s": median_time,
        "p95_s": statistics.quantiles(timings, n=20)[18],
        "max_s": max(timings),
        "mean_s": statistics.fmean(timings),
        "stdev_s": statistics.stdev(timings) if len(timings) > 1 else 0.0,
        "elements_per_sec": total_elements / median_time,
        "batches": float(batches),
        "dimension": float(dimension),
        "repeat": float(repeat),
    }


@pytest.mark.benchmark
def test_bench_k002_small():
    """Benchmark small tensor generation (8 batches × 256 elements)."""
    result = benchmark_tensor_generation(batches=8, dimension=256, repeat=50)
    print(f"\n[K002-SMALL] Median: {result['median_s']*1000:.3f}ms, "
          f"Throughput: {result['elements_per_sec']/1e6:.2f}M elem/s")
    assert result['median_s'] > 0


@pytest.mark.benchmark
def test_bench_k002_medium():
    """Benchmark medium tensor generation (32 batches × 2048 elements)."""
    result = benchmark_tensor_generation(batches=32, dimension=2048, repeat=20)
    print(f"\n[K002-MEDIUM] Median: {result['median_s']*1000:.3f}ms, "
          f"Throughput: {result['elements_per_sec']/1e6:.2f}M elem/s")
    assert result['median_s'] > 0


@pytest.mark.benchmark
def test_bench_k002_large():
    """Benchmark large tensor generation (64 batches × 4096 elements)."""
    result = benchmark_tensor_generation(batches=64, dimension=4096, repeat=10)
    print(f"\n[K002-LARGE] Median: {result['median_s']*1000:.3f}ms, "
          f"Throughput: {result['elements_per_sec']/1e6:.2f}M elem/s")
    assert result['median_s'] > 0


if __name__ == "__main__":
    print("=" * 70)
    print("K002: Tensor Generation Benchmark Suite")
    print("=" * 70)
    
    for name, batches, dimension in [
        ("SMALL", 8, 256),
        ("MEDIUM", 32, 2048),
        ("LARGE", 64, 4096),
    ]:
        result = benchmark_tensor_generation(batches, dimension)
        print(f"\n{name} ({batches} batches × {dimension} elements)")
        print(f"  Min:        {result['min_s']*1000:7.3f} ms")
        print(f"  Median:     {result['median_s']*1000:7.3f} ms")
        print(f"  P95:        {result['p95_s']*1000:7.3f} ms")
        print(f"  Max:        {result['max_s']*1000:7.3f} ms")
        print(f"  Throughput: {result['elements_per_sec']/1e6:7.2f} M elem/s")
