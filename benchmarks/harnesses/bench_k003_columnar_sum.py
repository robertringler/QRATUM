"""Microbenchmark for K003: Columnar Sum.

This benchmark measures the performance of columnar aggregation
operations in sdk/rapids/dataframe.py.

See kernels/MANIFEST.md#K003 for kernel details.
"""
from __future__ import annotations

import random
import statistics
import time
from typing import Dict, List

import pytest

from sdk.rapids.dataframe import columnar_sum


def _generate_table(num_columns: int, num_rows: int) -> Dict[str, List[float]]:
    """Generate a deterministic test table."""
    random.seed(42)
    table = {}
    for col_idx in range(num_columns):
        col_name = f"col_{col_idx}"
        table[col_name] = [random.uniform(-100.0, 100.0) for _ in range(num_rows)]
    return table


def benchmark_columnar_sum(num_columns: int, num_rows: int, repeat: int = 50) -> dict[str, float]:
    """Run columnar sum benchmark.
    
    Args:
        num_columns: Number of columns in the table
        num_rows: Number of rows per column
        repeat: Number of iterations for statistical stability
        
    Returns:
        Dictionary with timing statistics and throughput metrics
    """
    table = _generate_table(num_columns, num_rows)
    
    # Warmup
    _ = columnar_sum(table)
    
    # Timed runs
    timings: List[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = columnar_sum(table)
        end = time.perf_counter()
        timings.append(end - start)
        
        # Sanity check
        assert len(result) == num_columns
    
    total_elements = num_columns * num_rows
    median_time = statistics.median(timings)
    
    return {
        "min_s": min(timings),
        "median_s": median_time,
        "p95_s": statistics.quantiles(timings, n=20)[18],
        "max_s": max(timings),
        "mean_s": statistics.fmean(timings),
        "stdev_s": statistics.stdev(timings) if len(timings) > 1 else 0.0,
        "elements_per_sec": total_elements / median_time,
        "num_columns": float(num_columns),
        "num_rows": float(num_rows),
        "repeat": float(repeat),
    }


@pytest.mark.benchmark
def test_bench_k003_small():
    """Benchmark small table (10 columns × 1000 rows)."""
    result = benchmark_columnar_sum(num_columns=10, num_rows=1000, repeat=50)
    print(f"\n[K003-SMALL] Median: {result['median_s']*1000:.3f}ms, "
          f"Throughput: {result['elements_per_sec']/1e6:.2f}M elem/s")
    assert result['median_s'] > 0


@pytest.mark.benchmark
def test_bench_k003_medium():
    """Benchmark medium table (50 columns × 10000 rows)."""
    result = benchmark_columnar_sum(num_columns=50, num_rows=10000, repeat=20)
    print(f"\n[K003-MEDIUM] Median: {result['median_s']*1000:.3f}ms, "
          f"Throughput: {result['elements_per_sec']/1e6:.2f}M elem/s")
    assert result['median_s'] > 0


@pytest.mark.benchmark
def test_bench_k003_large():
    """Benchmark large table (100 columns × 50000 rows)."""
    result = benchmark_columnar_sum(num_columns=100, num_rows=50000, repeat=10)
    print(f"\n[K003-LARGE] Median: {result['median_s']*1000:.3f}ms, "
          f"Throughput: {result['elements_per_sec']/1e6:.2f}M elem/s")
    assert result['median_s'] > 0


if __name__ == "__main__":
    print("=" * 70)
    print("K003: Columnar Sum Benchmark Suite")
    print("=" * 70)
    
    for name, num_columns, num_rows in [
        ("SMALL", 10, 1000),
        ("MEDIUM", 50, 10000),
        ("LARGE", 100, 50000),
    ]:
        result = benchmark_columnar_sum(num_columns, num_rows)
        print(f"\n{name} ({num_columns} columns × {num_rows} rows)")
        print(f"  Min:        {result['min_s']*1000:7.3f} ms")
        print(f"  Median:     {result['median_s']*1000:7.3f} ms")
        print(f"  P95:        {result['p95_s']*1000:7.3f} ms")
        print(f"  Max:        {result['max_s']*1000:7.3f} ms")
        print(f"  Throughput: {result['elements_per_sec']/1e6:7.2f} M elem/s")
