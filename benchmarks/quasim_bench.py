"""Micro-benchmark driver for the libquasim Python runtime facade.

Kernel: K002 - Tensor Generation (Benchmark Workload)
======================================================

Purpose:
    Generates deterministic synthetic tensor payloads for benchmarking.
    Ensures reproducible workloads across benchmark runs.

Expected Shapes/Dtypes:
    - Output: 1D list of complex values
    - Typical dimensions: 256, 512, 1024, 2048, 4096
    - Dtypes: complex64, complex128

Mathematical Summary:
    For dimension d, rank r:
        scale = r + 1
        step = 1.0 / (d - 1)
        tensor[i] = complex(i * step * scale, -i * step * scale)

Performance (Baseline):
    - Small (8×256): ~0.31ms, 6.6 M elem/s
    - Medium (32×2048): ~10.0ms, 6.5 M elem/s
    - Large (64×4096): ~40.2ms, 6.5 M elem/s

Optimization Opportunities (see kernels/MANIFEST.md#K002):
    1. Pre-allocate NumPy arrays
    2. Cache pre-computed tensors
    3. Vectorize arithmetic operations

Test Status: ✅ Pass (via benchmark harness)
Last Profiled: 2025-11-01
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import Iterable, List

from quasim.runtime import Config, runtime


def _generate_tensor(rank: int, dimension: int) -> List[complex]:
    """Generate a deterministic tensor payload for benchmarking.
    
    K002 OPTIMIZED: NumPy vectorization for faster tensor generation.
    """
    # Try to use NumPy for vectorization
    try:
        import numpy as np
        
        if dimension <= 1:
            scale = 0.0
            step = 0.0
        else:
            scale = float(rank + 1)
            step = 1.0 / float(dimension - 1)
        
        # Vectorized: create index array and compute in one go
        indices = np.arange(dimension)
        real_parts = indices * step * scale
        imag_parts = -indices * step * scale
        result = (real_parts + 1j * imag_parts).tolist()
        return result
    except ImportError:
        # Fallback to pure Python
        if dimension <= 1:
            scale = 0.0
            step = 0.0
        else:
            scale = float(rank + 1)
            step = 1.0 / float(dimension - 1)
        return [complex(idx * step * scale, -idx * step * scale) for idx in range(dimension)]


def _generate_workload(batches: int, rank: int, dimension: int) -> Iterable[Iterable[complex]]:
    # TODO(K002): Call site for tensor generation (K002)
    for batch in range(batches):
        yield _generate_tensor(rank + batch, dimension)


def run_benchmark(batches: int, rank: int, dimension: int, repeat: int) -> dict[str, float]:
    """Execute the simulated tensor workload and record latency statistics."""
    timings: List[float] = []
    config = Config(simulation_precision="fp8", max_workspace_mb=32)

    for _ in range(repeat):
        start = time.perf_counter()
        with runtime(config) as handle:
            # TODO(K001): Hot call site for tensor contraction (K001)
            handle.simulate(_generate_workload(batches, rank, dimension))
        end = time.perf_counter()
        timings.append(end - start)

    return {
        "min_s": min(timings),
        "median_s": statistics.median(timings),
        "max_s": max(timings),
        "mean_s": statistics.fmean(timings),
        "runs": float(len(timings)),
    }


def _format_results(results: dict[str, float], batches: int, rank: int, dimension: int) -> str:
    header = f"QuASIM Tensor Benchmark — batches={batches} rank={rank} dim={dimension}"
    lines = [header, "=" * len(header)]
    lines.append(f"runs:        {int(results['runs'])}")
    lines.append(f"min (s):     {results['min_s']:.6f}")
    lines.append(f"median (s):  {results['median_s']:.6f}")
    lines.append(f"mean (s):    {results['mean_s']:.6f}")
    lines.append(f"max (s):     {results['max_s']:.6f}")
    throughput = (batches * dimension) / results["mean_s"]
    lines.append(f"elements/s:  {throughput:,.0f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the QuASIM runtime simulator")
    parser.add_argument("--batches", type=int, default=32, help="Number of tensor batches to process")
    parser.add_argument("--rank", type=int, default=4, help="Rank parameter controlling tensor scaling")
    parser.add_argument("--dimension", type=int, default=2048, help="Tensor dimension per batch")
    parser.add_argument("--repeat", type=int, default=5, help="Number of repeated runs")
    args = parser.parse_args()

    results = run_benchmark(args.batches, args.rank, args.dimension, args.repeat)
    print(_format_results(results, args.batches, args.rank, args.dimension))


if __name__ == "__main__":
    main()
