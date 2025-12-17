"""Micro-benchmark driver for the libquasim Python runtime facade."""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Iterable

from quasim.runtime import Config, runtime


def _generate_tensor(rank: int, dimension: int) -> list[complex]:
    """Generate a deterministic tensor payload for benchmarking."""

    if dimension <= 1:
        scale = 0.0
        step = 0.0
    else:
        scale = float(rank + 1)
        step = 1.0 / float(dimension - 1)
    return [complex(idx * step * scale, -idx * step * scale) for idx in range(dimension)]


def _generate_workload(batches: int, rank: int, dimension: int) -> Iterable[Iterable[complex]]:
    for batch in range(batches):
        yield _generate_tensor(rank + batch, dimension)


def run_benchmark(batches: int, rank: int, dimension: int, repeat: int) -> dict[str, float]:
    """Execute the simulated tensor workload and record latency statistics."""

    timings: list[float] = []
    config = Config(simulation_precision="fp8", max_workspace_mb=32)

    for _ in range(repeat):
        start = time.perf_counter()
        with runtime(config) as handle:
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
    parser.add_argument(
        "--batches", type=int, default=32, help="Number of tensor batches to process"
    )
    parser.add_argument(
        "--rank", type=int, default=4, help="Rank parameter controlling tensor scaling"
    )
    parser.add_argument("--dimension", type=int, default=2048, help="Tensor dimension per batch")
    parser.add_argument("--repeat", type=int, default=5, help="Number of repeated runs")
    args = parser.parse_args()

    results = run_benchmark(args.batches, args.rank, args.dimension, args.repeat)
    print(_format_results(results, args.batches, args.rank, args.dimension))


if __name__ == "__main__":
    main()
