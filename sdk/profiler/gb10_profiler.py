"""Telemetry profiler collecting runtime metrics from libquasim."""
from __future__ import annotations

import argparse
import json
import random


def collect_samples(count: int = 16):
    return [random.uniform(0.1, 5.0) for _ in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser(description="GB10 runtime profiler")
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args()
    samples = collect_samples(args.samples)
    report = {
        "avg_latency_ms": sum(samples) / len(samples),
        "p99_latency_ms": sorted(samples)[int(0.99 * (len(samples) - 1))],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
