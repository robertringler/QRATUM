"""Benchmark validation check (TECH-001).

Validates QuASIM benchmark results against expected speedups from baseline.
"""

import os
from typing import Any

import numpy as np

from ..models import CheckResult


def run(cfg: dict[str, Any]) -> CheckResult:
    """Run benchmark validation check.

    Loads benchmark NPZ files and validates that speedups meet expected thresholds.

    Args:
        cfg: Configuration dictionary containing:
            - inputs.artifacts.benchmarks_npz_dir: Directory with benchmark NPZ files
            - policy.tolerances.benchmark_speedup_min: Minimum required speedup (default: 10.0)

    Returns:
        CheckResult with pass/fail status and detailed metrics
    """

    try:
        bench_dir = cfg["inputs"]["artifacts"]["benchmarks_npz_dir"]
        min_speedup = cfg["policy"]["tolerances"].get("benchmark_speedup_min", 10.0)

        if not os.path.exists(bench_dir):
            return CheckResult(
                id="TECH-001",
                passed=False,
                details={"error": f"Benchmark directory not found: {bench_dir}"},
            )

        npz_files = [f for f in os.listdir(bench_dir) if f.endswith(".npz")]

        if not npz_files:
            return CheckResult(
                id="TECH-001",
                passed=False,
                details={"error": "No benchmark NPZ files found", "dir": bench_dir},
            )

        results = {}
        all_pass = True

        for npz_file in npz_files:
            path = os.path.join(bench_dir, npz_file)
            try:
                data = np.load(path)
                # Expected structure: baseline_time, quasim_time, speedup
                if "speedup" in data:
                    speedup = float(data["speedup"])
                elif "baseline_time" in data and "quasim_time" in data:
                    baseline = float(data["baseline_time"])
                    quasim = float(data["quasim_time"])
                    speedup = baseline / max(quasim, 1e-9)
                else:
                    results[npz_file] = {"error": "Missing speedup or timing data"}
                    all_pass = False
                    continue

                passed = speedup >= min_speedup
                results[npz_file] = {"speedup": speedup, "passed": passed}
                if not passed:
                    all_pass = False

            except Exception as e:
                results[npz_file] = {"error": str(e)}
                all_pass = False

        evidence_paths = [os.path.join(bench_dir, f) for f in npz_files]

        return CheckResult(
            id="TECH-001",
            passed=all_pass,
            details={
                "results": results,
                "min_speedup": min_speedup,
                "files_checked": len(npz_files),
            },
            evidence_paths=evidence_paths,
        )

    except Exception as e:
        return CheckResult(id="TECH-001", passed=False, details={"error": str(e)})
