#!/usr/bin/env python3
"""QuASIM Benchmark Suite - Main orchestrator for comprehensive kernel benchmarking.

Discovers all available kernels, runs standardized benchmarks, and generates
reproducible metrics with regression detection.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "runtime" / "python"))

from tools.metrics import (
    AccuracyResult,
    BenchmarkResult,
    MemoryResult,
    MetricsCollector,
    TimingResult,
    generate_markdown_table,
    get_system_info,
    load_json,
    save_json,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class KernelInfo:
    """Information about a discovered kernel."""

    name: str
    path: str
    backend: str
    module_name: Optional[str] = None
    function_name: Optional[str] = None
    test_command: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)


class KernelDiscovery:
    """Discovers available kernels in the repository."""

    KERNEL_PATHS = [
        "kernels",
        "src/kernels",
        "quasim/kernels",
        "integrations/kernels",
        "autonomous_systems_platform/services/backend/quasim/kernels",
    ]

    def __init__(self, root_dir: Path):
        """Initialize kernel discovery.

        Args:
            root_dir: Repository root directory
        """
        self.root_dir = root_dir

    def discover_kernels(self) -> List[KernelInfo]:
        """Discover all available kernels.

        Returns:
            List of discovered kernels
        """
        kernels = []

        # Search kernel directories
        for kernel_path in self.KERNEL_PATHS:
            full_path = self.root_dir / kernel_path
            if full_path.exists():
                logger.info(f"Scanning {full_path}")
                kernels.extend(self._scan_directory(full_path, kernel_path))

        # Add special known kernels
        kernels.extend(self._discover_special_kernels())

        logger.info(f"Discovered {len(kernels)} kernels")
        return kernels

    def _scan_directory(self, path: Path, base_path: str) -> List[KernelInfo]:
        """Scan directory for kernel implementations.

        Args:
            path: Directory to scan
            base_path: Base path relative to root

        Returns:
            List of discovered kernels
        """
        kernels = []

        for py_file in path.rglob("*.py"):
            if py_file.name.startswith("__") or py_file.name.startswith("test_"):
                continue

            # Determine kernel name and backend
            rel_path = py_file.relative_to(self.root_dir)
            kernel_name = py_file.stem

            # Try to infer backend from code or location
            backend = self._infer_backend(py_file)

            # Check for bench.yaml config
            config_path = py_file.parent / "bench.yaml"
            config = {}
            if config_path.exists():
                import yaml

                with open(config_path) as f:
                    config = yaml.safe_load(f) or {}

            kernels.append(
                KernelInfo(
                    name=kernel_name,
                    path=str(rel_path),
                    backend=backend,
                    module_name=str(rel_path.with_suffix("")).replace("/", "."),
                    config=config,
                )
            )

        return kernels

    def _infer_backend(self, py_file: Path) -> str:
        """Infer backend from file content or location.

        Args:
            py_file: Python file to analyze

        Returns:
            Backend name (cuda, rocm, cpu, jax, etc.)
        """
        try:
            content = py_file.read_text()

            if "import cupy" in content or "from cupy" in content:
                return "cuda"
            elif "torch.cuda" in content and "hip" in content.lower():
                return "rocm"
            elif "import jax" in content or "from jax" in content:
                return "jax"
            elif "torch" in content:
                return "pytorch"
            else:
                return "cpu"
        except Exception as e:
            logger.debug(f"Error reading {py_file}: {e}")
            return "cpu"

    def _discover_special_kernels(self) -> List[KernelInfo]:
        """Discover special known kernels.

        Returns:
            List of special kernels
        """
        special = []

        # QuASIM runtime benchmark
        runtime_bench = self.root_dir / "benchmarks" / "quasim_bench.py"
        if runtime_bench.exists():
            special.append(
                KernelInfo(
                    name="quasim_runtime",
                    path=str(runtime_bench.relative_to(self.root_dir)),
                    backend="cpu",
                    module_name="benchmarks.quasim_bench",
                    function_name="run_benchmark",
                )
            )

        return special


class BenchmarkRunner:
    """Runs benchmarks for discovered kernels."""

    def __init__(
        self,
        kernels: List[KernelInfo],
        output_dir: Path,
        iterations: int = 30,
        warmup: int = 3,
        precisions: List[str] = None,
        backends: List[str] = None,
        seed: int = 1337,
    ):
        """Initialize benchmark runner.

        Args:
            kernels: List of kernels to benchmark
            output_dir: Output directory for results
            iterations: Number of benchmark iterations
            warmup: Number of warmup iterations
            precisions: List of precisions to test (fp32, fp16, etc.)
            backends: List of backends to use (auto uses available)
            seed: Random seed for reproducibility
        """
        self.kernels = kernels
        self.output_dir = output_dir
        self.iterations = iterations
        self.warmup = warmup
        self.precisions = precisions or ["fp32"]
        self.backends = backends or ["auto"]
        self.seed = seed

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "kernels").mkdir(exist_ok=True)

        # Set global seed
        np.random.seed(seed)

    def run_all(self) -> List[BenchmarkResult]:
        """Run benchmarks for all kernels.

        Returns:
            List of benchmark results
        """
        results = []

        for kernel in self.kernels:
            logger.info(f"Benchmarking {kernel.name} ({kernel.backend})")

            try:
                result = self._run_kernel_benchmark(kernel)
                if result:
                    results.append(result)

                    # Save individual kernel result
                    kernel_output = self.output_dir / "kernels" / f"{kernel.name}.json"
                    save_json(result, kernel_output)

            except Exception as e:
                logger.error(f"Failed to benchmark {kernel.name}: {e}")
                logger.debug(traceback.format_exc())

                # Create failure result
                results.append(
                    BenchmarkResult(
                        kernel_name=kernel.name,
                        backend=kernel.backend,
                        precision="unknown",
                        problem_size={},
                        timing=TimingResult(0, 0, 0, 0, 0, 0, 0),
                        memory=None,
                        energy=None,
                        accuracy=None,
                        success=False,
                        error_message=str(e),
                    )
                )

        return results

    def _run_kernel_benchmark(self, kernel: KernelInfo) -> Optional[BenchmarkResult]:
        """Run benchmark for a single kernel.

        Args:
            kernel: Kernel to benchmark

        Returns:
            Benchmark result or None if failed
        """
        # Determine backend to use
        backend = self._resolve_backend(kernel.backend)
        metrics = MetricsCollector(backend)

        # Load kernel module
        module = self._load_kernel_module(kernel)
        if not module:
            return None

        # Find benchmark function
        bench_func = self._find_benchmark_function(module, kernel)
        if not bench_func:
            return None

        # Get problem size from config or use defaults
        problem_size = kernel.config.get("problem_size", self._default_problem_size(kernel))

        # Run benchmark
        logger.debug(f"Running {self.iterations} iterations with {self.warmup} warmup")

        metrics.reset_memory_stats()

        # Wrap function for timing
        def run_once():
            return bench_func(**problem_size)

        # Time execution
        timing = metrics.time_execution(run_once, iterations=self.iterations, warmup=self.warmup)

        # Measure memory
        memory = metrics.measure_memory()

        # Estimate energy
        energy = metrics.measure_energy(timing.latency_ms_mean / 1000.0)

        # Run for accuracy if reference available
        accuracy = None
        if hasattr(module, "reference_result"):
            result = run_once()
            reference = module.reference_result(**problem_size)
            accuracy = metrics.compute_accuracy(result, reference)

        return BenchmarkResult(
            kernel_name=kernel.name,
            backend=backend,
            precision=self.precisions[0],  # Use first precision for now
            problem_size=problem_size,
            timing=timing,
            memory=memory,
            energy=energy,
            accuracy=accuracy,
            success=True,
            metadata={"iterations": self.iterations, "warmup": self.warmup, "seed": self.seed},
        )

    def _resolve_backend(self, backend: str) -> str:
        """Resolve backend to actual available backend.

        Args:
            backend: Requested backend

        Returns:
            Available backend
        """
        if backend == "auto" or "auto" in self.backends:
            # Try CUDA first, then ROCm, then CPU
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass

            return "cpu"

        return backend

    def _load_kernel_module(self, kernel: KernelInfo) -> Optional[Any]:
        """Load kernel module.

        Args:
            kernel: Kernel to load

        Returns:
            Loaded module or None
        """
        try:
            if kernel.module_name:
                # Try importing as module
                spec = importlib.util.find_spec(kernel.module_name)
                if spec and spec.origin:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    return module

            # Try loading from path
            kernel_path = PROJECT_ROOT / kernel.path
            if kernel_path.exists():
                spec = importlib.util.spec_from_file_location(kernel.name, kernel_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    return module

        except Exception as e:
            logger.error(f"Failed to load kernel {kernel.name}: {e}")
            logger.debug(traceback.format_exc())

        return None

    def _find_benchmark_function(self, module: Any, kernel: KernelInfo) -> Optional[callable]:
        """Find benchmark function in module.

        Args:
            module: Loaded module
            kernel: Kernel info

        Returns:
            Benchmark function or None
        """
        # Check for explicit function name
        if kernel.function_name and hasattr(module, kernel.function_name):
            return getattr(module, kernel.function_name)

        # Look for kernel-specific function (e.g., autonomous_systems_kernel)
        if hasattr(module, f"{kernel.name}_kernel"):
            return getattr(module, f"{kernel.name}_kernel")

        # Look for common benchmark function names
        for name in ["run_benchmark", "benchmark", "solve"]:
            if hasattr(module, name):
                func = getattr(module, name)
                if callable(func):
                    return func

        # Special handling for pressure_poisson - wrap main function
        if kernel.name == "pressure_poisson" and hasattr(module, "PressurePoissonSolver"):
            def wrapper(**kwargs):
                from integrations.kernels.cfd.pressure_poisson import PressurePoissonConfig, PressurePoissonSolver
                
                config = PressurePoissonConfig(
                    grid_size=kwargs.get("grid_size", (32, 32, 32)),
                    max_iterations=kwargs.get("max_iterations", 50),
                    tolerance=kwargs.get("tolerance", 1e-5),
                    precision=module.Precision.FP32,
                    backend=module.Backend.CPU,
                    deterministic=True,
                    seed=self.seed
                )
                solver = PressurePoissonSolver(config)
                return solver.solve()
            
            return wrapper

        logger.warning(f"No benchmark function found for {kernel.name}")
        return None

    def _default_problem_size(self, kernel: KernelInfo) -> Dict[str, Any]:
        """Get default problem size for kernel.

        Args:
            kernel: Kernel info

        Returns:
            Default problem size configuration
        """
        # Kernel-specific defaults
        if "pressure_poisson" in kernel.name:
            return {"batches": 1, "rank": 2, "dimension": 64}
        elif "autonomous" in kernel.name:
            return {"seed": self.seed, "scale": 1.0}
        elif "quasim_runtime" in kernel.name:
            return {"batches": 16, "rank": 2, "dimension": 1024, "repeat": self.iterations}

        # Generic defaults
        return {"size": 1024, "iterations": 1}


class ReportGenerator:
    """Generates benchmark reports."""

    def __init__(self, results: List[BenchmarkResult], output_dir: Path, env_info: Dict[str, Any]):
        """Initialize report generator.

        Args:
            results: Benchmark results
            output_dir: Output directory
            env_info: Environment information
        """
        self.results = results
        self.output_dir = output_dir
        self.env_info = env_info

    def generate_all(self, baseline_path: Optional[Path] = None):
        """Generate all reports.

        Args:
            baseline_path: Path to baseline summary.json for regression detection
        """
        # Save environment info
        save_json(self.env_info, self.output_dir / "env.json")

        # Generate summary JSON
        summary = self._generate_summary()
        save_json(summary, self.output_dir / "summary.json")

        # Generate markdown report
        md_report = self._generate_markdown_report(summary)
        (self.output_dir / "summary.md").write_text(md_report)

        # Generate regressions report if baseline provided
        if baseline_path and baseline_path.exists():
            regressions = self._generate_regressions(baseline_path)
            (self.output_dir / "regressions.md").write_text(regressions)

        logger.info(f"Generated reports in {self.output_dir}")

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics.

        Returns:
            Summary dictionary
        """
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]

        summary = {
            "timestamp": self.env_info.get("git", {}).get("commit", "unknown"),
            "total_kernels": len(self.results),
            "successful": len(successful),
            "failed": len(failed),
            "environment": self.env_info,
            "results": [asdict(r) for r in self.results],
        }

        return summary

    def _generate_markdown_report(self, summary: Dict[str, Any]) -> str:
        """Generate markdown report.

        Args:
            summary: Summary dictionary

        Returns:
            Markdown report string
        """
        lines = ["# QuASIM Benchmark Report", ""]

        # Environment section
        lines.extend(self._format_environment_section())
        lines.append("")

        # Overall stats
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Total Kernels**: {summary['total_kernels']}")
        lines.append(f"- **Successful**: {summary['successful']}")
        lines.append(f"- **Failed**: {summary['failed']}")
        lines.append("")

        # Performance leaderboard
        lines.extend(self._format_leaderboard_section())
        lines.append("")

        # Memory and energy
        lines.extend(self._format_resource_section())
        lines.append("")

        # Key findings
        lines.extend(self._format_key_findings())
        lines.append("")

        # Recommendations
        lines.extend(self._format_recommendations())
        lines.append("")

        return "\n".join(lines)

    def _format_environment_section(self) -> List[str]:
        """Format environment section.

        Returns:
            List of markdown lines
        """
        lines = ["## Environment", ""]

        git_info = self.env_info.get("git", {})
        if git_info:
            lines.append(f"- **Commit**: `{git_info.get('commit', 'unknown')[:8]}`")
            lines.append(f"- **Branch**: `{git_info.get('branch', 'unknown')}`")
            lines.append(f"- **Dirty**: {git_info.get('dirty', False)}")
        lines.append("")

        lines.append(f"- **OS**: {self.env_info.get('os', 'unknown')}")
        lines.append(f"- **Python**: {self.env_info.get('python_version', 'unknown')}")
        lines.append("")

        cuda_info = self.env_info.get("cuda", {})
        if cuda_info and cuda_info.get("available"):
            lines.append(f"- **CUDA**: {cuda_info.get('version', 'unknown')}")
            lines.append(f"- **GPU**: {cuda_info.get('device_name', 'unknown')}")
            lines.append("")

        return lines

    def _format_leaderboard_section(self) -> List[str]:
        """Format performance leaderboard.

        Returns:
            List of markdown lines
        """
        lines = ["## Performance Leaderboard", ""]

        successful = [r for r in self.results if r.success]
        if not successful:
            lines.append("No successful benchmarks.")
            return lines

        # Sort by p50 latency
        sorted_results = sorted(successful, key=lambda r: r.timing.latency_ms_p50)

        headers = ["Kernel", "Backend", "Precision", "p50 (ms)", "p90 (ms)", "Throughput (ops/s)"]
        rows = []

        for result in sorted_results[:10]:  # Top 10
            rows.append(
                [
                    result.kernel_name,
                    result.backend,
                    result.precision,
                    f"{result.timing.latency_ms_p50:.3f}",
                    f"{result.timing.latency_ms_p90:.3f}",
                    f"{result.timing.throughput_ops_per_sec:.2f}",
                ]
            )

        lines.append(generate_markdown_table(headers, rows))

        return lines

    def _format_resource_section(self) -> List[str]:
        """Format resource usage section.

        Returns:
            List of markdown lines
        """
        lines = ["## Resource Usage", ""]

        successful = [r for r in self.results if r.success and r.memory]
        if not successful:
            lines.append("No memory data available.")
            return lines

        headers = ["Kernel", "Peak Memory (MB)", "Energy (J)", "Power (W)"]
        rows = []

        for result in successful:
            energy_j = result.energy.energy_j if result.energy else 0.0
            power_w = result.energy.power_w if result.energy else 0.0

            rows.append(
                [
                    result.kernel_name,
                    f"{result.memory.peak_mb:.1f}" if result.memory else "N/A",
                    f"{energy_j:.3f}" if energy_j > 0 else "N/A",
                    f"{power_w:.1f}" if power_w > 0 else "N/A",
                ]
            )

        lines.append(generate_markdown_table(headers, rows))

        return lines

    def _format_key_findings(self) -> List[str]:
        """Format key findings section.

        Returns:
            List of markdown lines
        """
        lines = ["## Key Findings", ""]

        successful = [r for r in self.results if r.success]

        if successful:
            # Fastest kernel
            fastest = min(successful, key=lambda r: r.timing.latency_ms_p50)
            lines.append(
                f"- **Fastest Kernel**: `{fastest.kernel_name}` "
                f"({fastest.timing.latency_ms_p50:.3f} ms p50)"
            )

            # Highest throughput
            highest_throughput = max(successful, key=lambda r: r.timing.throughput_ops_per_sec)
            lines.append(
                f"- **Highest Throughput**: `{highest_throughput.kernel_name}` "
                f"({highest_throughput.timing.throughput_ops_per_sec:.2f} ops/s)"
            )

            # Memory efficient
            mem_results = [r for r in successful if r.memory]
            if mem_results:
                most_efficient = min(mem_results, key=lambda r: r.memory.peak_mb)
                lines.append(
                    f"- **Most Memory Efficient**: `{most_efficient.kernel_name}` "
                    f"({most_efficient.memory.peak_mb:.1f} MB peak)"
                )

            # Backend comparison
            backends = set(r.backend for r in successful)
            if len(backends) > 1:
                lines.append(f"- **Backends Tested**: {', '.join(sorted(backends))}")

        failed = [r for r in self.results if not r.success]
        if failed:
            lines.append(f"- **Failed Benchmarks**: {len(failed)} kernel(s) failed to run")

        return lines

    def _format_recommendations(self) -> List[str]:
        """Format recommendations section.

        Returns:
            List of markdown lines
        """
        lines = ["## Recommendations", ""]

        successful = [r for r in self.results if r.success]

        if successful:
            # Check for high variance
            high_variance = [r for r in successful if r.timing.latency_ms_std > r.timing.latency_ms_mean * 0.1]
            if high_variance:
                lines.append(
                    f"- **High Variance Detected**: {len(high_variance)} kernel(s) show >10% stddev. "
                    "Consider investigating sources of non-determinism."
                )

            # Check for memory usage
            mem_results = [r for r in successful if r.memory and r.memory.peak_mb > 1000]
            if mem_results:
                lines.append(
                    f"- **High Memory Usage**: {len(mem_results)} kernel(s) use >1GB memory. "
                    "Consider memory optimization."
                )

            # General recommendations
            lines.append(
                "- **Precision Testing**: Consider testing additional precisions (FP16, FP8) "
                "for speed vs. accuracy trade-offs."
            )

        else:
            lines.append("- No successful benchmarks to analyze.")

        return lines

    def _generate_regressions(self, baseline_path: Path) -> str:
        """Generate regressions report.

        Args:
            baseline_path: Path to baseline summary.json

        Returns:
            Markdown regressions report
        """
        lines = ["# Regression Report", ""]

        try:
            baseline = load_json(baseline_path)
            baseline_results = {r["kernel_name"]: r for r in baseline.get("results", [])}

            regressions = []

            for result in self.results:
                if not result.success:
                    continue

                baseline_result = baseline_results.get(result.kernel_name)
                if not baseline_result or not baseline_result.get("success"):
                    continue

                # Check latency regression (>10%)
                baseline_latency = baseline_result["timing"]["latency_ms_p50"]
                current_latency = result.timing.latency_ms_p50
                latency_delta = (current_latency - baseline_latency) / baseline_latency

                if latency_delta > 0.10:
                    regressions.append(
                        f"- **{result.kernel_name}**: Latency increased by "
                        f"{latency_delta * 100:.1f}% ({baseline_latency:.3f} ms → {current_latency:.3f} ms)"
                    )

            if regressions:
                lines.append("## Detected Regressions")
                lines.append("")
                lines.extend(regressions)
            else:
                lines.append("No regressions detected. ✅")

        except Exception as e:
            lines.append(f"Error comparing to baseline: {e}")

        return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="QuASIM Comprehensive Benchmark Suite")

    parser.add_argument(
        "--iters", type=int, default=30, help="Number of benchmark iterations (default: 30)"
    )
    parser.add_argument(
        "--warmup", type=int, default=3, help="Number of warmup iterations (default: 3)"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        help="Comma-separated precisions to test (default: fp32)",
    )
    parser.add_argument(
        "--backends",
        type=str,
        default="auto",
        help="Comma-separated backends to use (default: auto)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports"), help="Output directory (default: reports)"
    )
    parser.add_argument(
        "--seed", type=int, default=1337, help="Random seed for reproducibility (default: 1337)"
    )
    parser.add_argument(
        "--compare-to",
        type=str,
        help="Git ref to compare against for regression detection (e.g., main)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse precisions and backends
    precisions = [p.strip() for p in args.precision.split(",")]
    backends = [b.strip() for b in args.backends.split(",")]

    logger.info("=" * 60)
    logger.info("QuASIM Benchmark Suite")
    logger.info("=" * 60)

    # Get system info
    logger.info("Collecting environment information...")
    env_info = get_system_info()

    # Discover kernels
    logger.info("Discovering kernels...")
    discovery = KernelDiscovery(PROJECT_ROOT)
    kernels = discovery.discover_kernels()

    # Save kernel manifest
    manifest = [asdict(k) for k in kernels]
    save_json(manifest, args.output_dir / "kernel_manifest.json")

    # Run benchmarks
    logger.info(f"Running benchmarks with {args.iters} iterations...")
    runner = BenchmarkRunner(
        kernels=kernels,
        output_dir=args.output_dir,
        iterations=args.iters,
        warmup=args.warmup,
        precisions=precisions,
        backends=backends,
        seed=args.seed,
    )

    results = runner.run_all()

    # Get baseline for comparison if specified
    baseline_path = None
    if args.compare_to:
        logger.info(f"Fetching baseline from {args.compare_to}...")
        try:
            # Try to get baseline from git
            subprocess.run(
                ["git", "fetch", "origin", args.compare_to], check=True, capture_output=True
            )

            baseline_file = f"origin/{args.compare_to}:reports/summary.json"
            baseline_json = subprocess.run(
                ["git", "show", baseline_file], capture_output=True, text=True, check=True
            ).stdout

            baseline_path = args.output_dir / "baseline_summary.json"
            baseline_path.write_text(baseline_json)

        except subprocess.CalledProcessError as e:
            logger.warning(f"Could not fetch baseline: {e}")

    # Generate reports
    logger.info("Generating reports...")
    generator = ReportGenerator(results, args.output_dir, env_info)
    generator.generate_all(baseline_path)

    # Print summary
    logger.info("=" * 60)
    logger.info("Benchmark Complete")
    logger.info("=" * 60)
    logger.info(f"Total kernels: {len(results)}")
    logger.info(f"Successful: {len([r for r in results if r.success])}")
    logger.info(f"Failed: {len([r for r in results if not r.success])}")
    logger.info(f"Reports saved to: {args.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
