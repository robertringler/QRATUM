"""Metrics collection utilities for QuASIM benchmarking.

Provides timing, memory, energy, and accuracy measurement utilities
with support for CUDA, ROCm, and CPU backends.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TimingResult:
    """Timing statistics for a benchmark run."""

    latency_ms_p50: float
    latency_ms_p90: float
    latency_ms_p99: float
    latency_ms_mean: float
    latency_ms_std: float
    throughput_ops_per_sec: float
    iterations: int
    raw_times_ms: List[float] = field(default_factory=list)


@dataclass
class MemoryResult:
    """Memory usage statistics."""

    peak_mb: float
    allocated_mb: float
    reserved_mb: float


@dataclass
class EnergyResult:
    """Energy consumption statistics."""

    energy_j: float
    power_w: float
    duration_s: float


@dataclass
class AccuracyResult:
    """Accuracy and determinism metrics."""

    rmse: float
    max_error: float
    relative_error: float
    determinism_std: float


@dataclass
class BenchmarkResult:
    """Complete benchmark result for a kernel."""

    kernel_name: str
    backend: str
    precision: str
    problem_size: Dict[str, Any]
    timing: TimingResult
    memory: Optional[MemoryResult]
    energy: Optional[EnergyResult]
    accuracy: Optional[AccuracyResult]
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collects performance and accuracy metrics for benchmarks."""

    def __init__(self, backend: str = "cpu"):
        """Initialize metrics collector.

        Args:
            backend: Computation backend (cpu, cuda, rocm)
        """
        self.backend = backend.lower()
        self._cuda_available = False
        self._rocm_available = False

        if self.backend == "cuda":
            try:
                import torch

                self._cuda_available = torch.cuda.is_available()
                if self._cuda_available:
                    self.torch = torch
                    logger.info("CUDA backend available")
            except ImportError:
                logger.warning("PyTorch not available; CUDA metrics disabled")

        elif self.backend == "rocm":
            try:
                import torch

                self._rocm_available = torch.cuda.is_available()  # ROCm uses torch.cuda
                if self._rocm_available:
                    self.torch = torch
                    logger.info("ROCm backend available")
            except ImportError:
                logger.warning("PyTorch not available; ROCm metrics disabled")

    def time_execution(
        self, func: Callable, iterations: int = 30, warmup: int = 3
    ) -> TimingResult:
        """Time function execution with warmup.

        Args:
            func: Function to benchmark
            iterations: Number of timed iterations
            warmup: Number of warmup iterations

        Returns:
            TimingResult with statistics
        """
        logger.debug(f"Running {warmup} warmup iterations")
        for _ in range(warmup):
            func()

        logger.debug(f"Running {iterations} timed iterations")
        times_ms = []

        for _ in range(iterations):
            if self._cuda_available or self._rocm_available:
                self.torch.cuda.synchronize()
                start = time.perf_counter()
                func()
                self.torch.cuda.synchronize()
                end = time.perf_counter()
            else:
                start = time.perf_counter()
                func()
                end = time.perf_counter()

            times_ms.append((end - start) * 1000.0)

        times_array = np.array(times_ms)
        mean_time = float(np.mean(times_array))
        throughput = 1000.0 / mean_time if mean_time > 0 else 0.0

        return TimingResult(
            latency_ms_p50=float(np.percentile(times_array, 50)),
            latency_ms_p90=float(np.percentile(times_array, 90)),
            latency_ms_p99=float(np.percentile(times_array, 99)),
            latency_ms_mean=mean_time,
            latency_ms_std=float(np.std(times_array)),
            throughput_ops_per_sec=throughput,
            iterations=iterations,
            raw_times_ms=times_ms,
        )

    def measure_memory(self) -> Optional[MemoryResult]:
        """Measure current memory usage.

        Returns:
            MemoryResult or None if backend doesn't support memory tracking
        """
        if not (self._cuda_available or self._rocm_available):
            return None

        allocated = self.torch.cuda.memory_allocated() / 1024**2
        reserved = self.torch.cuda.memory_reserved() / 1024**2
        peak = self.torch.cuda.max_memory_allocated() / 1024**2

        return MemoryResult(peak_mb=peak, allocated_mb=allocated, reserved_mb=reserved)

    def reset_memory_stats(self):
        """Reset peak memory statistics."""
        if self._cuda_available or self._rocm_available:
            self.torch.cuda.reset_peak_memory_stats()

    def measure_energy(self, duration_s: float) -> Optional[EnergyResult]:
        """Estimate energy consumption.

        Args:
            duration_s: Duration of measurement in seconds

        Returns:
            EnergyResult or None if energy measurement unavailable
        """
        if self.backend == "cuda":
            # Try nvidia-smi for power measurement
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    power_w = float(result.stdout.strip().split("\n")[0])
                    energy_j = power_w * duration_s
                    return EnergyResult(energy_j=energy_j, power_w=power_w, duration_s=duration_s)
            except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
                logger.debug(f"nvidia-smi unavailable: {e}")

        elif self.backend == "rocm":
            # Try rocm-smi for power measurement
            try:
                result = subprocess.run(
                    ["rocm-smi", "--showpower"], capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    # Parse power from rocm-smi output (simplified)
                    for line in result.stdout.split("\n"):
                        if "Average Graphics Package Power" in line:
                            power_w = float(line.split()[-2])
                            energy_j = power_w * duration_s
                            return EnergyResult(
                                energy_j=energy_j, power_w=power_w, duration_s=duration_s
                            )
            except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
                logger.debug(f"rocm-smi unavailable: {e}")

        return None

    def compute_accuracy(
        self, result: np.ndarray, reference: np.ndarray
    ) -> Optional[AccuracyResult]:
        """Compute accuracy metrics against reference.

        Args:
            result: Computed result
            reference: Reference result

        Returns:
            AccuracyResult with error metrics
        """
        if result is None or reference is None:
            return None

        result_flat = np.asarray(result).flatten()
        reference_flat = np.asarray(reference).flatten()

        if len(result_flat) != len(reference_flat):
            logger.warning("Result and reference shapes differ")
            return None

        diff = result_flat - reference_flat
        rmse = float(np.sqrt(np.mean(diff**2)))
        max_error = float(np.max(np.abs(diff)))

        ref_norm = np.linalg.norm(reference_flat)
        relative_error = rmse / ref_norm if ref_norm > 0 else float("inf")

        return AccuracyResult(
            rmse=rmse,
            max_error=max_error,
            relative_error=relative_error,
            determinism_std=0.0,  # Computed separately across seeds
        )


def save_json(data: Any, output_path: Path, indent: int = 2):
    """Save data as JSON.

    Args:
        data: Data to save (must be JSON serializable)
        output_path: Output file path
        indent: JSON indentation level
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclasses to dicts
    if hasattr(data, "__dataclass_fields__"):
        data = asdict(data)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=indent, default=str)

    logger.info(f"Saved JSON to {output_path}")


def load_json(input_path: Path) -> Any:
    """Load JSON data.

    Args:
        input_path: Input file path

    Returns:
        Loaded data
    """
    with open(input_path, "r") as f:
        return json.load(f)


def generate_markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    """Generate markdown table.

    Args:
        headers: Column headers
        rows: Table rows

    Returns:
        Markdown table string
    """
    lines = []

    # Header
    lines.append("| " + " | ".join(str(h) for h in headers) + " |")

    # Separator
    lines.append("| " + " | ".join("---" for _ in headers) + " |")

    # Rows
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")

    return "\n".join(lines)


def get_system_info() -> Dict[str, Any]:
    """Collect system and environment information.

    Returns:
        Dictionary with system information
    """
    info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "kernel": platform.release(),
        "python_version": platform.python_version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
    }

    # Git info
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5
        ).stdout.strip()

        info["git"] = {"commit": commit, "branch": branch, "dirty": bool(dirty)}
    except (subprocess.TimeoutExpired, FileNotFoundError):
        info["git"] = None

    # CUDA info
    try:
        import torch

        if torch.cuda.is_available():
            info["cuda"] = {
                "available": True,
                "version": torch.version.cuda,
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None,
            }
        else:
            info["cuda"] = {"available": False}
    except ImportError:
        info["cuda"] = None

    # Library versions
    try:
        import numpy

        info["numpy_version"] = numpy.__version__
    except ImportError:
        info["numpy_version"] = None

    return info
