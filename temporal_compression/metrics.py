"""Temporal Compression Metrics Module.

Provides measurement and calculation of temporal compression metrics
for deterministic simulation experiments.

Metrics include:
- Temporal Compression Ratio (simulated time / wall-clock time)
- Effective Real-Time Acceleration Factor
- Determinism Consistency (bit-exact reproducibility)
- Verification Overhead
- Scaling Behavior
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TimingMeasurement:
    """Single timing measurement.

    Attributes:
        start_time: Wall-clock start time
        end_time: Wall-clock end time
        steps_simulated: Number of simulation steps
        simulated_duration: Simulated time units
    """

    start_time: float
    end_time: float
    steps_simulated: int
    simulated_duration: float

    @property
    def wall_clock_duration(self) -> float:
        """Get wall-clock duration in seconds."""
        return self.end_time - self.start_time

    @property
    def compression_ratio(self) -> float:
        """Calculate temporal compression ratio."""
        if self.wall_clock_duration <= 0:
            return 0.0
        return self.simulated_duration / self.wall_clock_duration

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "steps_simulated": self.steps_simulated,
            "simulated_duration": self.simulated_duration,
            "wall_clock_duration": self.wall_clock_duration,
            "compression_ratio": self.compression_ratio,
        }


@dataclass
class CompressionStatistics:
    """Aggregated compression statistics across multiple runs.

    Attributes:
        measurements: Individual timing measurements
        mean_compression_ratio: Mean compression ratio
        std_compression_ratio: Standard deviation of compression ratio
        min_compression_ratio: Minimum observed ratio
        max_compression_ratio: Maximum observed ratio
        total_steps: Total steps simulated across all runs
        total_wall_time: Total wall-clock time
        reproducibility_verified: Whether all runs were bitwise identical
    """

    measurements: list[TimingMeasurement] = field(default_factory=list)
    mean_compression_ratio: float = 0.0
    std_compression_ratio: float = 0.0
    min_compression_ratio: float = 0.0
    max_compression_ratio: float = 0.0
    total_steps: int = 0
    total_wall_time: float = 0.0
    reproducibility_verified: bool = False

    def add_measurement(self, measurement: TimingMeasurement) -> None:
        """Add a timing measurement and update statistics."""
        self.measurements.append(measurement)
        self._update_statistics()

    def _update_statistics(self) -> None:
        """Recalculate aggregate statistics."""
        if not self.measurements:
            return

        ratios = [m.compression_ratio for m in self.measurements]
        self.mean_compression_ratio = float(np.mean(ratios))
        self.std_compression_ratio = float(np.std(ratios))
        self.min_compression_ratio = float(np.min(ratios))
        self.max_compression_ratio = float(np.max(ratios))
        self.total_steps = sum(m.steps_simulated for m in self.measurements)
        self.total_wall_time = sum(m.wall_clock_duration for m in self.measurements)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "measurements": [m.to_dict() for m in self.measurements],
            "mean_compression_ratio": self.mean_compression_ratio,
            "std_compression_ratio": self.std_compression_ratio,
            "min_compression_ratio": self.min_compression_ratio,
            "max_compression_ratio": self.max_compression_ratio,
            "total_steps": self.total_steps,
            "total_wall_time": self.total_wall_time,
            "reproducibility_verified": self.reproducibility_verified,
        }


@dataclass
class VerificationOverhead:
    """Metrics for verification overhead.

    Attributes:
        hash_computation_time: Time spent computing hashes
        merkle_update_time: Time spent updating Merkle chain
        total_overhead_time: Total verification overhead
        overhead_percentage: Overhead as percentage of total time
    """

    hash_computation_time: float = 0.0
    merkle_update_time: float = 0.0
    total_overhead_time: float = 0.0
    overhead_percentage: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hash_computation_time": self.hash_computation_time,
            "merkle_update_time": self.merkle_update_time,
            "total_overhead_time": self.total_overhead_time,
            "overhead_percentage": self.overhead_percentage,
        }


@dataclass
class ScalingMetrics:
    """Metrics for scaling behavior analysis.

    Attributes:
        state_sizes: List of state sizes tested
        compression_ratios: Compression ratio at each size
        times_per_step: Wall-clock time per step at each size
        memory_usage: Memory usage at each size
    """

    state_sizes: list[int] = field(default_factory=list)
    compression_ratios: list[float] = field(default_factory=list)
    times_per_step: list[float] = field(default_factory=list)
    memory_usage: list[int] = field(default_factory=list)

    def add_data_point(
        self,
        size: int,
        compression_ratio: float,
        time_per_step: float,
        memory: int = 0,
    ) -> None:
        """Add scaling measurement."""
        self.state_sizes.append(size)
        self.compression_ratios.append(compression_ratio)
        self.times_per_step.append(time_per_step)
        self.memory_usage.append(memory)

    def get_complexity_estimate(self) -> str:
        """Estimate computational complexity from measurements."""
        if len(self.state_sizes) < 2:
            return "insufficient_data"

        # Simple log-log regression for complexity estimation
        sizes = np.array(self.state_sizes, dtype=float)
        times = np.array(self.times_per_step, dtype=float)

        # Filter out zero times
        mask = times > 0
        if np.sum(mask) < 2:
            return "insufficient_data"

        log_sizes = np.log(sizes[mask])
        log_times = np.log(times[mask])

        # Linear regression in log-log space
        slope, _ = np.polyfit(log_sizes, log_times, 1)

        if slope < 1.2:
            return "O(n)"
        elif slope < 1.8:
            return "O(n log n)"
        elif slope < 2.5:
            return "O(n²)"
        else:
            return f"O(n^{slope:.1f})"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "state_sizes": self.state_sizes,
            "compression_ratios": self.compression_ratios,
            "times_per_step": self.times_per_step,
            "memory_usage": self.memory_usage,
            "complexity_estimate": self.get_complexity_estimate(),
        }


class TemporalMetrics:
    """Comprehensive temporal compression metrics calculator.

    Measures and calculates all metrics required for experimental
    validation of deterministic temporal compression.

    Example:
        metrics = TemporalMetrics()
        metrics.start_measurement()

        for _ in range(1000):
            automaton.evolve(1)

        metrics.end_measurement(steps=1000, simulated_time=1000.0)
        print(f"Compression ratio: {metrics.get_compression_ratio():.2f}×")
    """

    def __init__(self, time_unit: str = "step") -> None:
        """Initialize metrics calculator.

        Args:
            time_unit: Unit of simulated time (e.g., 'step', 'second')
        """
        self.time_unit = time_unit
        self.statistics = CompressionStatistics()
        self.verification_overhead = VerificationOverhead()
        self.scaling = ScalingMetrics()

        self._current_start: float | None = None
        self._hash_times: list[float] = []
        self._merkle_times: list[float] = []

    def start_measurement(self) -> None:
        """Start a timing measurement."""
        self._current_start = time.perf_counter()

    def end_measurement(
        self,
        steps: int,
        simulated_time: float | None = None,
    ) -> TimingMeasurement:
        """End timing measurement and record results.

        Args:
            steps: Number of simulation steps completed
            simulated_time: Simulated time duration (defaults to steps)

        Returns:
            TimingMeasurement with results
        """
        if self._current_start is None:
            raise RuntimeError("No measurement in progress")

        end_time = time.perf_counter()
        simulated_time = simulated_time if simulated_time is not None else float(steps)

        measurement = TimingMeasurement(
            start_time=self._current_start,
            end_time=end_time,
            steps_simulated=steps,
            simulated_duration=simulated_time,
        )

        self.statistics.add_measurement(measurement)
        self._current_start = None

        return measurement

    def record_hash_time(self, duration: float) -> None:
        """Record time spent on hash computation."""
        self._hash_times.append(duration)
        self.verification_overhead.hash_computation_time = sum(self._hash_times)
        self._update_overhead()

    def record_merkle_time(self, duration: float) -> None:
        """Record time spent on Merkle chain operations."""
        self._merkle_times.append(duration)
        self.verification_overhead.merkle_update_time = sum(self._merkle_times)
        self._update_overhead()

    def _update_overhead(self) -> None:
        """Update total overhead calculations."""
        total_overhead = (
            self.verification_overhead.hash_computation_time
            + self.verification_overhead.merkle_update_time
        )
        self.verification_overhead.total_overhead_time = total_overhead

        if self.statistics.total_wall_time > 0:
            self.verification_overhead.overhead_percentage = (
                100.0 * total_overhead / self.statistics.total_wall_time
            )

    def get_compression_ratio(self) -> float:
        """Get mean compression ratio across all measurements."""
        return self.statistics.mean_compression_ratio

    def verify_reproducibility(self, state_hashes: list[str]) -> bool:
        """Verify reproducibility across multiple runs.

        Args:
            state_hashes: List of final state hashes from multiple runs

        Returns:
            True if all hashes match (bitwise reproducibility)
        """
        if not state_hashes:
            return False

        first_hash = state_hashes[0]
        self.statistics.reproducibility_verified = all(h == first_hash for h in state_hashes)
        return self.statistics.reproducibility_verified

    def get_summary(self) -> dict[str, Any]:
        """Get comprehensive metrics summary.

        Returns:
            Dictionary with all metrics suitable for reporting
        """
        return {
            "temporal_compression": {
                "mean_ratio": self.statistics.mean_compression_ratio,
                "std_ratio": self.statistics.std_compression_ratio,
                "min_ratio": self.statistics.min_compression_ratio,
                "max_ratio": self.statistics.max_compression_ratio,
                "interpretation": self._interpret_compression_ratio(),
            },
            "timing": {
                "total_steps": self.statistics.total_steps,
                "total_wall_time": self.statistics.total_wall_time,
                "time_unit": self.time_unit,
                "num_measurements": len(self.statistics.measurements),
            },
            "verification": self.verification_overhead.to_dict(),
            "reproducibility": {
                "verified": self.statistics.reproducibility_verified,
            },
            "scaling": self.scaling.to_dict(),
        }

    def _interpret_compression_ratio(self) -> str:
        """Interpret the compression ratio for the report."""
        ratio = self.statistics.mean_compression_ratio

        if ratio <= 0:
            return "No compression data available"
        elif ratio < 1:
            return f"Simulation slower than real-time ({ratio:.2f}×)"
        elif ratio < 10:
            return f"Moderate temporal compression ({ratio:.2f}×)"
        elif ratio < 100:
            return f"Significant temporal compression ({ratio:.2f}×)"
        elif ratio < 1000:
            return f"High temporal compression ({ratio:.2f}×)"
        else:
            return f"Extreme temporal compression ({ratio:.2f}×)"

    def to_dict(self) -> dict[str, Any]:
        """Convert all metrics to dictionary."""
        return {
            "statistics": self.statistics.to_dict(),
            "verification_overhead": self.verification_overhead.to_dict(),
            "scaling": self.scaling.to_dict(),
            "summary": self.get_summary(),
        }
