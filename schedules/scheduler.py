"""Differentiable scheduling with gradient-based optimization."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ScheduleParams:
    """Differentiable scheduling parameters."""

    block_size: float = 256.0
    thread_count: float = 128.0
    register_pressure: float = 32.0
    memory_coalesce_factor: float = 1.0
    prefetch_distance: float = 2.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "block_size": self.block_size,
            "thread_count": self.thread_count,
            "register_pressure": self.register_pressure,
            "memory_coalesce_factor": self.memory_coalesce_factor,
            "prefetch_distance": self.prefetch_distance,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ScheduleParams:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ScheduleMetadata:
    """Metadata for an optimized schedule."""

    schedule_id: str
    params: ScheduleParams
    latency_ms: float = 0.0
    energy_j: float = 0.0
    loss_value: float = 0.0
    optimization_steps: int = 0
    timestamp: float = field(default_factory=time.time)
    benchmark_trace: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "schedule_id": self.schedule_id,
            "params": self.params.to_dict(),
            "latency_ms": self.latency_ms,
            "energy_j": self.energy_j,
            "loss_value": self.loss_value,
            "optimization_steps": self.optimization_steps,
            "timestamp": self.timestamp,
            "benchmark_trace": self.benchmark_trace,
        }


class DifferentiableScheduler:
    """Scheduler with differentiable parameters for gradient-based optimization."""

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.schedules: dict[str, ScheduleMetadata] = {}

    def compute_latency_loss(self, params: ScheduleParams) -> float:
        """
        Compute latency loss function (differentiable approximation).
        Lower is better.
        """
        # Simplified model: latency depends on block size and thread count
        # Real implementation would use actual hardware measurements

        # Normalize parameters
        norm_block_size = params.block_size / 256.0
        norm_threads = params.thread_count / 128.0

        # Latency increases with smaller blocks (more overhead)
        # and with suboptimal thread count
        block_penalty = 1.0 / (norm_block_size + 0.1)
        thread_penalty = abs(norm_threads - 1.0) + 0.5

        # Register pressure penalty
        register_penalty = max(0.0, (params.register_pressure - 32.0) / 32.0)

        # Memory coalesce bonus (lower loss for better coalescing)
        coalesce_bonus = 1.0 / (params.memory_coalesce_factor + 0.1)

        latency_loss = (
            block_penalty * 0.3
            + thread_penalty * 0.3
            + register_penalty * 0.2
            + coalesce_bonus * 0.2
        )

        return latency_loss

    def compute_energy_loss(self, params: ScheduleParams) -> float:
        """
        Compute energy loss function (differentiable approximation).
        Lower is better.
        """
        # Energy increases with thread count and register pressure
        norm_threads = params.thread_count / 128.0
        norm_registers = params.register_pressure / 32.0

        energy_loss = (
            norm_threads * 0.5
            + norm_registers * 0.3
            + (1.0 / (params.memory_coalesce_factor + 0.1)) * 0.2
        )

        return energy_loss

    def compute_gradients(self, params: ScheduleParams, epsilon: float = 1e-5) -> dict[str, float]:
        """
        Compute numerical gradients for all parameters.
        Uses finite differences.
        """

        # Combined loss (weighted sum of latency and energy)
        def loss_fn(p: ScheduleParams) -> float:
            return self.compute_latency_loss(p) + 0.5 * self.compute_energy_loss(p)

        base_loss = loss_fn(params)
        gradients = {}

        # Compute gradient for block_size
        params_copy = ScheduleParams(**params.to_dict())
        params_copy.block_size += epsilon
        gradients["block_size"] = (loss_fn(params_copy) - base_loss) / epsilon

        # Compute gradient for thread_count
        params_copy = ScheduleParams(**params.to_dict())
        params_copy.thread_count += epsilon
        gradients["thread_count"] = (loss_fn(params_copy) - base_loss) / epsilon

        # Compute gradient for register_pressure
        params_copy = ScheduleParams(**params.to_dict())
        params_copy.register_pressure += epsilon
        gradients["register_pressure"] = (loss_fn(params_copy) - base_loss) / epsilon

        # Compute gradient for memory_coalesce_factor
        params_copy = ScheduleParams(**params.to_dict())
        params_copy.memory_coalesce_factor += epsilon
        gradients["memory_coalesce_factor"] = (loss_fn(params_copy) - base_loss) / epsilon

        # Compute gradient for prefetch_distance
        params_copy = ScheduleParams(**params.to_dict())
        params_copy.prefetch_distance += epsilon
        gradients["prefetch_distance"] = (loss_fn(params_copy) - base_loss) / epsilon

        return gradients

    def optimize_schedule(
        self, schedule_id: str, initial_params: ScheduleParams | None = None, steps: int = 100
    ) -> ScheduleMetadata:
        """
        Optimize schedule parameters using gradient descent.
        """
        params = initial_params or ScheduleParams()

        for step in range(steps):
            # Compute gradients
            gradients = self.compute_gradients(params)

            # Update parameters (gradient descent)
            params.block_size -= self.learning_rate * gradients["block_size"]
            params.thread_count -= self.learning_rate * gradients["thread_count"]
            params.register_pressure -= self.learning_rate * gradients["register_pressure"]
            params.memory_coalesce_factor -= (
                self.learning_rate * gradients["memory_coalesce_factor"]
            )
            params.prefetch_distance -= self.learning_rate * gradients["prefetch_distance"]

            # Clamp to valid ranges
            params.block_size = max(64.0, min(1024.0, params.block_size))
            params.thread_count = max(32.0, min(512.0, params.thread_count))
            params.register_pressure = max(8.0, min(64.0, params.register_pressure))
            params.memory_coalesce_factor = max(0.5, min(4.0, params.memory_coalesce_factor))
            params.prefetch_distance = max(1.0, min(8.0, params.prefetch_distance))

        # Compute final metrics
        latency_loss = self.compute_latency_loss(params)
        energy_loss = self.compute_energy_loss(params)
        total_loss = latency_loss + 0.5 * energy_loss

        metadata = ScheduleMetadata(
            schedule_id=schedule_id,
            params=params,
            latency_ms=latency_loss * 100.0,  # Scale to realistic ms
            energy_j=energy_loss * 10.0,  # Scale to realistic J
            loss_value=total_loss,
            optimization_steps=steps,
            benchmark_trace={
                "latency_loss": latency_loss,
                "energy_loss": energy_loss,
                "total_loss": total_loss,
            },
        )

        self.schedules[schedule_id] = metadata
        return metadata

    def save_schedule(self, schedule_id: str, output_dir: str = "schedules") -> Path:
        """Save optimized schedule to disk."""
        if schedule_id not in self.schedules:
            raise ValueError(f"No schedule found for {schedule_id}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        schedule_file = output_path / f"{schedule_id}.json"
        metadata = self.schedules[schedule_id]

        with open(schedule_file, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        return schedule_file
