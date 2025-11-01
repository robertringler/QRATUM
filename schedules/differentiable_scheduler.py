"""Differentiable compiler scheduling system.

Makes scheduling parameters differentiable and uses gradient descent on latency
or energy loss functions to optimize scheduling decisions.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class ScheduleParameter:
    """A differentiable scheduling parameter."""
    
    name: str
    value: float
    gradient: float = 0.0
    min_value: float = 0.0
    max_value: float = 1.0
    learning_rate: float = 0.01
    
    def update(self) -> None:
        """Update parameter value using gradient descent."""
        self.value = max(self.min_value, min(self.max_value, 
                        self.value - self.learning_rate * self.gradient))
        self.gradient = 0.0  # Reset gradient


@dataclass
class Schedule:
    """Represents a complete kernel schedule with metadata."""
    
    schedule_id: str
    kernel_id: str
    parameters: Dict[str, ScheduleParameter]
    latency_ms: float = 0.0
    energy_j: float = 0.0
    loss: float = 0.0
    iteration: int = 0
    metadata: Dict[str, any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert schedule to dictionary."""
        return {
            "schedule_id": self.schedule_id,
            "kernel_id": self.kernel_id,
            "parameters": {
                name: {
                    "value": param.value,
                    "min": param.min_value,
                    "max": param.max_value
                }
                for name, param in self.parameters.items()
            },
            "latency_ms": self.latency_ms,
            "energy_j": self.energy_j,
            "loss": self.loss,
            "iteration": self.iteration,
            "metadata": self.metadata
        }


class DifferentiableScheduler:
    """Scheduler that optimizes kernel scheduling using gradient descent."""
    
    def __init__(self, schedules_dir: Optional[Path] = None):
        """Initialize differentiable scheduler.
        
        Args:
            schedules_dir: Directory to store optimized schedules
        """
        self.schedules_dir = schedules_dir or Path(__file__).parent
        self.schedules_dir.mkdir(parents=True, exist_ok=True)
        self._active_schedules: Dict[str, Schedule] = {}
        
    def create_schedule(self, schedule_id: str, kernel_id: str) -> Schedule:
        """Create a new differentiable schedule.
        
        Args:
            schedule_id: Unique identifier for schedule
            kernel_id: Associated kernel identifier
            
        Returns:
            New Schedule instance
        """
        # Create default scheduling parameters
        parameters = {
            "tile_parallelism": ScheduleParameter(
                name="tile_parallelism",
                value=0.5,
                min_value=0.0,
                max_value=1.0,
                learning_rate=0.01
            ),
            "memory_coalescing": ScheduleParameter(
                name="memory_coalescing",
                value=0.7,
                min_value=0.0,
                max_value=1.0,
                learning_rate=0.01
            ),
            "loop_unrolling": ScheduleParameter(
                name="loop_unrolling",
                value=0.4,
                min_value=0.0,
                max_value=1.0,
                learning_rate=0.01
            ),
            "vectorization_width": ScheduleParameter(
                name="vectorization_width",
                value=0.6,
                min_value=0.0,
                max_value=1.0,
                learning_rate=0.01
            )
        }
        
        schedule = Schedule(
            schedule_id=schedule_id,
            kernel_id=kernel_id,
            parameters=parameters
        )
        
        self._active_schedules[schedule_id] = schedule
        return schedule
    
    def compute_loss(self, schedule: Schedule, target_latency: float = 10.0,
                    energy_weight: float = 0.3) -> float:
        """Compute loss function for schedule optimization.
        
        Args:
            schedule: Schedule to evaluate
            target_latency: Target latency in milliseconds
            energy_weight: Weight for energy term in loss
            
        Returns:
            Loss value (lower is better)
        """
        # Simulate latency based on parameters
        # In practice, this would be measured from actual execution
        param_vector = [p.value for p in schedule.parameters.values()]
        
        # Simple synthetic model: latency depends on parallelism and coalescing
        parallelism = schedule.parameters["tile_parallelism"].value
        coalescing = schedule.parameters["memory_coalescing"].value
        
        # Higher parallelism and coalescing reduce latency
        simulated_latency = target_latency * (1.5 - 0.5 * parallelism - 0.3 * coalescing)
        
        # Energy depends on unrolling and vectorization
        unrolling = schedule.parameters["loop_unrolling"].value
        vectorization = schedule.parameters["vectorization_width"].value
        simulated_energy = 1.0 + 0.5 * unrolling + 0.3 * vectorization
        
        schedule.latency_ms = simulated_latency
        schedule.energy_j = simulated_energy
        
        # Combined loss: latency + energy
        latency_loss = (simulated_latency / target_latency) ** 2
        energy_loss = simulated_energy ** 2
        
        loss = latency_loss + energy_weight * energy_loss
        schedule.loss = loss
        
        return loss
    
    def compute_gradients(self, schedule: Schedule, epsilon: float = 1e-4) -> None:
        """Compute gradients using finite differences.
        
        Args:
            schedule: Schedule to compute gradients for
            epsilon: Small value for finite difference approximation
        """
        baseline_loss = schedule.loss
        
        # Compute gradient for each parameter
        for param in schedule.parameters.values():
            # Perturb parameter
            original_value = param.value
            param.value = min(param.max_value, original_value + epsilon)
            
            # Compute loss with perturbation
            perturbed_loss = self.compute_loss(schedule)
            
            # Estimate gradient
            param.gradient = (perturbed_loss - baseline_loss) / epsilon
            
            # Restore original value
            param.value = original_value
        
        # Restore baseline loss
        schedule.loss = baseline_loss
    
    def optimize_schedule(self, schedule: Schedule, iterations: int = 100,
                         target_latency: float = 10.0) -> Schedule:
        """Optimize schedule using gradient descent.
        
        Args:
            schedule: Schedule to optimize
            iterations: Number of optimization iterations
            target_latency: Target latency for loss computation
            
        Returns:
            Optimized schedule
        """
        for i in range(iterations):
            # Compute loss and gradients
            self.compute_loss(schedule, target_latency)
            self.compute_gradients(schedule)
            
            # Update parameters
            for param in schedule.parameters.values():
                param.update()
            
            schedule.iteration = i + 1
            
            # Log progress periodically
            if (i + 1) % 20 == 0:
                print(f"Iteration {i+1}/{iterations}: Loss = {schedule.loss:.6f}, "
                      f"Latency = {schedule.latency_ms:.3f} ms")
        
        return schedule
    
    def save_schedule(self, schedule_id: str) -> Path:
        """Save optimized schedule to disk.
        
        Args:
            schedule_id: Schedule identifier
            
        Returns:
            Path to saved schedule
        """
        if schedule_id not in self._active_schedules:
            raise ValueError(f"No schedule found: {schedule_id}")
        
        schedule = self._active_schedules[schedule_id]
        schedule_path = self.schedules_dir / f"{schedule_id}.json"
        
        with open(schedule_path, 'w') as f:
            json.dump(schedule.to_dict(), f, indent=2)
        
        return schedule_path
    
    def load_schedule(self, schedule_id: str) -> Optional[Schedule]:
        """Load schedule from disk.
        
        Args:
            schedule_id: Schedule identifier
            
        Returns:
            Loaded Schedule or None if not found
        """
        schedule_path = self.schedules_dir / f"{schedule_id}.json"
        if not schedule_path.exists():
            return None
        
        with open(schedule_path, 'r') as f:
            data = json.load(f)
        
        # Reconstruct schedule
        parameters = {}
        for name, param_data in data["parameters"].items():
            parameters[name] = ScheduleParameter(
                name=name,
                value=param_data["value"],
                min_value=param_data["min"],
                max_value=param_data["max"]
            )
        
        schedule = Schedule(
            schedule_id=data["schedule_id"],
            kernel_id=data["kernel_id"],
            parameters=parameters,
            latency_ms=data["latency_ms"],
            energy_j=data["energy_j"],
            loss=data["loss"],
            iteration=data["iteration"],
            metadata=data.get("metadata", {})
        )
        
        self._active_schedules[schedule_id] = schedule
        return schedule


if __name__ == "__main__":
    print("Differentiable Scheduler Demo")
    print("=" * 60)
    
    # Initialize scheduler
    scheduler = DifferentiableScheduler()
    
    # Create schedule
    schedule_id = "matmul_schedule_001"
    kernel_id = "matmul_kernel_001"
    schedule = scheduler.create_schedule(schedule_id, kernel_id)
    
    print(f"\nInitial Schedule Parameters:")
    for name, param in schedule.parameters.items():
        print(f"  {name}: {param.value:.3f}")
    
    # Optimize schedule
    print("\n" + "=" * 60)
    print("Optimizing schedule...")
    optimized = scheduler.optimize_schedule(schedule, iterations=100, target_latency=10.0)
    
    print("\n" + "=" * 60)
    print(f"Optimized Schedule Parameters:")
    for name, param in optimized.parameters.items():
        print(f"  {name}: {param.value:.3f}")
    
    print(f"\nFinal Metrics:")
    print(f"  Latency: {optimized.latency_ms:.3f} ms")
    print(f"  Energy:  {optimized.energy_j:.3f} J")
    print(f"  Loss:    {optimized.loss:.6f}")
    
    # Save schedule
    schedule_path = scheduler.save_schedule(schedule_id)
    print(f"\nSchedule saved to: {schedule_path}")
