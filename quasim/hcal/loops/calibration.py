"""Calibration loop framework for HCAL."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Tuple

import numpy as np


class ConvergenceStatus(Enum):
    """Calibration convergence status."""

    CONVERGED = "converged"
    IN_PROGRESS = "in_progress"
    DIVERGED = "diverged"
    MAX_ITERATIONS = "max_iterations"


@dataclass
class CalibrationResult:
    """Result of a calibration run."""

    status: ConvergenceStatus
    final_setpoint: Dict[str, Any]
    iterations: int
    objective_history: List[float]
    best_objective: float


class CalibrationLoop:
    """Closed-loop calibration framework."""

    def __init__(
        self,
        measure_fn: Callable[[str, Any], Dict[str, Any]],
        apply_fn: Callable[[str, Dict[str, Any], Any], bool],
        objective_fn: Callable[[Dict[str, Any]], float],
        convergence_threshold: float = 0.01,
        max_iterations: int = 100,
    ):
        """Initialize calibration loop.

        Args:
            measure_fn: Function to measure current state.
            apply_fn: Function to apply setpoint.
            objective_fn: Function to compute objective value (lower is better).
            convergence_threshold: Convergence threshold for objective.
            max_iterations: Maximum number of iterations.
        """
        self.measure_fn = measure_fn
        self.apply_fn = apply_fn
        self.objective_fn = objective_fn
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations

    def run(
        self,
        device_id: str,
        backend: Any,
        initial_setpoint: Dict[str, Any],
        optimizer: str = "pid",
    ) -> CalibrationResult:
        """Run calibration loop.

        Args:
            device_id: Device identifier.
            backend: Backend driver instance.
            initial_setpoint: Initial setpoint.
            optimizer: Optimizer type ('pid', 'gradient', 'mppi').

        Returns:
            CalibrationResult instance.
        """
        setpoint = initial_setpoint.copy()
        objective_history = []
        best_objective = float("inf")
        best_setpoint = setpoint.copy()

        for iteration in range(self.max_iterations):
            # Apply setpoint
            success = self.apply_fn(device_id, setpoint, backend)
            if not success:
                return CalibrationResult(
                    status=ConvergenceStatus.DIVERGED,
                    final_setpoint=best_setpoint,
                    iterations=iteration,
                    objective_history=objective_history,
                    best_objective=best_objective,
                )

            # Measure state
            state = self.measure_fn(device_id, backend)

            # Compute objective
            objective = self.objective_fn(state)
            objective_history.append(objective)

            # Update best
            if objective < best_objective:
                best_objective = objective
                best_setpoint = setpoint.copy()

            # Check convergence
            if len(objective_history) >= 2:
                improvement = abs(objective_history[-2] - objective_history[-1])
                if improvement < self.convergence_threshold:
                    return CalibrationResult(
                        status=ConvergenceStatus.CONVERGED,
                        final_setpoint=best_setpoint,
                        iterations=iteration + 1,
                        objective_history=objective_history,
                        best_objective=best_objective,
                    )

            # Update setpoint using optimizer
            if optimizer == "pid":
                setpoint = self._pid_update(setpoint, state, objective)
            elif optimizer == "gradient":
                setpoint = self._gradient_update(setpoint, state, objective_history)
            elif optimizer == "mppi":
                setpoint = self._mppi_update(setpoint, state, backend, device_id)
            else:
                raise ValueError(f"Unknown optimizer: {optimizer}")

        return CalibrationResult(
            status=ConvergenceStatus.MAX_ITERATIONS,
            final_setpoint=best_setpoint,
            iterations=self.max_iterations,
            objective_history=objective_history,
            best_objective=best_objective,
        )

    def _pid_update(
        self, setpoint: Dict[str, Any], state: Dict[str, Any], objective: float
    ) -> Dict[str, Any]:
        """Update setpoint using PID control.

        Args:
            setpoint: Current setpoint.
            state: Current state.
            objective: Current objective value.

        Returns:
            Updated setpoint.
        """
        # Simple proportional control
        new_setpoint = setpoint.copy()

        # Adjust first parameter (simplified)
        if setpoint:
            key = list(setpoint.keys())[0]
            current_value = setpoint[key]
            # Decrease by small amount to minimize objective
            new_setpoint[key] = current_value * 0.95

        return new_setpoint

    def _gradient_update(
        self, setpoint: Dict[str, Any], state: Dict[str, Any], objective_history: List[float]
    ) -> Dict[str, Any]:
        """Update setpoint using gradient descent.

        Args:
            setpoint: Current setpoint.
            state: Current state.
            objective_history: History of objective values.

        Returns:
            Updated setpoint.
        """
        new_setpoint = setpoint.copy()

        if len(objective_history) >= 2:
            # Estimate gradient
            gradient = objective_history[-1] - objective_history[-2]

            # Update with small step
            if setpoint:
                key = list(setpoint.keys())[0]
                current_value = setpoint[key]
                step = -0.01 * gradient * current_value
                new_setpoint[key] = current_value + step

        return new_setpoint

    def _mppi_update(
        self, setpoint: Dict[str, Any], state: Dict[str, Any], backend: Any, device_id: str
    ) -> Dict[str, Any]:
        """Update setpoint using MPPI-style optimization.

        Args:
            setpoint: Current setpoint.
            state: Current state.
            backend: Backend driver instance.
            device_id: Device identifier.

        Returns:
            Updated setpoint.
        """
        # Sample perturbations
        num_samples = 10
        samples = []
        costs = []

        for _ in range(num_samples):
            sample = setpoint.copy()

            # Add noise to each parameter
            for key in sample:
                if isinstance(sample[key], (int, float)):
                    noise = np.random.normal(0, 0.05 * sample[key])
                    sample[key] = sample[key] + noise

            samples.append(sample)

            # Evaluate cost (in dry-run, use random cost)
            cost = np.random.random()
            costs.append(cost)

        # Weight samples by cost
        costs = np.array(costs)
        weights = np.exp(-10 * costs)
        weights = weights / np.sum(weights)

        # Compute weighted average
        new_setpoint = {}
        for key in setpoint:
            if isinstance(setpoint[key], (int, float)):
                weighted_sum = sum(w * s[key] for w, s in zip(weights, samples))
                new_setpoint[key] = weighted_sum
            else:
                new_setpoint[key] = setpoint[key]

        return new_setpoint


def bias_trim_v1(
    device_id: str, backend: Any, measure_fn: Callable, apply_fn: Callable
) -> CalibrationResult:
    """Bias trim calibration routine v1.

    Args:
        device_id: Device identifier.
        backend: Backend driver instance.
        measure_fn: Measurement function.
        apply_fn: Apply function.

    Returns:
        CalibrationResult instance.
    """

    def objective_fn(state: Dict[str, Any]) -> float:
        """Minimize power while maintaining performance."""
        power = state.get("power_watts", 0)
        utilization = state.get("utilization_percent", 0)

        # Penalty for low utilization
        if utilization < 50:
            return power + 1000

        return power

    loop = CalibrationLoop(
        measure_fn=measure_fn,
        apply_fn=apply_fn,
        objective_fn=objective_fn,
        convergence_threshold=1.0,
        max_iterations=20,
    )

    initial_setpoint = {"power_limit_watts": 250.0}

    return loop.run(device_id, backend, initial_setpoint, optimizer="pid")


def power_sweep(
    device_id: str,
    backend: Any,
    measure_fn: Callable,
    apply_fn: Callable,
    power_range: Tuple[float, float],
    steps: int = 10,
) -> List[Tuple[float, Dict[str, Any]]]:
    """Power sweep calibration routine.

    Args:
        device_id: Device identifier.
        backend: Backend driver instance.
        measure_fn: Measurement function.
        apply_fn: Apply function.
        power_range: (min_power, max_power) in watts.
        steps: Number of power levels to test.

    Returns:
        List of (power, telemetry) tuples.
    """
    results = []

    power_levels = np.linspace(power_range[0], power_range[1], steps)

    for power in power_levels:
        setpoint = {"power_limit_watts": power}
        success = apply_fn(device_id, setpoint, backend)

        if success:
            state = measure_fn(device_id, backend)
            results.append((power, state))

    return results
