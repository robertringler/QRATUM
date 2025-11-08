"""QCMG (Quantacosmorphysigenetic) Field Simulation Module.

This module provides simulation capabilities for quantum field dynamics
with tracking of coherence, entropy, and energy evolution.
"""QuASIM Simulation Module.

This module provides field simulation capabilities including the
Quantacosmorphysigenetic (QCMG) field evolution system.
"""QCMG - Quantacosmomorphysigenetic Field Simulation Module.

This module provides simulation capabilities for coupled quantum-classical
field dynamics based on the Quantacosmomorphysigenetic (QCMG) model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


@dataclass
class QCMGParameters:
    """Configuration parameters for QCMG field simulation.

    Attributes:
        grid_size: Size of the spatial grid (NxN)
        dt: Time step for evolution
        coupling_strength: Strength of field coupling
        interaction_strength: Strength of field interactions
        thermal_noise: Thermal noise amplitude
        random_seed: Random seed for reproducibility
    """

    grid_size: int = 64
    dt: float = 0.01
    coupling_strength: float = 0.1
    interaction_strength: float = 0.05
    thermal_noise: float = 0.001
    random_seed: int | None = None


@dataclass
class FieldState:
    """State snapshot of the QCMG field at a point in time.

    Attributes:
        time: Current simulation time
        coherence: Quantum coherence measure (0 to 1)
        entropy: System entropy
        energy: Total field energy
    """

    time: float
    coherence: float
    entropy: float
    energy: float


class QuantacosmorphysigeneticField:
    """QCMG field simulation with quantum dynamics.

    Simulates the evolution of a quantacosmorphysigenetic field with
    tracking of coherence, entropy, and energy dynamics over time.

    Attributes:
        params: Simulation parameters
    """

    def __init__(self, params: QCMGParameters):
        """Initialize the QCMG field.

        Args:
            params: Simulation parameters
        """
        self.params = params
        self._time = 0.0
        self._history: list[FieldState] = []
        self._field: np.ndarray | None = None

        # Set random seed if provided
        if params.random_seed is not None:
            np.random.seed(params.random_seed)

    def initialize(self, mode: Literal["gaussian", "uniform", "zero"] = "gaussian") -> None:
        """Initialize the field with a specific configuration.

        Args:
            mode: Initialization mode:
                - "gaussian": Gaussian distribution centered in the field
                - "uniform": Uniform random distribution
                - "zero": All zeros
        """
        size = self.params.grid_size

        if mode == "gaussian":
            # Create Gaussian distribution centered in the grid
            x = np.linspace(-3, 3, size)
            y = np.linspace(-3, 3, size)
            x_grid, y_grid = np.meshgrid(x, y)
            self._field = np.exp(-(x_grid**2 + y_grid**2) / 2.0)

        elif mode == "uniform":
            # Uniform random distribution
            self._field = np.random.uniform(0, 1, (size, size))

        elif mode == "zero":
            # Zero field
            self._field = np.zeros((size, size))

        else:
            raise ValueError(f"Unknown initialization mode: {mode}")

        # Record initial state
        self._time = 0.0
        self._record_state()

    def evolve(self) -> FieldState:
        """Evolve the field by one time step.

        Returns:
            Current state after evolution
        """
        if self._field is None:
            raise RuntimeError("Field not initialized. Call initialize() first.")

        # Apply field evolution using quantum dynamics
        dt = self.params.dt
        coupling = self.params.coupling_strength
        interaction = self.params.interaction_strength
        noise = self.params.thermal_noise

        # Compute Laplacian for diffusion-like dynamics
        laplacian = self._compute_laplacian(self._field)

        # Evolution: field dynamics with coupling and interactions
        field_evolution = (
            coupling * laplacian
            - interaction * self._field**3
            + noise * np.random.randn(*self._field.shape)
        )

        # Update field
        self._field += dt * field_evolution

        # Normalize to maintain stability
        field_norm = np.sqrt(np.sum(self._field**2))
        if field_norm > 0:
            self._field = self._field / field_norm * self.params.grid_size

        # Update time
        self._time += dt

        # Record state and return
        self._record_state()
        return self._history[-1]

    def get_state(self) -> FieldState:
        """Get the current field state.

        Returns:
            Current field state
        """
        if not self._history:
            raise RuntimeError("No state available. Initialize the field first.")
        return self._history[-1]

    def get_history(self) -> list[FieldState]:
        """Get the complete evolution history.

        Returns:
            List of all recorded states
        """
        return self._history.copy()

    def export_state(self) -> dict[str, Any]:
        """Export current state and parameters to dictionary.

        Returns:
            Dictionary with simulation parameters and current state
        """
        current_state = self.get_state()

        return {
            "parameters": {
                "grid_size": self.params.grid_size,
                "dt": self.params.dt,
                "coupling_strength": self.params.coupling_strength,
                "interaction_strength": self.params.interaction_strength,
                "thermal_noise": self.params.thermal_noise,
                "random_seed": self.params.random_seed,
            },
            "state": {
                "time": current_state.time,
                "coherence": current_state.coherence,
                "entropy": current_state.entropy,
                "energy": current_state.energy,
            },
            "history_length": len(self._history),
        }

    def _compute_laplacian(self, field_array: np.ndarray) -> np.ndarray:
        """Compute discrete Laplacian for field evolution.

        Args:
            field_array: 2D field array

        Returns:
            Laplacian of the field
        """
        # Simple 5-point stencil for Laplacian
        laplacian = np.zeros_like(field_array)
        laplacian[1:-1, 1:-1] = (
            field_array[:-2, 1:-1]  # up
            + field_array[2:, 1:-1]  # down
            + field_array[1:-1, :-2]  # left
            + field_array[1:-1, 2:]  # right
            - 4 * field_array[1:-1, 1:-1]  # center
        )
        return laplacian

    def _record_state(self) -> None:
        """Record current field state in history."""
        if self._field is None:
            raise RuntimeError("Field not initialized")

        # Calculate coherence (normalized field uniformity)
        # High coherence = uniform field, low coherence = scattered field
        field_std = np.std(self._field)
        field_mean = np.mean(np.abs(self._field))
        coherence = np.exp(-field_std / (field_mean + 1e-10))
        coherence = np.clip(coherence, 0.0, 1.0)

        # Calculate entropy (measure of disorder)
        # Using field distribution as proxy
        field_normalized = np.abs(self._field) / (np.sum(np.abs(self._field)) + 1e-10)
        field_normalized = field_normalized.flatten()
        field_normalized = field_normalized[field_normalized > 1e-10]
        entropy = -np.sum(field_normalized * np.log(field_normalized + 1e-10))

        # Calculate energy (total field magnitude)
        energy = np.sum(self._field**2)

        # Create and store state
        state = FieldState(
            time=self._time,
            coherence=float(coherence),
            entropy=float(entropy),
            energy=float(energy),
        )

        self._history.append(state)


__all__ = ["QCMGParameters", "FieldState", "QuantacosmorphysigeneticField"]
from quasim.sim.qcmg_field import (FieldState, QCMGParameters,
                                   QuantacosmomorphysigeneticField)

__version__ = "0.1.0"

__all__ = [
    "QCMGParameters",
    "FieldState",
    "QuantacosmomorphysigeneticField",
    "__version__",
from quasim.sim.qcmg import (QCMGParameters, QCMGState,
                             QuantacosmorphysigeneticField)

__all__ = [
    "QCMGParameters",
    "QCMGState",
    "QuantacosmorphysigeneticField",
]
