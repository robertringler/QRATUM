"""Quantacosmomorphysigenetic Field Simulation.

This module implements the QCMG model for coupled quantum-classical field dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class QCMGParameters:
    """Parameters for QCMG field simulation.

    Attributes:
        grid_size: Size of the spatial grid (default: 64)
        dt: Time step for evolution (default: 0.01)
        coupling_strength: Strength of field coupling (default: 0.1)
        interaction_strength: Strength of self-interaction (default: 0.1)
        damping_coeff: Damping coefficient (default: 0.01)
        thermal_noise: Thermal noise level (default: 0.001)
        random_seed: Random seed for reproducibility (default: None)
    """

    grid_size: int = 64
    dt: float = 0.01
    coupling_strength: float = 0.1
    interaction_strength: float = 0.1
    damping_coeff: float = 0.01
    thermal_noise: float = 0.001
    random_seed: int | None = None


@dataclass
class QCMGState:
    """State of the QCMG field system.

    Attributes:
        time: Current simulation time
        phi_m: Physical/material field (complex array)
        phi_i: Informational field (complex array)
        coherence: Quantum coherence measure [0, 1]
        entropy: System entropy [0, ∞)
        energy: Total field energy
    """

    time: float
    phi_m: np.ndarray
    phi_i: np.ndarray
    coherence: float
    entropy: float
    energy: float


class QuantacosmorphysigeneticField:
    """Quantacosmomorphysigenetic field simulator.

    Simulates coupled quantum-classical field dynamics with:
    - Physical field (Φ_m): Material/energy distribution
    - Informational field (Φ_i): Information density and quantum correlations

    The fields evolve according to coupled differential equations with
    damping, coupling, and thermal noise.
    """

    def __init__(self, params: QCMGParameters):
        """Initialize the field simulator.

        Args:
            params: Simulation parameters
        """
        self.params = params
        self.time = 0.0
        self.phi_m: np.ndarray | None = None
        self.phi_i: np.ndarray | None = None
        self.trajectory: dict[str, list] = {
            "time": [],
            "coherence": [],
            "entropy": [],
            "energy": [],
        }

        # Set random seed if provided
        if params.random_seed is not None:
            np.random.seed(params.random_seed)

        # Initialize spatial grid
        self.x = np.linspace(-10, 10, params.grid_size)
        self.dx = self.x[1] - self.x[0]

    def initialize(self, mode: str = "gaussian") -> None:
        """Initialize the field state.

        Args:
            mode: Initialization mode - "gaussian", "soliton", or "random"
        """
        n = self.params.grid_size

        if mode == "gaussian":
            # Gaussian wave packet for both fields
            self.phi_m = np.exp(-0.5 * (self.x**2)) * (1 + 0.1j)
            self.phi_i = np.exp(-0.5 * (self.x**2)) * (1 - 0.1j)

        elif mode == "soliton":
            # Soliton-like initial condition
            self.phi_m = 1.0 / np.cosh(self.x) * (1 + 0.2j)
            self.phi_i = 1.0 / np.cosh(self.x) * (1 - 0.2j)

        elif mode == "random":
            # Random initial state
            real_m = np.random.randn(n)
            imag_m = np.random.randn(n)
            real_i = np.random.randn(n)
            imag_i = np.random.randn(n)

            self.phi_m = (real_m + 1j * imag_m) * 0.1
            self.phi_i = (real_i + 1j * imag_i) * 0.1

        else:
            raise ValueError(f"Unknown initialization mode: {mode}")

        # Normalize
        self.phi_m = self.phi_m / np.sqrt(np.sum(np.abs(self.phi_m) ** 2) * self.dx)
        self.phi_i = self.phi_i / np.sqrt(np.sum(np.abs(self.phi_i) ** 2) * self.dx)

        # Reset time and trajectory
        self.time = 0.0
        self.trajectory = {
            "time": [],
            "coherence": [],
            "entropy": [],
            "energy": [],
        }

    def _compute_laplacian(self, field: np.ndarray) -> np.ndarray:
        """Compute spatial Laplacian using finite differences.

        Args:
            field: Input field array

        Returns:
            Laplacian of the field
        """
        laplacian = np.zeros_like(field)
        laplacian[1:-1] = (field[2:] - 2 * field[1:-1] + field[:-2]) / self.dx**2

        # Periodic boundary conditions
        laplacian[0] = (field[1] - 2 * field[0] + field[-1]) / self.dx**2
        laplacian[-1] = (field[0] - 2 * field[-1] + field[-2]) / self.dx**2

        return laplacian

    def _compute_coherence(self) -> float:
        """Compute quantum coherence measure.

        Returns:
            Coherence value in [0, 1]
        """
        if self.phi_m is None or self.phi_i is None:
            return 0.0

        # Coherence based on field overlap
        overlap = np.abs(np.sum(np.conj(self.phi_m) * self.phi_i) * self.dx)
        norm_m = np.sqrt(np.sum(np.abs(self.phi_m) ** 2) * self.dx)
        norm_i = np.sqrt(np.sum(np.abs(self.phi_i) ** 2) * self.dx)

        if norm_m * norm_i > 0:
            coherence = overlap / (norm_m * norm_i)
        else:
            coherence = 0.0

        return float(np.clip(coherence, 0.0, 1.0))

    def _compute_entropy(self) -> float:
        """Compute system entropy.

        Returns:
            Entropy value >= 0
        """
        if self.phi_m is None or self.phi_i is None:
            return 0.0

        # Von Neumann-like entropy based on field distribution
        prob_m = np.abs(self.phi_m) ** 2
        prob_i = np.abs(self.phi_i) ** 2

        # Normalize
        prob_m = prob_m / (np.sum(prob_m) + 1e-10)
        prob_i = prob_i / (np.sum(prob_i) + 1e-10)

        # Shannon entropy
        entropy_m = -np.sum(prob_m * np.log(prob_m + 1e-10))
        entropy_i = -np.sum(prob_i * np.log(prob_i + 1e-10))

        return float(entropy_m + entropy_i)

    def _compute_energy(self) -> float:
        """Compute total field energy.

        Returns:
            Total energy
        """
        if self.phi_m is None or self.phi_i is None:
            return 0.0

        # Kinetic energy (gradient terms)
        grad_m = np.gradient(self.phi_m, self.dx)
        grad_i = np.gradient(self.phi_i, self.dx)

        kinetic_m = 0.5 * np.sum(np.abs(grad_m) ** 2) * self.dx
        kinetic_i = 0.5 * np.sum(np.abs(grad_i) ** 2) * self.dx

        # Potential energy (field amplitudes)
        potential_m = 0.5 * np.sum(np.abs(self.phi_m) ** 2) * self.dx
        potential_i = 0.5 * np.sum(np.abs(self.phi_i) ** 2) * self.dx

        # Interaction energy
        interaction = (
            self.params.coupling_strength
            * np.sum(np.abs(self.phi_m * np.conj(self.phi_i)))
            * self.dx
        )

        return float(kinetic_m + kinetic_i + potential_m + potential_i + interaction)

    def evolve(self) -> QCMGState:
        """Evolve the field forward by one time step.

        Returns:
            Current field state after evolution
        """
        if self.phi_m is None or self.phi_i is None:
            raise RuntimeError("Field not initialized. Call initialize() first.")

        dt = self.params.dt

        # Add thermal noise
        noise_m = (
            np.random.randn(self.params.grid_size) + 1j * np.random.randn(self.params.grid_size)
        ) * self.params.thermal_noise
        noise_i = (
            np.random.randn(self.params.grid_size) + 1j * np.random.randn(self.params.grid_size)
        ) * self.params.thermal_noise

        # Compute Laplacians (diffusion)
        laplacian_m = self._compute_laplacian(self.phi_m)
        laplacian_i = self._compute_laplacian(self.phi_i)

        # Coupled field equations
        # dφ_m/dt = i*Laplacian(φ_m) + coupling*φ_i - damping*φ_m + noise
        dphi_m = (
            1j * laplacian_m
            + self.params.coupling_strength * self.phi_i
            - self.params.damping_coeff * self.phi_m
            - self.params.interaction_strength * np.abs(self.phi_m) ** 2 * self.phi_m
            + noise_m
        )

        # dφ_i/dt = i*Laplacian(φ_i) + coupling*φ_m - damping*φ_i + noise
        dphi_i = (
            1j * laplacian_i
            + self.params.coupling_strength * self.phi_m
            - self.params.damping_coeff * self.phi_i
            - self.params.interaction_strength * np.abs(self.phi_i) ** 2 * self.phi_i
            + noise_i
        )

        # Update fields (Euler method)
        self.phi_m = self.phi_m + dt * dphi_m
        self.phi_i = self.phi_i + dt * dphi_i

        # Update time
        self.time += dt

        # Compute observables
        coherence = self._compute_coherence()
        entropy = self._compute_entropy()
        energy = self._compute_energy()

        # Store trajectory
        self.trajectory["time"].append(self.time)
        self.trajectory["coherence"].append(coherence)
        self.trajectory["entropy"].append(entropy)
        self.trajectory["energy"].append(energy)

        return QCMGState(
            time=self.time,
            phi_m=self.phi_m.copy(),
            phi_i=self.phi_i.copy(),
            coherence=coherence,
            entropy=entropy,
            energy=energy,
        )

    def get_state(self) -> QCMGState:
        """Get current field state without evolving.

        Returns:
            Current field state
        """
        if self.phi_m is None or self.phi_i is None:
            raise RuntimeError("Field not initialized. Call initialize() first.")

        return QCMGState(
            time=self.time,
            phi_m=self.phi_m.copy(),
            phi_i=self.phi_i.copy(),
            coherence=self._compute_coherence(),
            entropy=self._compute_entropy(),
            energy=self._compute_energy(),
        )

    def export_state(self) -> dict[str, Any]:
        """Export field state and trajectory data.

        Returns:
            Dictionary containing state and trajectory information
        """
        if self.phi_m is None or self.phi_i is None:
            raise RuntimeError("Field not initialized. Call initialize() first.")

        return {
            "parameters": {
                "grid_size": self.params.grid_size,
                "dt": self.params.dt,
                "coupling_strength": self.params.coupling_strength,
                "interaction_strength": self.params.interaction_strength,
                "damping_coeff": self.params.damping_coeff,
                "thermal_noise": self.params.thermal_noise,
                "random_seed": self.params.random_seed,
            },
            "current_state": {
                "time": self.time,
                "coherence": self._compute_coherence(),
                "entropy": self._compute_entropy(),
                "energy": self._compute_energy(),
            },
            "trajectory": {
                "time": self.trajectory["time"],
                "coherence": self.trajectory["coherence"],
                "entropy": self.trajectory["entropy"],
                "energy": self.trajectory["energy"],
            },
        }
