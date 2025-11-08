"""Quantacosmorphysigenetic (QCMG) Field Simulation Module.

This module implements a mathematically rigorous symbolic field evolution system
modeling the coevolution of physical and informational fields.

The QCMG module implements coupled field equations:
    Φ = ∂_t (Φ_m ⊗ Φ_i)

Where:
    - Φ_m: Physical/material field (energy density, spatial structure)
    - Φ_i: Informational field (entropy, complexity, coherence)
    - ⊗: Tensor coupling operator

Evolution follows Hamiltonian dynamics with entropy production:
    ∂Φ_m/∂t = -δH/δΦ_i + η_m(t)
    ∂Φ_i/∂t = -δH/δΦ_m + η_i(t)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import json
import numpy as np


@dataclass
class QCMGParameters:
    """Configuration parameters for QCMG field simulation.

    Attributes:
        grid_size: Number of spatial grid points (default: 64)
        dt: Time step for integration (default: 0.01)
        coupling_strength: Coupling between material and informational fields (default: 1.0)
        dissipation_rate: Energy dissipation rate (default: 0.01)
        random_seed: Random seed for reproducibility (default: None)
    """

    grid_size: int = 64
    dt: float = 0.01
    coupling_strength: float = 1.0
    dissipation_rate: float = 0.01
    random_seed: int | None = None

    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.grid_size <= 0:
            raise ValueError("grid_size must be positive")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.coupling_strength < 0:
            raise ValueError("coupling_strength must be non-negative")
        if self.dissipation_rate < 0:
            raise ValueError("dissipation_rate must be non-negative")


@dataclass
class FieldState:
    """State representation of the QCMG field system.

    Attributes:
        phi_m: Material field (complex array)
        phi_i: Informational field (complex array)
        time: Current simulation time
        coherence: Quantum coherence measure [0, 1]
        entropy: Von Neumann entropy (non-negative)
        energy: Total field energy
    """

    phi_m: np.ndarray
    phi_i: np.ndarray
    time: float
    coherence: float
    entropy: float
    energy: float

    def validate(self) -> bool:
        """Validate that state values are within physical bounds.

        Returns:
            True if state is valid, False otherwise
        """
        # Check for NaN or Inf
        if not np.all(np.isfinite(self.phi_m)) or not np.all(np.isfinite(self.phi_i)):
            return False

        # Check coherence bounds
        if not 0 <= self.coherence <= 1:
            return False

        # Check entropy non-negativity
        if self.entropy < 0:
            return False

        # Check energy finiteness
        if not np.isfinite(self.energy):
            return False

        return True


class QuantacosmomorphysigeneticField:
    """Main evolution engine for QCMG field simulation.

    This class implements the core field evolution using RK4 numerical integration
    with periodic boundary conditions, and computes physical observables.
    """

    def __init__(self, params: QCMGParameters):
        """Initialize the QCMG field with given parameters.

        Args:
            params: Configuration parameters for the simulation
        """
        self.params = params

        # Set random seed if provided
        if params.random_seed is not None:
            np.random.seed(params.random_seed)

        # Initialize fields
        self.phi_m = np.zeros(params.grid_size, dtype=complex)
        self.phi_i = np.zeros(params.grid_size, dtype=complex)
        self.time = 0.0

        # History tracking
        self.history: list[FieldState] = []

        # Grid spacing for derivatives
        self.dx = 2 * np.pi / params.grid_size

    def initialize(self, mode: Literal["gaussian", "soliton", "random"] = "gaussian"):
        """Initialize the field state with a specific mode.

        Args:
            mode: Initialization mode
                - "gaussian": Gaussian wave packet
                - "soliton": Soliton-like localized state
                - "random": Random field configuration
        """
        x = np.linspace(0, 2 * np.pi, self.params.grid_size, endpoint=False)

        if mode == "gaussian":
            # Gaussian wave packet centered at π
            sigma = 0.5
            self.phi_m = np.exp(-((x - np.pi) ** 2) / (2 * sigma**2)) * np.exp(1j * x)
            self.phi_i = np.exp(-((x - np.pi) ** 2) / (2 * sigma**2)) * np.exp(-1j * x)

        elif mode == "soliton":
            # Soliton-like profile using sech function
            width = 1.0
            self.phi_m = 1 / np.cosh((x - np.pi) / width) * np.exp(1j * x)
            self.phi_i = 1 / np.cosh((x - np.pi) / width) * np.exp(-1j * 0.5 * x)

        elif mode == "random":
            # Random complex field
            self.phi_m = np.random.randn(self.params.grid_size) + 1j * np.random.randn(
                self.params.grid_size
            )
            self.phi_i = np.random.randn(self.params.grid_size) + 1j * np.random.randn(
                self.params.grid_size
            )

        else:
            raise ValueError(f"Unknown initialization mode: {mode}")

        # Normalize fields
        self._normalize_fields()

        # Store initial state
        self.time = 0.0
        state = self._compute_state()
        self.history.append(state)

    def _normalize_fields(self):
        """Normalize the field amplitudes to prevent runaway growth."""
        norm_m = np.sqrt(np.sum(np.abs(self.phi_m) ** 2) * self.dx)
        norm_i = np.sqrt(np.sum(np.abs(self.phi_i) ** 2) * self.dx)

        if norm_m > 0:
            self.phi_m /= norm_m
        if norm_i > 0:
            self.phi_i /= norm_i

    def _compute_gradient(self, field: np.ndarray) -> np.ndarray:
        """Compute spatial gradient using finite differences with periodic BC.

        Args:
            field: Field array

        Returns:
            Gradient of the field
        """
        # Central difference with periodic boundaries
        grad = np.zeros_like(field)
        grad[1:-1] = (field[2:] - field[:-2]) / (2 * self.dx)
        grad[0] = (field[1] - field[-1]) / (2 * self.dx)
        grad[-1] = (field[0] - field[-2]) / (2 * self.dx)
        return grad

    def _compute_laplacian(self, field: np.ndarray) -> np.ndarray:
        """Compute spatial Laplacian using finite differences with periodic BC.

        Args:
            field: Field array

        Returns:
            Laplacian of the field
        """
        lapl = np.zeros_like(field)
        lapl[1:-1] = (field[2:] - 2 * field[1:-1] + field[:-2]) / (self.dx**2)
        lapl[0] = (field[1] - 2 * field[0] + field[-1]) / (self.dx**2)
        lapl[-1] = (field[0] - 2 * field[-1] + field[-2]) / (self.dx**2)
        return lapl

    def _field_derivative(
        self, phi_m: np.ndarray, phi_i: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute time derivatives of the coupled fields.

        Implements the coupled field equations:
            ∂Φ_m/∂t = -δH/δΦ_i + η_m(t)
            ∂Φ_i/∂t = -δH/δΦ_m + η_i(t)

        Args:
            phi_m: Material field
            phi_i: Informational field

        Returns:
            Time derivatives (dphi_m_dt, dphi_i_dt)
        """
        # Compute Laplacians (kinetic energy terms)
        lapl_m = self._compute_laplacian(phi_m)
        lapl_i = self._compute_laplacian(phi_i)

        # Hamiltonian derivatives (variational principle)
        # δH/δΦ_i includes kinetic and coupling terms
        dH_dphi_i = -lapl_m + self.params.coupling_strength * phi_m * np.conj(phi_i)
        dH_dphi_m = -lapl_i + self.params.coupling_strength * phi_i * np.conj(phi_m)

        # Time derivatives with dissipation
        dphi_m_dt = -1j * dH_dphi_i - self.params.dissipation_rate * phi_m
        dphi_i_dt = -1j * dH_dphi_m - self.params.dissipation_rate * phi_i

        return dphi_m_dt, dphi_i_dt

    def _rk4_step(self) -> tuple[np.ndarray, np.ndarray]:
        """Perform one RK4 integration step.

        Returns:
            Updated (phi_m, phi_i) after one time step
        """
        dt = self.params.dt

        # RK4 stage 1
        k1_m, k1_i = self._field_derivative(self.phi_m, self.phi_i)

        # RK4 stage 2
        k2_m, k2_i = self._field_derivative(
            self.phi_m + 0.5 * dt * k1_m, self.phi_i + 0.5 * dt * k1_i
        )

        # RK4 stage 3
        k3_m, k3_i = self._field_derivative(
            self.phi_m + 0.5 * dt * k2_m, self.phi_i + 0.5 * dt * k2_i
        )

        # RK4 stage 4
        k4_m, k4_i = self._field_derivative(self.phi_m + dt * k3_m, self.phi_i + dt * k3_i)

        # Combine stages
        phi_m_new = self.phi_m + (dt / 6) * (k1_m + 2 * k2_m + 2 * k3_m + k4_m)
        phi_i_new = self.phi_i + (dt / 6) * (k1_i + 2 * k2_i + 2 * k3_i + k4_i)

        return phi_m_new, phi_i_new

    def _compute_coherence(self) -> float:
        """Compute quantum coherence measure.

        Coherence is defined as the normalized inner product magnitude between
        material and informational fields, bounded in [0, 1].

        Returns:
            Coherence value in [0, 1]
        """
        inner_product = np.sum(np.conj(self.phi_m) * self.phi_i) * self.dx
        coherence = np.abs(inner_product)

        # Ensure bounded in [0, 1]
        return float(np.clip(coherence, 0.0, 1.0))

    def _compute_entropy(self) -> float:
        """Compute Von Neumann entropy of the field state.

        Simplified entropy based on field amplitude distribution.

        Returns:
            Entropy value (non-negative)
        """
        # Compute probability distribution from field amplitudes
        probs_m = np.abs(self.phi_m) ** 2
        probs_i = np.abs(self.phi_i) ** 2

        # Combined probability distribution
        probs = (probs_m + probs_i) / 2
        probs = probs / (np.sum(probs) + 1e-16)  # Normalize with small epsilon

        # Von Neumann entropy: S = -sum(p * log(p))
        # Avoid log(0) by filtering out zero probabilities
        nonzero_probs = probs[probs > 1e-16]
        entropy = -np.sum(nonzero_probs * np.log(nonzero_probs + 1e-16))

        return float(np.maximum(entropy, 0.0))

    def _compute_energy(self) -> float:
        """Compute total field energy.

        Energy includes kinetic and interaction terms.

        Returns:
            Total energy
        """
        # Kinetic energy from gradients
        grad_m = self._compute_gradient(self.phi_m)
        grad_i = self._compute_gradient(self.phi_i)

        kinetic_m = np.sum(np.abs(grad_m) ** 2) * self.dx
        kinetic_i = np.sum(np.abs(grad_i) ** 2) * self.dx

        # Potential energy from field coupling
        potential = (
            self.params.coupling_strength
            * np.sum(np.abs(self.phi_m * np.conj(self.phi_i)) ** 2)
            * self.dx
        )

        energy = kinetic_m + kinetic_i + potential
        return float(energy)

    def _compute_state(self) -> FieldState:
        """Compute current field state with all observables.

        Returns:
            Complete field state
        """
        coherence = self._compute_coherence()
        entropy = self._compute_entropy()
        energy = self._compute_energy()

        state = FieldState(
            phi_m=self.phi_m.copy(),
            phi_i=self.phi_i.copy(),
            time=self.time,
            coherence=coherence,
            entropy=entropy,
            energy=energy,
        )

        return state

    def evolve(self, steps: int = 1) -> FieldState:
        """Evolve the field for a given number of steps.

        Args:
            steps: Number of time steps to evolve (default: 1)

        Returns:
            Field state after evolution

        Raises:
            RuntimeError: If state becomes invalid during evolution
        """
        for _ in range(steps):
            # Perform RK4 step
            self.phi_m, self.phi_i = self._rk4_step()

            # Periodic renormalization to prevent runaway growth
            if _ % 10 == 0:
                self._normalize_fields()

            # Update time
            self.time += self.params.dt

            # Compute and validate state
            state = self._compute_state()

            if not state.validate():
                raise RuntimeError(
                    f"Invalid state at time {self.time}: "
                    f"C={state.coherence}, S={state.entropy}, E={state.energy}"
                )

            # Store in history
            self.history.append(state)

        return state

    def export_state(self, include_history: bool = True) -> dict:
        """Export current state and parameters as dictionary.

        Args:
            include_history: Whether to include full evolution history

        Returns:
            Dictionary containing state and parameters
        """
        current_state = self.history[-1] if self.history else self._compute_state()

        export_data = {
            "parameters": {
                "grid_size": self.params.grid_size,
                "dt": self.params.dt,
                "coupling_strength": self.params.coupling_strength,
                "dissipation_rate": self.params.dissipation_rate,
                "random_seed": self.params.random_seed,
            },
            "current_state": {
                "time": current_state.time,
                "coherence": current_state.coherence,
                "entropy": current_state.entropy,
                "energy": current_state.energy,
                "phi_m_real": current_state.phi_m.real.tolist(),
                "phi_m_imag": current_state.phi_m.imag.tolist(),
                "phi_i_real": current_state.phi_i.real.tolist(),
                "phi_i_imag": current_state.phi_i.imag.tolist(),
            },
        }

        if include_history:
            export_data["history"] = [
                {
                    "time": state.time,
                    "coherence": state.coherence,
                    "entropy": state.entropy,
                    "energy": state.energy,
                }
                for state in self.history
            ]

        return export_data

    def save_to_json(self, filename: str, include_history: bool = True):
        """Save state to JSON file.

        Args:
            filename: Output file path
            include_history: Whether to include full evolution history
        """
        data = self.export_state(include_history=include_history)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
