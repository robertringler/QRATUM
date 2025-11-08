"""
Quantacosmomorphysigenesis (QCMG) Field Evolution Module

Implements the coevolution of physical (Φ_m) and informational (Φ_i) fields
through coupled differential equations:

    Φ = ∂_t (Φ_m ⊗ Φ_i)

Where:
- Φ_m: Physical/material field state (energy density, spatial structure)
- Φ_i: Informational field state (entropy, complexity, coherence)
- ⊗: Tensor coupling operator representing field interaction

The evolution follows a Hamiltonian-like dynamics with entropy production
and coherence constraints, modeling the emergence of structure from
quantum-classical interactions.

Mathematical Framework:
----------------------
∂Φ_m/∂t = -δH/δΦ_i + η_m(t)
∂Φ_i/∂t = -δH/δΦ_m + η_i(t)

H[Φ_m, Φ_i] = ∫ [½|∇Φ_m|² + ½|∇Φ_i|² + V(Φ_m, Φ_i)] dx

With constraints:
- Coherence: C(t) = |⟨Φ_m|Φ_i⟩| / (||Φ_m|| ||Φ_i||)
- Entropy: S(t) = -Tr(ρ log ρ) where ρ ∝ Φ_m ⊗ Φ_i

References:
-----------
Theoretical foundations based on quantum field theory, information geometry,
and non-equilibrium thermodynamics.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FieldState:
    """
    Represents the state of the QCMG field at a given time.

    Attributes:
        phi_m: Physical field amplitude (complex array)
        phi_i: Informational field amplitude (complex array)
        coherence: Quantum coherence measure [0, 1]
        entropy: Von Neumann entropy (non-negative)
        time: Evolution time
        energy: Total field energy
    """

    phi_m: np.ndarray
    phi_i: np.ndarray
    coherence: float
    entropy: float
    time: float
    energy: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to serializable dictionary."""
        return {
            "phi_m_real": self.phi_m.real.tolist(),
            "phi_m_imag": self.phi_m.imag.tolist(),
            "phi_i_real": self.phi_i.real.tolist(),
            "phi_i_imag": self.phi_i.imag.tolist(),
            "coherence": float(self.coherence),
            "entropy": float(self.entropy),
            "time": float(self.time),
            "energy": float(self.energy),
        }


@dataclass
class QCMGParameters:
    """
    Parameters governing QCMG field evolution.

    Physical constants and coupling strengths that determine
    the dynamics of the field system.
    """

    # Spatial discretization
    grid_size: int = 64
    spatial_extent: float = 10.0

    # Temporal evolution
    dt: float = 0.01
    coupling_strength: float = 0.1

    # Physical parameters
    mass_scale: float = 1.0
    info_scale: float = 1.0
    interaction_strength: float = 0.05

    # Noise and damping
    thermal_noise: float = 0.001
    damping_coeff: float = 0.01

    # Random seed for reproducibility
    random_seed: Optional[int] = 42


class QuantacosmorphysigeneticField:
    """
    Quantacosmorphysigenetic Field Evolution Engine.

    Simulates the coupled evolution of physical and informational fields
    through numerical integration of the QCMG field equations.

    The system exhibits:
    - Energy conservation (approximate, with controlled dissipation)
    - Entropy production (second law compliance)
    - Coherence dynamics (quantum-classical transition)
    - Emergent spatial structure

    Example:
    --------
    >>> params = QCMGParameters(grid_size=32, dt=0.01)
    >>> field = QuantacosmorphysigeneticField(params)
    >>> field.initialize()
    >>>
    >>> for i in range(100):
    ...     state = field.evolve()
    ...     print(f"t={state.time:.2f}, C={state.coherence:.3f}, S={state.entropy:.3f}")
    """

    # Numerical stability threshold
    EPSILON = 1e-10

    def __init__(self, parameters: Optional[QCMGParameters] = None):
        """
        Initialize QCMG field simulator.

        Args:
            parameters: Evolution parameters (uses defaults if None)
        """
        self.params = parameters or QCMGParameters()

        # Set random seed for reproducibility
        if self.params.random_seed is not None:
            np.random.seed(self.params.random_seed)

        # Spatial grid
        self.x = np.linspace(
            -self.params.spatial_extent / 2, self.params.spatial_extent / 2, self.params.grid_size
        )
        self.dx = self.x[1] - self.x[0]

        # Field states
        self.phi_m: Optional[np.ndarray] = None
        self.phi_i: Optional[np.ndarray] = None
        self.time: float = 0.0

        # Evolution history
        self.history: list[FieldState] = []

        logger.info(f"QCMG field initialized with grid_size={self.params.grid_size}")

    def initialize(self, mode: str = "gaussian") -> None:
        """
        Initialize field states with specified mode.

        Args:
            mode: Initialization mode - "gaussian", "soliton", "random"
        """
        n = self.params.grid_size

        if mode == "gaussian":
            # Gaussian wave packet for both fields
            sigma_m = self.params.spatial_extent / 8
            sigma_i = self.params.spatial_extent / 6

            self.phi_m = np.exp(-(self.x**2) / (2 * sigma_m**2)) * np.exp(1j * self.x)
            self.phi_i = np.exp(-(self.x**2) / (2 * sigma_i**2)) * np.exp(-1j * 0.5 * self.x)

        elif mode == "soliton":
            # Solitonic initial condition
            v = 0.5  # velocity
            width = 1.0

            self.phi_m = 1.0 / np.cosh(self.x / width) * np.exp(1j * v * self.x)
            self.phi_i = 1.0 / np.cosh(self.x / (width * 1.5)) * np.exp(1j * v * self.x / 2)

        elif mode == "random":
            # Random initial state
            self.phi_m = (np.random.randn(n) + 1j * np.random.randn(n)) / np.sqrt(2)
            self.phi_i = (np.random.randn(n) + 1j * np.random.randn(n)) / np.sqrt(2)

        else:
            raise ValueError(f"Unknown initialization mode: {mode}")

        # Normalize fields
        self.phi_m = self.phi_m / np.linalg.norm(self.phi_m)
        self.phi_i = self.phi_i / np.linalg.norm(self.phi_i)

        self.time = 0.0
        self.history = []

        # Record initial state
        self._record_state()

        logger.info(f"Fields initialized with mode='{mode}'")

    def evolve(self) -> FieldState:
        """
        Evolve fields by one time step using RK4 integration.

        Returns:
            Current field state after evolution
        """
        if self.phi_m is None or self.phi_i is None:
            raise RuntimeError("Fields not initialized. Call initialize() first.")

        dt = self.params.dt

        # RK4 integration
        k1_m, k1_i = self._compute_derivatives(self.phi_m, self.phi_i)

        k2_m, k2_i = self._compute_derivatives(
            self.phi_m + 0.5 * dt * k1_m, self.phi_i + 0.5 * dt * k1_i
        )

        k3_m, k3_i = self._compute_derivatives(
            self.phi_m + 0.5 * dt * k2_m, self.phi_i + 0.5 * dt * k2_i
        )

        k4_m, k4_i = self._compute_derivatives(self.phi_m + dt * k3_m, self.phi_i + dt * k3_i)

        # Update fields
        self.phi_m = self.phi_m + (dt / 6) * (k1_m + 2 * k2_m + 2 * k3_m + k4_m)
        self.phi_i = self.phi_i + (dt / 6) * (k1_i + 2 * k2_i + 2 * k3_i + k4_i)

        # Apply damping and renormalization
        self.phi_m *= 1 - self.params.damping_coeff * dt
        self.phi_i *= 1 - self.params.damping_coeff * dt

        # Renormalize to prevent numerical blowup
        norm_m = np.linalg.norm(self.phi_m)
        norm_i = np.linalg.norm(self.phi_i)

        if norm_m > self.EPSILON:
            self.phi_m = self.phi_m / norm_m
        if norm_i > self.EPSILON:
            self.phi_i = self.phi_i / norm_i

        self.time += dt

        # Record state
        state = self._record_state()

        return state

    def _compute_derivatives(
        self, phi_m: np.ndarray, phi_i: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute time derivatives of fields: ∂Φ_m/∂t and ∂Φ_i/∂t.

        Uses finite differences for spatial derivatives and includes:
        - Kinetic energy (Laplacian terms)
        - Coupling interaction (tensor product)
        - Thermal noise (stochastic forcing)

        Args:
            phi_m: Physical field state
            phi_i: Informational field state

        Returns:
            Tuple of (dphi_m/dt, dphi_i/dt)
        """
        # Compute Laplacians (second spatial derivatives)
        laplacian_m = self._laplacian(phi_m)
        laplacian_i = self._laplacian(phi_i)

        # Coupling terms: V = g * Φ_m* ⊗ Φ_i
        coupling = self.params.coupling_strength * np.conj(phi_m) * phi_i

        # Physical field evolution
        dphi_m_dt = (
            1j * self.params.mass_scale * laplacian_m  # Kinetic term
            - 1j * self.params.interaction_strength * coupling * phi_i  # Interaction
            + self._thermal_noise(phi_m.shape)  # Stochastic forcing
        )

        # Informational field evolution
        dphi_i_dt = (
            1j * self.params.info_scale * laplacian_i  # Kinetic term
            - 1j * self.params.interaction_strength * coupling * phi_m  # Interaction
            + self._thermal_noise(phi_i.shape)  # Stochastic forcing
        )

        return dphi_m_dt, dphi_i_dt

    def _laplacian(self, field: np.ndarray) -> np.ndarray:
        """
        Compute Laplacian using second-order finite differences.

        Uses periodic boundary conditions.
        """
        # Roll arrays for neighbor access
        field_plus = np.roll(field, -1)
        field_minus = np.roll(field, 1)

        # Second derivative: ∂²Φ/∂x² ≈ (Φ_{i+1} - 2Φ_i + Φ_{i-1}) / dx²
        laplacian = (field_plus - 2 * field + field_minus) / (self.dx**2)

        return laplacian

    def _thermal_noise(self, shape: tuple) -> np.ndarray:
        """Generate complex thermal noise."""
        noise_real = np.random.randn(*shape)
        noise_imag = np.random.randn(*shape)

        return self.params.thermal_noise * (noise_real + 1j * noise_imag) / np.sqrt(2)

    def _compute_coherence(self) -> float:
        """
        Compute quantum coherence: C = |⟨Φ_m|Φ_i⟩| / (||Φ_m|| ||Φ_i||).

        Ranges from 0 (completely decoherent) to 1 (perfect coherence).
        """
        inner_product = np.vdot(self.phi_m, self.phi_i)
        norm_m = np.linalg.norm(self.phi_m)
        norm_i = np.linalg.norm(self.phi_i)

        if norm_m < self.EPSILON or norm_i < self.EPSILON:
            return 0.0

        coherence = np.abs(inner_product) / (norm_m * norm_i)

        return float(np.clip(coherence, 0.0, 1.0))

    def _compute_entropy(self) -> float:
        """
        Compute von Neumann entropy: S = -Tr(ρ log ρ).

        Approximated from the field density matrix ρ ∝ Φ_m ⊗ Φ_i.
        """
        # Construct reduced density matrix (simplified)
        prob_m = np.abs(self.phi_m) ** 2
        prob_i = np.abs(self.phi_i) ** 2

        # Normalize
        prob_m = prob_m / (np.sum(prob_m) + self.EPSILON)
        prob_i = prob_i / (np.sum(prob_i) + self.EPSILON)

        # Entropy contributions
        entropy_m = -np.sum(prob_m * np.log(prob_m + self.EPSILON))
        entropy_i = -np.sum(prob_i * np.log(prob_i + self.EPSILON))

        # Combined entropy (tensor product)
        entropy = entropy_m + entropy_i

        return float(entropy)

    def _compute_energy(self) -> float:
        """
        Compute total field energy: H = T + V.

        T: Kinetic energy (gradient terms)
        V: Interaction potential
        """
        # Kinetic energy
        grad_m = np.gradient(self.phi_m, self.dx)
        grad_i = np.gradient(self.phi_i, self.dx)

        kinetic_m = 0.5 * self.params.mass_scale * np.sum(np.abs(grad_m) ** 2) * self.dx
        kinetic_i = 0.5 * self.params.info_scale * np.sum(np.abs(grad_i) ** 2) * self.dx

        # Interaction energy
        coupling = np.conj(self.phi_m) * self.phi_i
        potential = self.params.interaction_strength * np.sum(np.abs(coupling) ** 2) * self.dx

        energy = kinetic_m + kinetic_i + potential

        return float(energy)

    def _record_state(self) -> FieldState:
        """Record current field state to history."""
        state = FieldState(
            phi_m=self.phi_m.copy(),
            phi_i=self.phi_i.copy(),
            coherence=self._compute_coherence(),
            entropy=self._compute_entropy(),
            time=self.time,
            energy=self._compute_energy(),
        )

        self.history.append(state)

        return state

    def get_state(self) -> FieldState:
        """Get current field state."""
        if self.history:
            return self.history[-1]
        raise RuntimeError("No state available. Call initialize() and evolve().")

    def get_history(self) -> list[FieldState]:
        """Get full evolution history."""
        return self.history.copy()

    def export_state(self, state: Optional[FieldState] = None) -> Dict[str, Any]:
        """
        Export field state to JSON-serializable dictionary.

        Args:
            state: State to export (uses current if None)

        Returns:
            Dictionary with all state information
        """
        if state is None:
            state = self.get_state()

        return {
            "qcmg_version": "0.1.0",
            "parameters": {
                "grid_size": self.params.grid_size,
                "spatial_extent": self.params.spatial_extent,
                "dt": self.params.dt,
                "coupling_strength": self.params.coupling_strength,
            },
            "state": state.to_dict(),
            "metadata": {
                "history_length": len(self.history),
                "bounded": self._check_bounded(state),
            },
        }

    def _check_bounded(self, state: FieldState) -> bool:
        """Check if field values are within reasonable bounds."""
        coherence_ok = 0.0 <= state.coherence <= 1.0
        entropy_ok = state.entropy >= 0.0
        energy_ok = np.isfinite(state.energy)

        return bool(coherence_ok and entropy_ok and energy_ok)
