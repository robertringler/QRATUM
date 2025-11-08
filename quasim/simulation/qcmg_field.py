"""Quantacosmorphysigenetic Field simulation engine.

This module provides symbolic and numerical simulation engines for
quantum-classical systems using the QCMG (Quantacosmorphysigenetic) field theory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QCMGParameters:
    """Parameters for Quantacosmorphysigenetic Field simulation.

    Attributes:
        coupling_strength: Field coupling strength coefficient (default: 1.0)
        field_dimension: Dimensionality of the field space (default: 3)
        evolution_steps: Number of evolution steps for simulation (default: 100)
        precision: Numerical precision mode ('fp32', 'fp64') (default: 'fp64')
        temperature: System temperature in field units (default: 1.0)
    """

    coupling_strength: float = 1.0
    field_dimension: int = 3
    evolution_steps: int = 100
    precision: str = "fp64"
    temperature: float = 1.0

    def __post_init__(self) -> None:
        """Validate QCMG parameters."""
        if self.field_dimension < 1:
            raise ValueError("Field dimension must be at least 1")
        if self.evolution_steps < 1:
            raise ValueError("Evolution steps must be at least 1")
        if self.precision not in {"fp32", "fp64"}:
            raise ValueError("Precision must be 'fp32' or 'fp64'")
        if self.temperature < 0:
            raise ValueError("Temperature must be non-negative")


@dataclass
class FieldState:
    """State representation for Quantacosmorphysigenetic Field.

    Attributes:
        field_values: Current field configuration values
        momentum: Conjugate momentum field values
        energy: Total energy of the field state
        time: Current simulation time
        metadata: Additional state metadata
    """

    field_values: list[complex] = field(default_factory=list)
    momentum: list[complex] = field(default_factory=list)
    energy: float = 0.0
    time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate field state."""
        if len(self.field_values) != len(self.momentum) and self.momentum:
            raise ValueError("Field values and momentum must have same length")

    def copy(self) -> FieldState:
        """Create a deep copy of the field state.

        Returns:
            New FieldState instance with copied values
        """
        return FieldState(
            field_values=self.field_values.copy(),
            momentum=self.momentum.copy(),
            energy=self.energy,
            time=self.time,
            metadata=self.metadata.copy(),
        )


class QuantacosmorphysigeneticField:
    """Quantacosmorphysigenetic Field simulator for quantum-classical systems.

    This class provides both symbolic and numerical simulation engines for
    quantum-classical hybrid systems based on QCMG field theory. It supports
    field evolution, state management, and quantum-classical coupling.

    Attributes:
        parameters: QCMG field parameters configuration
        state: Current field state
        history: Historical record of field states
    """

    def __init__(self, parameters: QCMGParameters | None = None) -> None:
        """Initialize the QCMG field simulator.

        Args:
            parameters: QCMG field parameters, uses defaults if None
        """
        self.parameters = parameters if parameters is not None else QCMGParameters()
        self.state = FieldState()
        self.history: list[FieldState] = []
        self._initialized = False

    def initialize(self, initial_state: FieldState | None = None) -> None:
        """Initialize the field with an initial state.

        Args:
            initial_state: Initial field state, creates default if None
        """
        if initial_state is not None:
            self.state = initial_state.copy()
        else:
            # Initialize with default vacuum state
            dim = self.parameters.field_dimension
            self.state = FieldState(
                field_values=[0j] * dim,
                momentum=[0j] * dim,
                energy=0.0,
                time=0.0,
            )

        self._initialized = True
        self.history = [self.state.copy()]

    def evolve(self, time_delta: float = 1.0) -> FieldState:
        """Evolve the field forward in time.

        Uses quantum-classical hybrid evolution algorithm based on QCMG theory.

        Args:
            time_delta: Time step for evolution

        Returns:
            Updated field state after evolution

        Raises:
            RuntimeError: If field not initialized
        """
        if not self._initialized:
            raise RuntimeError("Field not initialized. Call initialize() first.")

        # Quantum-classical evolution (simplified Hamiltonian dynamics)
        new_field_values = []
        new_momentum = []

        for i in range(len(self.state.field_values)):
            # Update field using momentum (position update)
            field_val = self.state.field_values[i]
            mom_val = self.state.momentum[i]

            # Simplified symplectic evolution
            new_field = field_val + time_delta * mom_val

            # Update momentum using field coupling
            coupling_factor = self.parameters.coupling_strength
            new_mom = mom_val - time_delta * coupling_factor * field_val

            new_field_values.append(new_field)
            new_momentum.append(new_mom)

        # Calculate energy (simplified Hamiltonian)
        kinetic_energy = sum(abs(p) ** 2 for p in new_momentum) / 2
        potential_energy = (
            self.parameters.coupling_strength * sum(abs(f) ** 2 for f in new_field_values) / 2
        )
        total_energy = kinetic_energy + potential_energy

        # Update state
        self.state = FieldState(
            field_values=new_field_values,
            momentum=new_momentum,
            energy=total_energy,
            time=self.state.time + time_delta,
        )

        # Record in history
        self.history.append(self.state.copy())

        return self.state.copy()

    def simulate(self, num_steps: int | None = None) -> list[FieldState]:
        """Run a full simulation for specified number of steps.

        Args:
            num_steps: Number of evolution steps, uses parameter default if None

        Returns:
            List of field states at each evolution step

        Raises:
            RuntimeError: If field not initialized
        """
        if not self._initialized:
            raise RuntimeError("Field not initialized. Call initialize() first.")

        steps = num_steps if num_steps is not None else self.parameters.evolution_steps
        trajectory = []

        for _ in range(steps):
            state = self.evolve()
            trajectory.append(state)

        return trajectory

    def get_state(self) -> FieldState:
        """Get the current field state.

        Returns:
            Copy of current field state
        """
        return self.state.copy()

    def get_history(self) -> list[FieldState]:
        """Get the full evolution history.

        Returns:
            List of all recorded field states
        """
        return [state.copy() for state in self.history]

    def reset(self) -> None:
        """Reset the field to uninitialized state."""
        self.state = FieldState()
        self.history = []
        self._initialized = False
