"""Langevin dynamics simulation with thermal noise.

Brownian dynamics for chemical systems with continuous concentrations.
Suitable for systems with large molecule counts where discrete stochasticity
is less important.

Reference:
Gillespie, D. T. (2000). The chemical Langevin equation.
The Journal of Chemical Physics, 113(1), 297-306.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.mechanism import BioMechanism


class LangevinSimulator:
    """Langevin dynamics simulator for chemical systems.

    Integrates the chemical Langevin equation:
    dX/dt = drift(X) + sqrt(diffusion(X)) * noise(t)

    where:
    - drift = deterministic reaction dynamics
    - diffusion = stochastic fluctuations (proportional to sqrt(propensity))
    - noise = Gaussian white noise

    Attributes:
        mechanism: Biological mechanism to simulate
        temperature: Temperature in Kelvin
        boltzmann: Boltzmann constant
    """

    def __init__(
        self,
        mechanism: BioMechanism,
        temperature: float = 310.0,  # Body temperature (37°C)
    ):
        """Initialize Langevin simulator.

        Args:
            mechanism: Mechanism to simulate
            temperature: Temperature in Kelvin
        """

        self.mechanism = mechanism
        self.temperature = temperature
        self.boltzmann = 1.987e-3  # Boltzmann constant (kcal/mol/K)

        self._rng: Optional[np.random.Generator] = None

    def run(
        self,
        t_max: float,
        dt: float,
        initial_state: dict[str, float],
        seed: Optional[int] = None,
    ) -> tuple[list[float], dict[str, list[float]]]:
        """Run Langevin dynamics simulation.

        Args:
            t_max: Maximum simulation time (seconds)
            dt: Time step size (seconds)
            initial_state: Initial concentrations (nM)
            seed: Random seed for reproducibility

        Returns:
            Tuple of (times, trajectories) where trajectories[species] = [concentrations]
        """

        # Initialize random number generator
        self._rng = np.random.default_rng(seed)

        # Initialize state
        state = initial_state.copy()

        # Ensure all species have initial values
        for state_name in self.mechanism._states:
            if state_name not in state:
                state[state_name] = 0.0

        # Storage for trajectory
        n_steps = int(t_max / dt) + 1
        times = [i * dt for i in range(n_steps)]
        trajectories: dict[str, list[float]] = {species: [state[species]] for species in state}

        # Main integration loop
        current_time = 0.0
        for step in range(1, n_steps):
            current_time = step * dt

            # Compute forces (chemical potentials)
            forces = self._compute_forces(state)

            # Add thermal noise
            noise = self._add_thermal_noise(forces, dt)

            # Update concentrations
            for species in state:
                drift = forces.get(species, 0.0)
                stochastic = noise.get(species, 0.0)

                # Euler-Maruyama integration
                state[species] += drift * dt + stochastic * np.sqrt(dt)

                # Enforce non-negativity
                state[species] = max(0.0, state[species])

            # Record trajectory
            for species in state:
                trajectories[species].append(state[species])

        return times, trajectories

    def _compute_forces(self, state: dict[str, float]) -> dict[str, float]:
        """Compute chemical potential forces (drift terms).

        For each species, compute net rate of change from all reactions:
        dX_i/dt = sum_j (stoich_ij × rate_j)

        Args:
            state: Current concentrations

        Returns:
            Dictionary of drift terms for each species
        """

        forces: dict[str, float] = dict.fromkeys(state, 0.0)

        for transition in self.mechanism._transitions:
            source = transition.source
            target = transition.target

            # Reaction rate (mass action kinetics)
            rate = transition.rate_constant * state[source]

            # Update forces (stoichiometry)
            forces[source] -= rate  # Source is consumed
            forces[target] += rate  # Target is produced

            # Handle reverse reaction if applicable
            if transition.reversible and transition.reverse_rate is not None:
                reverse_rate = transition.reverse_rate * state[target]
                forces[target] -= reverse_rate
                forces[source] += reverse_rate

        return forces

    def _add_thermal_noise(
        self,
        forces: dict[str, float],
        dt: float,
    ) -> dict[str, float]:
        """Add thermal noise to forces.

        Noise amplitude is proportional to sqrt(reaction propensity):
        sigma = sqrt(2 * k_B * T * mobility)

        For chemical reactions:
        sigma_i = sqrt(sum_j |stoich_ij| × rate_j)

        Args:
            forces: Drift terms
            dt: Time step

        Returns:
            Dictionary of noise terms
        """

        noise: dict[str, float] = {}

        for species in forces:
            # Compute noise amplitude from force magnitude
            # Higher reaction rates → more stochastic fluctuations
            force_magnitude = abs(forces[species])

            # Noise amplitude (sqrt of rate for Poisson-like fluctuations)
            amplitude = np.sqrt(force_magnitude) if force_magnitude > 0 else 0.0

            # Generate Gaussian noise
            noise[species] = self._rng.normal(0.0, amplitude)

        return noise

    def run_with_temperature_schedule(
        self,
        t_max: float,
        dt: float,
        initial_state: dict[str, float],
        temperature_schedule: list[tuple[float, float]],
        seed: Optional[int] = None,
    ) -> tuple[list[float], dict[str, list[float]]]:
        """Run simulation with time-varying temperature.

        Useful for simulated annealing or temperature-dependent studies.

        Args:
            t_max: Maximum simulation time
            dt: Time step
            initial_state: Initial concentrations
            temperature_schedule: List of (time, temperature) tuples
            seed: Random seed

        Returns:
            Tuple of (times, trajectories)
        """

        # For Phase 1, use constant temperature
        # Phase 2+ will implement full temperature scheduling
        return self.run(t_max, dt, initial_state, seed)
