"""Molecular dynamics simulation engine.

Provides real-time molecular dynamics simulation capabilities including:
- Velocity Verlet integration
- Lennard-Jones and Coulomb potentials
- Temperature control (Berendsen thermostat)
- Periodic boundary conditions
- Energy minimization
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import numpy as np

from .pdb_loader import PDBStructure

logger = logging.getLogger(__name__)


class Integrator(Enum):
    """MD integration schemes."""

    VERLET = "verlet"
    VELOCITY_VERLET = "velocity_verlet"
    LEAPFROG = "leapfrog"


class Thermostat(Enum):
    """Temperature control methods."""

    NONE = "none"
    BERENDSEN = "berendsen"
    NOSE_HOOVER = "nose_hoover"
    VELOCITY_RESCALE = "velocity_rescale"


class Barostat(Enum):
    """Pressure control methods."""

    NONE = "none"
    BERENDSEN = "berendsen"
    PARRINELLO_RAHMAN = "parrinello_rahman"


@dataclass
class MDConfig:
    """Configuration for MD simulation."""

    timestep: float = 0.002  # picoseconds
    temperature: float = 300.0  # Kelvin
    pressure: float = 1.0  # bar
    integrator: Integrator = Integrator.VELOCITY_VERLET
    thermostat: Thermostat = Thermostat.BERENDSEN
    barostat: Barostat = Barostat.NONE
    thermostat_tau: float = 0.1  # ps
    barostat_tau: float = 1.0  # ps
    cutoff: float = 10.0  # Angstroms
    pbc: bool = True  # Periodic boundary conditions
    box_size: tuple[float, float, float] = (50.0, 50.0, 50.0)
    random_seed: int = 42
    constraint_bonds: bool = True  # SHAKE/LINCS
    constraint_tolerance: float = 1e-6

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestep": self.timestep,
            "temperature": self.temperature,
            "pressure": self.pressure,
            "integrator": self.integrator.value,
            "thermostat": self.thermostat.value,
            "barostat": self.barostat.value,
            "thermostat_tau": self.thermostat_tau,
            "barostat_tau": self.barostat_tau,
            "cutoff": self.cutoff,
            "pbc": self.pbc,
            "box_size": list(self.box_size),
            "random_seed": self.random_seed,
        }


@dataclass
class SimulationState:
    """Current state of the MD simulation."""

    step: int = 0
    time: float = 0.0  # picoseconds
    positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    velocities: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    forces: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    masses: np.ndarray = field(default_factory=lambda: np.zeros(0))
    charges: np.ndarray = field(default_factory=lambda: np.zeros(0))
    kinetic_energy: float = 0.0
    potential_energy: float = 0.0
    temperature: float = 0.0
    pressure: float = 0.0

    @property
    def total_energy(self) -> float:
        """Get total energy."""
        return self.kinetic_energy + self.potential_energy

    @property
    def num_atoms(self) -> int:
        """Get number of atoms."""
        return len(self.positions)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "step": self.step,
            "time": self.time,
            "positions": self.positions.tolist(),
            "velocities": self.velocities.tolist(),
            "kinetic_energy": self.kinetic_energy,
            "potential_energy": self.potential_energy,
            "total_energy": self.total_energy,
            "temperature": self.temperature,
            "pressure": self.pressure,
            "num_atoms": self.num_atoms,
        }

    def get_frame_data(self) -> dict:
        """Get frame data for visualization."""
        return {
            "step": self.step,
            "time": self.time,
            "positions": self.positions.tolist(),
            "temperature": self.temperature,
            "total_energy": self.total_energy,
        }


@dataclass
class TrajectoryFrame:
    """Single trajectory frame."""

    step: int
    time: float
    positions: np.ndarray
    energies: dict

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "step": self.step,
            "time": self.time,
            "positions": self.positions.tolist(),
            "energies": self.energies,
        }


class MDSimulator:
    """Real-time molecular dynamics simulation engine.

    Provides methods for running MD simulations with real-time
    visualization and interactive control.
    """

    # Physical constants
    KB = 0.0019872041  # Boltzmann constant in kcal/(mol·K)
    AVOGADRO = 6.02214076e23

    # Atomic masses (g/mol)
    MASSES = {
        "H": 1.008,
        "C": 12.011,
        "N": 14.007,
        "O": 15.999,
        "S": 32.065,
        "P": 30.974,
        "F": 18.998,
        "Cl": 35.453,
        "Br": 79.904,
        "I": 126.904,
        "Fe": 55.845,
        "Zn": 65.38,
        "Ca": 40.078,
        "Mg": 24.305,
        "Na": 22.990,
        "K": 39.098,
    }

    # Lennard-Jones parameters (epsilon in kcal/mol, sigma in Angstroms)
    LJ_PARAMS = {
        "H": (0.0157, 1.00),
        "C": (0.1094, 3.40),
        "N": (0.1700, 3.25),
        "O": (0.2100, 3.12),
        "S": (0.2500, 3.55),
        "P": (0.2000, 3.74),
        "F": (0.0610, 3.12),
        "Cl": (0.2650, 3.47),
        "Br": (0.3200, 3.59),
    }

    def __init__(self, config: Optional[MDConfig] = None):
        """Initialize MD simulator.

        Args:
            config: Simulation configuration
        """
        self.config = config or MDConfig()
        self._rng = np.random.default_rng(self.config.random_seed)
        self._state = SimulationState()
        self._structure: Optional[PDBStructure] = None
        self._trajectory: list[TrajectoryFrame] = []
        self._callbacks: list[Callable[[SimulationState], None]] = []
        self._running = False
        self._elements: list[str] = []

    def load_structure(self, structure: PDBStructure) -> None:
        """Load molecular structure for simulation.

        Args:
            structure: PDB structure to simulate
        """
        self._structure = structure
        atoms = structure.all_atoms

        # Extract positions
        positions = np.array([a.coords for a in atoms])

        # Extract elements and masses
        self._elements = []
        masses = []
        charges = []

        for atom in atoms:
            elem = atom.element or atom.name[0]
            self._elements.append(elem)
            masses.append(self.MASSES.get(elem, 12.0))
            charges.append(0.0)  # Simplified: no explicit charges

        # Initialize state
        self._state = SimulationState(
            positions=positions,
            velocities=np.zeros_like(positions),
            forces=np.zeros_like(positions),
            masses=np.array(masses),
            charges=np.array(charges),
        )

        logger.info(f"Loaded structure: {structure.pdb_id} with {len(atoms)} atoms")

    def initialize_velocities(self, temperature: Optional[float] = None) -> None:
        """Initialize velocities from Maxwell-Boltzmann distribution.

        Args:
            temperature: Target temperature (uses config if not specified)
        """
        T = temperature or self.config.temperature
        num_atoms = self._state.num_atoms

        # Generate random velocities
        velocities = self._rng.normal(0, 1, (num_atoms, 3))

        # Scale by mass and temperature
        for i in range(num_atoms):
            mass = self._state.masses[i]
            sigma = np.sqrt(self.KB * T / mass)
            velocities[i] *= sigma

        # Remove center of mass motion
        total_mass = np.sum(self._state.masses)
        com_velocity = np.sum(velocities * self._state.masses[:, np.newaxis], axis=0) / total_mass
        velocities -= com_velocity

        self._state.velocities = velocities
        self._update_temperature()

        logger.info(f"Initialized velocities at T={self._state.temperature:.1f} K")

    def minimize_energy(
        self,
        max_steps: int = 1000,
        tolerance: float = 1.0,
        step_size: float = 0.01,
    ) -> float:
        """Perform energy minimization.

        Args:
            max_steps: Maximum minimization steps
            tolerance: Force tolerance for convergence
            step_size: Step size for steepest descent

        Returns:
            Final potential energy
        """
        logger.info("Starting energy minimization")

        for step in range(max_steps):
            self._compute_forces()

            # Maximum force
            max_force = np.max(np.linalg.norm(self._state.forces, axis=1))

            if max_force < tolerance:
                logger.info(f"Minimization converged at step {step}")
                break

            # Steepest descent step
            self._state.positions += step_size * self._state.forces / max_force

            # Apply PBC
            if self.config.pbc:
                self._apply_pbc()

            if step % 100 == 0:
                logger.debug(
                    f"Minimization step {step}: E={self._state.potential_energy:.2f}, "
                    f"max_force={max_force:.2f}"
                )

        return self._state.potential_energy

    def _compute_forces(self) -> None:
        """Compute forces on all atoms."""
        num_atoms = self._state.num_atoms
        positions = self._state.positions
        forces = np.zeros_like(positions)
        potential = 0.0

        cutoff = self.config.cutoff
        cutoff_sq = cutoff**2
        box = np.array(self.config.box_size)

        # Lennard-Jones interactions
        for i in range(num_atoms):
            elem_i = self._elements[i]
            eps_i, sig_i = self.LJ_PARAMS.get(elem_i, (0.1, 3.4))

            for j in range(i + 1, num_atoms):
                elem_j = self._elements[j]
                eps_j, sig_j = self.LJ_PARAMS.get(elem_j, (0.1, 3.4))

                # Lorentz-Berthelot combining rules
                epsilon = np.sqrt(eps_i * eps_j)
                sigma = (sig_i + sig_j) / 2

                # Distance vector
                rij = positions[j] - positions[i]

                # Apply minimum image convention for PBC
                if self.config.pbc:
                    rij -= box * np.round(rij / box)

                r_sq = np.sum(rij**2)

                if r_sq > cutoff_sq or r_sq < 0.01:
                    continue

                r = np.sqrt(r_sq)
                r_inv = 1.0 / r
                sig_r = sigma * r_inv
                sig_r6 = sig_r**6
                sig_r12 = sig_r6**2

                # Lennard-Jones energy
                lj_energy = 4 * epsilon * (sig_r12 - sig_r6)
                potential += lj_energy

                # Lennard-Jones force
                lj_force = 24 * epsilon * r_inv * (2 * sig_r12 - sig_r6)
                force_vec = lj_force * rij * r_inv

                forces[i] -= force_vec
                forces[j] += force_vec

        self._state.forces = forces
        self._state.potential_energy = potential

    def _apply_pbc(self) -> None:
        """Apply periodic boundary conditions."""
        box = np.array(self.config.box_size)
        self._state.positions = self._state.positions % box

    def _update_temperature(self) -> None:
        """Update instantaneous temperature from kinetic energy."""
        num_atoms = self._state.num_atoms
        if num_atoms == 0:
            return

        # Calculate kinetic energy
        ke = 0.0
        for i in range(num_atoms):
            v_sq = np.sum(self._state.velocities[i] ** 2)
            ke += 0.5 * self._state.masses[i] * v_sq

        self._state.kinetic_energy = ke

        # Temperature from equipartition theorem
        dof = 3 * num_atoms - 3  # Subtract COM motion
        if dof > 0:
            self._state.temperature = 2 * ke / (dof * self.KB)

    def _apply_thermostat(self) -> None:
        """Apply temperature control."""
        if self.config.thermostat == Thermostat.NONE:
            return

        T_target = self.config.temperature
        T_current = self._state.temperature

        if T_current < 1e-6:
            return

        if self.config.thermostat == Thermostat.BERENDSEN:
            # Berendsen thermostat
            dt = self.config.timestep
            tau = self.config.thermostat_tau
            lambda_factor = np.sqrt(1 + (dt / tau) * (T_target / T_current - 1))
            self._state.velocities *= lambda_factor

        elif self.config.thermostat == Thermostat.VELOCITY_RESCALE:
            # Simple velocity rescaling
            lambda_factor = np.sqrt(T_target / T_current)
            self._state.velocities *= lambda_factor

        self._update_temperature()

    def step(self) -> SimulationState:
        """Perform one MD integration step.

        Returns:
            Updated simulation state
        """
        dt = self.config.timestep

        if self.config.integrator == Integrator.VELOCITY_VERLET:
            # Velocity Verlet integrator

            # Half-step velocity update
            self._state.velocities += (
                0.5 * dt * self._state.forces / self._state.masses[:, np.newaxis]
            )

            # Full-step position update
            self._state.positions += dt * self._state.velocities

            # Apply PBC
            if self.config.pbc:
                self._apply_pbc()

            # Compute new forces
            self._compute_forces()

            # Half-step velocity update
            self._state.velocities += (
                0.5 * dt * self._state.forces / self._state.masses[:, np.newaxis]
            )

        elif self.config.integrator == Integrator.LEAPFROG:
            # Leapfrog integrator

            # Compute forces
            self._compute_forces()

            # Full-step velocity update
            self._state.velocities += dt * self._state.forces / self._state.masses[:, np.newaxis]

            # Full-step position update
            self._state.positions += dt * self._state.velocities

            if self.config.pbc:
                self._apply_pbc()

        # Apply thermostat
        self._apply_thermostat()

        # Update state
        self._state.step += 1
        self._state.time += dt
        self._update_temperature()

        # Call callbacks
        for callback in self._callbacks:
            callback(self._state)

        return self._state

    def run(
        self,
        num_steps: int,
        save_frequency: int = 100,
        callback: Optional[Callable[[SimulationState], None]] = None,
    ) -> list[TrajectoryFrame]:
        """Run MD simulation for specified number of steps.

        Args:
            num_steps: Number of integration steps
            save_frequency: Save trajectory every N steps
            callback: Optional callback for each step

        Returns:
            List of trajectory frames
        """
        self._running = True
        self._trajectory = []

        logger.info(f"Starting MD simulation for {num_steps} steps")

        # Initialize forces
        self._compute_forces()

        for step in range(num_steps):
            if not self._running:
                break

            self.step()

            if callback:
                callback(self._state)

            # Save trajectory frame
            if step % save_frequency == 0:
                frame = TrajectoryFrame(
                    step=self._state.step,
                    time=self._state.time,
                    positions=self._state.positions.copy(),
                    energies={
                        "kinetic": self._state.kinetic_energy,
                        "potential": self._state.potential_energy,
                        "total": self._state.total_energy,
                    },
                )
                self._trajectory.append(frame)

            if step % 1000 == 0:
                logger.info(
                    f"Step {step}: T={self._state.temperature:.1f} K, "
                    f"E={self._state.total_energy:.2f} kcal/mol"
                )

        logger.info(f"Simulation complete: {len(self._trajectory)} frames saved")
        return self._trajectory

    def stop(self) -> None:
        """Stop running simulation."""
        self._running = False

    def add_callback(self, callback: Callable[[SimulationState], None]) -> None:
        """Add callback for simulation steps.

        Args:
            callback: Function called after each step
        """
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[SimulationState], None]) -> None:
        """Remove callback.

        Args:
            callback: Callback to remove
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def get_state(self) -> SimulationState:
        """Get current simulation state.

        Returns:
            Current SimulationState
        """
        return self._state

    def get_trajectory(self) -> list[TrajectoryFrame]:
        """Get saved trajectory.

        Returns:
            List of TrajectoryFrame
        """
        return self._trajectory

    def export_trajectory_xyz(self) -> str:
        """Export trajectory in XYZ format.

        Returns:
            XYZ format string
        """
        lines = []

        for frame in self._trajectory:
            num_atoms = len(frame.positions)
            lines.append(str(num_atoms))
            lines.append(f"Step {frame.step}, Time {frame.time:.3f} ps")

            for i, pos in enumerate(frame.positions):
                elem = self._elements[i] if i < len(self._elements) else "X"
                lines.append(f"{elem} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")

        return "\n".join(lines)

    def calculate_rmsd(self, reference_positions: Optional[np.ndarray] = None) -> float:
        """Calculate RMSD from reference structure.

        Args:
            reference_positions: Reference coordinates (uses initial if not specified)

        Returns:
            RMSD in Angstroms
        """
        if reference_positions is None:
            if not self._trajectory:
                return 0.0
            reference_positions = self._trajectory[0].positions

        diff = self._state.positions - reference_positions
        return np.sqrt(np.mean(np.sum(diff**2, axis=1)))

    def calculate_radius_of_gyration(self) -> float:
        """Calculate radius of gyration.

        Returns:
            Rg in Angstroms
        """
        positions = self._state.positions
        masses = self._state.masses

        # Center of mass
        total_mass = np.sum(masses)
        com = np.sum(positions * masses[:, np.newaxis], axis=0) / total_mass

        # Radius of gyration
        diff = positions - com
        r_sq = np.sum(diff**2, axis=1)
        rg_sq = np.sum(masses * r_sq) / total_mass

        return np.sqrt(rg_sq)

    def set_positions(self, positions: np.ndarray) -> None:
        """Set atomic positions (for interactive manipulation).

        Args:
            positions: New positions array
        """
        if positions.shape != self._state.positions.shape:
            raise ValueError("Position array shape mismatch")
        self._state.positions = positions.copy()

    def apply_force(self, atom_index: int, force: tuple[float, float, float]) -> None:
        """Apply external force to atom (for haptic feedback).

        Args:
            atom_index: Index of atom
            force: Force vector (kcal/mol/Angstrom)
        """
        if 0 <= atom_index < self._state.num_atoms:
            self._state.forces[atom_index] += np.array(force)
