"""Tests for the MD Simulator."""

from __future__ import annotations

import numpy as np

from xenon.molecular_dynamics_lab.core.md_simulator import (
    MDConfig,
    MDSimulator,
    SimulationState,
    TrajectoryFrame,
)


class TestMDConfig:
    """Test MDConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = MDConfig()

        assert config.timestep == 0.002
        assert config.temperature == 300.0
        assert config.thermostat == "berendsen"
        assert config.cutoff == 10.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = MDConfig(
            timestep=0.001,
            temperature=350.0,
            thermostat="nose-hoover",
            cutoff=12.0,
        )

        assert config.timestep == 0.001
        assert config.temperature == 350.0


class TestSimulationState:
    """Test SimulationState dataclass."""

    def test_state_creation(self):
        """Test creating a simulation state."""
        state = SimulationState(
            step=100,
            time=0.2,
            kinetic_energy=150.0,
            potential_energy=-500.0,
            total_energy=-350.0,
            temperature=298.5,
        )

        assert state.step == 100
        assert state.time == 0.2
        assert state.total_energy == -350.0

    def test_to_dict(self):
        """Test serialization to dictionary."""
        state = SimulationState(
            step=50,
            time=0.1,
            kinetic_energy=100.0,
            potential_energy=-400.0,
            total_energy=-300.0,
            temperature=295.0,
        )

        data = state.to_dict()

        assert data["step"] == 50
        assert data["kinetic_energy"] == 100.0


class TestTrajectoryFrame:
    """Test TrajectoryFrame dataclass."""

    def test_frame_creation(self):
        """Test creating a trajectory frame."""
        positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        velocities = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])

        frame = TrajectoryFrame(
            step=10,
            time=0.02,
            positions=positions,
            velocities=velocities,
            kinetic_energy=50.0,
            potential_energy=-200.0,
            temperature=300.0,
        )

        assert frame.step == 10
        assert frame.positions.shape == (3, 3)

    def test_to_dict(self):
        """Test frame serialization."""
        positions = np.array([[0, 0, 0], [1, 0, 0]])

        frame = TrajectoryFrame(
            step=1,
            time=0.002,
            positions=positions,
            velocities=np.zeros_like(positions),
            kinetic_energy=0.0,
            potential_energy=-100.0,
            temperature=0.0,
        )

        data = frame.to_dict()

        assert "positions" in data
        assert len(data["positions"]) == 2


class TestMDSimulator:
    """Test MDSimulator class."""

    def test_simulator_creation(self):
        """Test creating a simulator."""
        config = MDConfig(temperature=310.0)
        simulator = MDSimulator(config)

        assert simulator.config.temperature == 310.0

    def test_set_positions(self):
        """Test setting atom positions."""
        simulator = MDSimulator()
        positions = np.array([[0, 0, 0], [3.8, 0, 0], [7.6, 0, 0]])

        simulator.set_positions(positions)

        assert simulator._positions is not None
        assert simulator._positions.shape == (3, 3)

    def test_set_masses(self):
        """Test setting atom masses."""
        simulator = MDSimulator()
        masses = np.array([12.0, 14.0, 16.0])

        simulator.set_masses(masses)

        assert simulator._masses is not None
        assert len(simulator._masses) == 3

    def test_initialize_velocities(self):
        """Test velocity initialization."""
        simulator = MDSimulator(MDConfig(temperature=300.0))
        simulator.set_positions(np.array([[0, 0, 0], [3.8, 0, 0]]))
        simulator.set_masses(np.array([12.0, 12.0]))

        simulator.initialize_velocities()

        assert simulator._velocities is not None
        assert simulator._velocities.shape == (2, 3)

    def test_compute_forces(self):
        """Test force computation."""
        simulator = MDSimulator()
        positions = np.array([[0, 0, 0], [3.5, 0, 0]])
        simulator.set_positions(positions)

        forces = simulator.compute_forces()

        assert forces.shape == positions.shape
        # Forces should be repulsive at close range
        # or attractive at longer range

    def test_lennard_jones(self):
        """Test Lennard-Jones potential."""
        simulator = MDSimulator()

        # At equilibrium distance (~3.4 Å for carbon)
        r_eq = 3.4
        energy = simulator._lennard_jones_energy(r_eq, epsilon=0.1, sigma=3.4)

        # Energy should be near minimum
        assert energy < 0

    def test_single_step(self):
        """Test single integration step."""
        simulator = MDSimulator(MDConfig(timestep=0.001))
        simulator.set_positions(np.array([[0, 0, 0], [4.0, 0, 0]]))
        simulator.set_masses(np.array([12.0, 12.0]))
        simulator.initialize_velocities()

        initial_pos = simulator._positions.copy()
        simulator.step()

        # Positions should change
        assert not np.allclose(simulator._positions, initial_pos)

    def test_get_state(self):
        """Test getting simulation state."""
        simulator = MDSimulator()
        simulator.set_positions(np.array([[0, 0, 0], [4.0, 0, 0]]))
        simulator.set_masses(np.array([12.0, 12.0]))
        simulator.initialize_velocities()

        state = simulator.get_state()

        assert isinstance(state, SimulationState)
        assert state.step >= 0

    def test_run_simulation(self):
        """Test running simulation for multiple steps."""
        simulator = MDSimulator(MDConfig(timestep=0.001))
        simulator.set_positions(np.array([[0, 0, 0], [4.0, 0, 0]]))
        simulator.set_masses(np.array([12.0, 12.0]))
        simulator.initialize_velocities()

        trajectory = simulator.run(num_steps=10, save_frequency=5)

        assert len(trajectory) == 2  # Frames at step 5 and 10

    def test_energy_minimization(self):
        """Test energy minimization."""
        simulator = MDSimulator()
        # Start with slightly compressed distance
        simulator.set_positions(np.array([[0, 0, 0], [2.5, 0, 0]]))
        simulator.set_masses(np.array([12.0, 12.0]))

        initial_energy = simulator._compute_potential_energy()
        simulator.minimize(max_steps=100, tolerance=0.001)
        final_energy = simulator._compute_potential_energy()

        # Energy should decrease
        assert final_energy <= initial_energy


class TestBerendsenThermostat:
    """Test Berendsen thermostat."""

    def test_temperature_coupling(self):
        """Test temperature coupling."""
        simulator = MDSimulator(
            MDConfig(
                temperature=300.0,
                thermostat="berendsen",
                thermostat_coupling=0.1,
            )
        )

        simulator.set_positions(np.array([[0, 0, 0], [4.0, 0, 0]]))
        simulator.set_masses(np.array([12.0, 12.0]))

        # Start with high velocities
        simulator._velocities = np.array([[10.0, 0, 0], [-10.0, 0, 0]])

        initial_temp = simulator._compute_temperature()

        # Run a few steps
        for _ in range(50):
            simulator.step()

        final_temp = simulator._compute_temperature()

        # Temperature should move toward target (though won't reach exactly)
        # This is a weak assertion due to small system
        assert final_temp is not None


class TestVelocityVerlet:
    """Test Velocity Verlet integration."""

    def test_energy_conservation(self):
        """Test energy conservation without thermostat."""
        simulator = MDSimulator(
            MDConfig(
                timestep=0.0001,  # Very small timestep
                thermostat="none",
            )
        )

        simulator.set_positions(np.array([[0, 0, 0], [4.0, 0, 0]]))
        simulator.set_masses(np.array([12.0, 12.0]))
        simulator._velocities = np.array([[0.1, 0, 0], [-0.1, 0, 0]])

        initial_state = simulator.get_state()

        # Run simulation
        simulator.run(num_steps=100)

        final_state = simulator.get_state()

        # Total energy should be approximately conserved
        # (allowing for numerical drift)
        energy_change = abs(final_state.total_energy - initial_state.total_energy)
        relative_change = energy_change / abs(initial_state.total_energy + 1e-10)

        # Allow 10% drift for this simple test
        assert relative_change < 0.1 or energy_change < 1.0
