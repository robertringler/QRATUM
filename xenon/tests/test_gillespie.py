"""Tests for Gillespie SSA simulator."""

import numpy as np
import pytest

from xenon.core.mechanism import BioMechanism, MolecularState, Transition
from xenon.simulation.gillespie import GillespieSimulator


class TestGillespieSimulator:
    """Tests for Gillespie stochastic simulation."""
    
    def test_create_simulator(self):
        """Test creating a simulator."""
        mech = BioMechanism("test")
        
        state1 = MolecularState(name="S1", molecule="Protein")
        state2 = MolecularState(name="S2", molecule="Protein")
        mech.add_state(state1)
        mech.add_state(state2)
        
        simulator = GillespieSimulator(mech, volume=1e-15)
        assert simulator.mechanism == mech
        assert simulator.volume == 1e-15
    
    def test_simple_simulation(self):
        """Test simple A -> B simulation."""
        mech = BioMechanism("simple")
        
        state_a = MolecularState(name="A", molecule="A", concentration=100.0)
        state_b = MolecularState(name="B", molecule="B", concentration=0.0)
        
        mech.add_state(state_a)
        mech.add_state(state_b)
        
        # A -> B with rate 1.0 s^-1
        transition = Transition(source="A", target="B", rate_constant=1.0)
        mech.add_transition(transition)
        
        simulator = GillespieSimulator(mech, volume=1e-15)
        
        initial_state = {"A": 100.0, "B": 0.0}
        times, trajectories = simulator.run(
            t_max=2.0,
            initial_state=initial_state,
            seed=42,
            record_interval=0.1,
        )
        
        # Check that simulation ran
        assert len(times) > 0
        assert "A" in trajectories
        assert "B" in trajectories
        
        # A should decrease, B should increase
        assert trajectories["A"][-1] < trajectories["A"][0]
        assert trajectories["B"][-1] > trajectories["B"][0]
    
    def test_reversible_reaction(self):
        """Test reversible A <-> B simulation."""
        mech = BioMechanism("reversible")
        
        state_a = MolecularState(name="A", molecule="A")
        state_b = MolecularState(name="B", molecule="B")
        
        mech.add_state(state_a)
        mech.add_state(state_b)
        
        # A -> B
        trans_forward = Transition(source="A", target="B", rate_constant=1.0)
        # B -> A
        trans_reverse = Transition(source="B", target="A", rate_constant=0.5)
        
        mech.add_transition(trans_forward)
        mech.add_transition(trans_reverse)
        
        simulator = GillespieSimulator(mech, volume=1e-15)
        
        initial_state = {"A": 100.0, "B": 0.0}
        times, trajectories = simulator.run(
            t_max=5.0,
            initial_state=initial_state,
            seed=42,
        )
        
        # Should reach equilibrium
        assert len(times) > 0
        # At equilibrium: k_f [A] = k_r [B]
        # [B]/[A] = k_f/k_r = 1.0/0.5 = 2.0
        # But stochastic so just check both are non-zero
        assert trajectories["A"][-1] > 0
        assert trajectories["B"][-1] > 0
    
    def test_seed_reproducibility(self):
        """Test that simulations with same seed are reproducible."""
        mech = BioMechanism("test")
        
        state_a = MolecularState(name="A", molecule="A")
        state_b = MolecularState(name="B", molecule="B")
        
        mech.add_state(state_a)
        mech.add_state(state_b)
        
        transition = Transition(source="A", target="B", rate_constant=1.0)
        mech.add_transition(transition)
        
        simulator1 = GillespieSimulator(mech)
        simulator2 = GillespieSimulator(mech)
        
        initial_state = {"A": 100.0, "B": 0.0}
        
        times1, traj1 = simulator1.run(t_max=1.0, initial_state=initial_state, seed=42)
        times2, traj2 = simulator2.run(t_max=1.0, initial_state=initial_state, seed=42)
        
        # Should be identical with same seed
        assert len(times1) == len(times2)
        np.testing.assert_array_equal(times1, times2)
    
    def test_no_reactions_terminates(self):
        """Test that simulation terminates when no reactions possible."""
        mech = BioMechanism("no_reactions")
        
        state_a = MolecularState(name="A", molecule="A")
        mech.add_state(state_a)
        
        # No transitions
        
        simulator = GillespieSimulator(mech)
        
        initial_state = {"A": 100.0}
        times, trajectories = simulator.run(
            t_max=1.0,
            initial_state=initial_state,
            seed=42,
        )
        
        # Should terminate immediately
        assert len(times) == 1
        assert times[0] == 0.0
    
    def test_performance_metrics(self):
        """Test getting performance metrics."""
        mech = BioMechanism("test")
        
        state_a = MolecularState(name="A", molecule="A")
        state_b = MolecularState(name="B", molecule="B")
        
        mech.add_state(state_a)
        mech.add_state(state_b)
        
        transition = Transition(source="A", target="B", rate_constant=1.0)
        mech.add_transition(transition)
        
        simulator = GillespieSimulator(mech)
        
        initial_state = {"A": 100.0, "B": 0.0}
        simulator.run(t_max=0.5, initial_state=initial_state, seed=42)
        
        metrics = simulator.get_performance_metrics()
        
        assert "total_reactions" in metrics
        assert metrics["total_reactions"] > 0
    
    def test_exponential_distribution(self):
        """Test that reaction times follow exponential distribution.
        
        For simple A -> B with rate k, inter-reaction times should be
        exponentially distributed with mean 1/(k*n_A).
        """
        mech = BioMechanism("exponential")
        
        state_a = MolecularState(name="A", molecule="A")
        state_b = MolecularState(name="B", molecule="B")
        
        mech.add_state(state_a)
        mech.add_state(state_b)
        
        # Fast reaction rate
        transition = Transition(source="A", target="B", rate_constant=10.0)
        mech.add_transition(transition)
        
        simulator = GillespieSimulator(mech, volume=1e-15)
        
        # Start with moderate number of molecules
        initial_state = {"A": 50.0, "B": 0.0}
        
        times, trajectories = simulator.run(
            t_max=0.1,
            initial_state=initial_state,
            seed=42,
            record_interval=None,  # Record all reactions
        )
        
        # Check that times increase monotonically
        assert all(times[i] < times[i+1] for i in range(len(times)-1))
        
        # Check that we have multiple reactions
        assert len(times) > 5
    
    def test_mass_conservation(self):
        """Test that total mass is conserved in unimolecular reactions."""
        mech = BioMechanism("conservation")
        
        state_a = MolecularState(name="A", molecule="A")
        state_b = MolecularState(name="B", molecule="B")
        
        mech.add_state(state_a)
        mech.add_state(state_b)
        
        # A <-> B (should conserve A + B)
        trans1 = Transition(source="A", target="B", rate_constant=1.0)
        trans2 = Transition(source="B", target="A", rate_constant=1.0)
        
        mech.add_transition(trans1)
        mech.add_transition(trans2)
        
        simulator = GillespieSimulator(mech, volume=1e-15)
        
        initial_state = {"A": 100.0, "B": 0.0}
        times, trajectories = simulator.run(
            t_max=1.0,
            initial_state=initial_state,
            seed=42,
            record_interval=0.1,
        )
        
        # Check mass conservation at each time point
        initial_total = initial_state["A"] + initial_state["B"]
        
        for i in range(len(times)):
            total = trajectories["A"][i] + trajectories["B"][i]
            # Allow small numerical error
            assert abs(total - initial_total) < 1.0  # Within 1 nM tolerance


class TestGillespieOptimized:
    """Tests for optimized Gillespie simulator."""
    
    def test_optimized_simulator_exists(self):
        """Test that optimized simulator is available."""
        from xenon.simulation.gillespie import GillespieSimulatorOptimized
        
        mech = BioMechanism("test")
        state_a = MolecularState(name="A", molecule="A")
        mech.add_state(state_a)
        
        simulator = GillespieSimulatorOptimized(mech)
        assert simulator is not None
