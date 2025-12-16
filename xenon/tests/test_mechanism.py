"""Tests for mechanism representation and validation."""

import pytest

from xenon.core.mechanism import BioMechanism, MolecularState, Transition
from xenon.core.mechanism_graph import MechanismGraph


class TestMolecularState:
    """Tests for MolecularState."""

    def test_create_state(self):
        """Test creating a molecular state."""
        state = MolecularState(
            name="S1",
            molecule="TestProtein",
            properties={"phosphorylated": True},
            concentration=100.0,
            free_energy=-10.0,
        )

        assert state.name == "S1"
        assert state.molecule == "TestProtein"
        assert state.concentration == 100.0
        assert state.free_energy == -10.0

    def test_state_serialization(self):
        """Test state to_dict."""
        state = MolecularState(
            name="S1",
            molecule="Protein",
            free_energy=-10.0,
        )

        state_dict = state.to_dict()
        assert state_dict["name"] == "S1"
        assert state_dict["molecule"] == "Protein"
        assert state_dict["free_energy"] == -10.0


class TestTransition:
    """Tests for Transition."""

    def test_create_transition(self):
        """Test creating a transition."""
        transition = Transition(
            source="S1",
            target="S2",
            rate_constant=1.5e-3,
            activation_energy=15.0,
        )

        assert transition.source == "S1"
        assert transition.target == "S2"
        assert transition.rate_constant == 1.5e-3

    def test_reversible_transition(self):
        """Test reversible transition."""
        transition = Transition(
            source="S1",
            target="S2",
            rate_constant=1.0,
            reversible=True,
            reverse_rate=0.1,
        )

        assert transition.reversible
        assert transition.reverse_rate == 0.1


class TestBioMechanism:
    """Tests for BioMechanism."""

    def test_create_mechanism(self):
        """Test creating a mechanism."""
        mech = BioMechanism("test_mechanism")
        assert mech.name == "test_mechanism"
        assert len(mech._states) == 0
        assert len(mech._transitions) == 0

    def test_add_states(self):
        """Test adding states to mechanism."""
        mech = BioMechanism("test")

        state1 = MolecularState(name="S1", molecule="Protein", free_energy=-10.0)
        state2 = MolecularState(name="S2", molecule="Protein", free_energy=-12.0)

        mech.add_state(state1)
        mech.add_state(state2)

        assert len(mech._states) == 2
        assert "S1" in mech._states
        assert "S2" in mech._states

    def test_add_transition(self):
        """Test adding transition to mechanism."""
        mech = BioMechanism("test")

        state1 = MolecularState(name="S1", molecule="Protein", free_energy=-10.0)
        state2 = MolecularState(name="S2", molecule="Protein", free_energy=-12.0)
        mech.add_state(state1)
        mech.add_state(state2)

        transition = Transition(source="S1", target="S2", rate_constant=1.5e-3)
        mech.add_transition(transition)

        assert len(mech._transitions) == 1

    def test_add_transition_missing_state(self):
        """Test adding transition with missing state raises error."""
        mech = BioMechanism("test")

        state1 = MolecularState(name="S1", molecule="Protein")
        mech.add_state(state1)

        transition = Transition(source="S1", target="S2", rate_constant=1.0)

        with pytest.raises(ValueError, match="Target state"):
            mech.add_transition(transition)

    def test_thermodynamic_feasibility(self):
        """Test thermodynamic feasibility check."""
        mech = BioMechanism("test")

        state1 = MolecularState(name="S1", molecule="Protein", free_energy=-10.0)
        state2 = MolecularState(name="S2", molecule="Protein", free_energy=-12.0)
        mech.add_state(state1)
        mech.add_state(state2)

        # Favorable transition (ΔG < 0)
        transition = Transition(
            source="S1",
            target="S2",
            rate_constant=1.5e-3,
        )
        mech.add_transition(transition)

        is_feasible = mech.is_thermodynamically_feasible()
        assert is_feasible

    def test_conservation_laws(self):
        """Test conservation law validation."""
        mech = BioMechanism("test")

        state1 = MolecularState(name="S1", molecule="Protein")
        state2 = MolecularState(name="S2", molecule="Protein")
        mech.add_state(state1)
        mech.add_state(state2)

        transition = Transition(source="S1", target="S2", rate_constant=1.0)
        mech.add_transition(transition)

        is_valid, violations = mech.validate_conservation_laws()
        assert is_valid

    def test_causal_paths(self):
        """Test finding causal paths."""
        mech = BioMechanism("test")

        state1 = MolecularState(name="S1", molecule="Protein")
        state2 = MolecularState(name="S2", molecule="Protein")
        state3 = MolecularState(name="S3", molecule="Protein")

        mech.add_state(state1)
        mech.add_state(state2)
        mech.add_state(state3)

        trans1 = Transition(source="S1", target="S2", rate_constant=1.0)
        trans2 = Transition(source="S2", target="S3", rate_constant=1.0)

        mech.add_transition(trans1)
        mech.add_transition(trans2)

        paths = mech.get_causal_paths("S1", "S3")
        assert len(paths) > 0

    def test_mechanism_hash(self):
        """Test mechanism hash computation."""
        mech = BioMechanism("test")

        state1 = MolecularState(name="S1", molecule="Protein", free_energy=-10.0)
        state2 = MolecularState(name="S2", molecule="Protein", free_energy=-12.0)
        mech.add_state(state1)
        mech.add_state(state2)

        transition = Transition(source="S1", target="S2", rate_constant=1.5e-3)
        mech.add_transition(transition)

        hash1 = mech.compute_mechanism_hash()
        assert len(hash1) == 64  # SHA256 hex digest

        # Same mechanism should produce same hash
        hash2 = mech.compute_mechanism_hash()
        assert hash1 == hash2

    def test_serialization(self):
        """Test mechanism serialization."""
        mech = BioMechanism("test")

        state1 = MolecularState(name="S1", molecule="Protein", free_energy=-10.0)
        mech.add_state(state1)

        transition = Transition(source="S1", target="S1", rate_constant=1.0)
        mech.add_transition(transition)

        # Serialize
        mech_dict = mech.to_dict()

        # Deserialize
        mech2 = BioMechanism.from_dict(mech_dict)

        assert mech2.name == mech.name
        assert len(mech2._states) == len(mech._states)
        assert len(mech2._transitions) == len(mech._transitions)


class TestMechanismGraph:
    """Tests for MechanismGraph operations."""

    def test_mutate_topology(self):
        """Test topology mutation."""
        mech = BioMechanism("parent")

        state1 = MolecularState(name="S1", molecule="Protein")
        state2 = MolecularState(name="S2", molecule="Protein")
        mech.add_state(state1)
        mech.add_state(state2)

        transition = Transition(source="S1", target="S2", rate_constant=1.0)
        mech.add_transition(transition)

        mutant = MechanismGraph.mutate_topology(mech, mutation_rate=0.5, seed=42)

        assert mutant.name == "parent_mutated"
        assert len(mutant._states) == 2

    def test_extract_subgraph(self):
        """Test subgraph extraction."""
        mech = BioMechanism("full")

        state1 = MolecularState(name="S1", molecule="Protein")
        state2 = MolecularState(name="S2", molecule="Protein")
        state3 = MolecularState(name="S3", molecule="Protein")

        mech.add_state(state1)
        mech.add_state(state2)
        mech.add_state(state3)

        trans1 = Transition(source="S1", target="S2", rate_constant=1.0)
        trans2 = Transition(source="S2", target="S3", rate_constant=1.0)

        mech.add_transition(trans1)
        mech.add_transition(trans2)

        # Extract subgraph with S1 and S2
        subgraph = MechanismGraph.extract_subgraph(mech, {"S1", "S2"})

        assert len(subgraph._states) == 2
        assert len(subgraph._transitions) == 1

    def test_recombine_mechanisms(self):
        """Test mechanism recombination."""
        mech1 = BioMechanism("parent1")
        mech2 = BioMechanism("parent2")

        state1 = MolecularState(name="S1", molecule="Protein")
        state2 = MolecularState(name="S2", molecule="Protein")

        mech1.add_state(state1)
        mech2.add_state(state2)

        child = MechanismGraph.recombine_mechanisms(mech1, mech2, "child")

        assert child.name == "child"
        assert len(child._states) == 2

    def test_detect_cycles(self):
        """Test cycle detection."""
        mech = BioMechanism("cyclic")

        state1 = MolecularState(name="S1", molecule="Protein")
        state2 = MolecularState(name="S2", molecule="Protein")

        mech.add_state(state1)
        mech.add_state(state2)

        trans1 = Transition(source="S1", target="S2", rate_constant=1.0)
        trans2 = Transition(source="S2", target="S1", rate_constant=1.0)

        mech.add_transition(trans1)
        mech.add_transition(trans2)

        cycles = MechanismGraph.detect_cycles(mech)
        assert len(cycles) > 0

    def test_isomorphic(self):
        """Test isomorphism check."""
        mech1 = BioMechanism("mech1")
        mech2 = BioMechanism("mech2")

        # Create identical mechanisms
        for mech in [mech1, mech2]:
            state1 = MolecularState(name="S1", molecule="Protein")
            state2 = MolecularState(name="S2", molecule="Protein")
            mech.add_state(state1)
            mech.add_state(state2)

            transition = Transition(source="S1", target="S2", rate_constant=1.0)
            mech.add_transition(transition)

        is_iso = MechanismGraph.is_isomorphic(mech1, mech2)
        assert is_iso
