"""Tests for XENON visualization adapters."""

import numpy as np
import pytest

from qubic.visualization.core.data_model import VisualizationData
from xenon.adapters.bio_mechanism_adapter import BioMechanismAdapter
from xenon.adapters.molecular_state_adapter import MolecularStateAdapter
from xenon.adapters.transition_adapter import TransitionAdapter
from xenon.core.mechanism import BioMechanism, MolecularState, Transition


class TestBioMechanismAdapter:
    """Tests for BioMechanismAdapter."""

    def test_to_viz_model(self):
        """Test conversion to visualization model."""

        # Create sample mechanism
        states = [
            MolecularState("S1", "ProteinA", -10.0, 0.5),
            MolecularState("S2", "ProteinB", -15.0, 0.3),
            MolecularState("S3", "ProteinC", -12.0, 0.2),
        ]
        transitions = [
            Transition("S1", "S2", 1.5, -5.0, 20.0),
            Transition("S2", "S3", 2.0, -3.0, 15.0),
        ]
        mechanism = BioMechanism("MECH_001", states, transitions, 0.9)

        adapter = BioMechanismAdapter(mechanism)
        viz_model = adapter.to_viz_model()

        assert "nodes" in viz_model
        assert "edges" in viz_model
        assert "evidence_score" in viz_model
        assert len(viz_model["nodes"]) == 3
        assert len(viz_model["edges"]) == 2
        assert viz_model["evidence_score"] == 0.9

        # Check node structure
        node = viz_model["nodes"][0]
        assert "id" in node
        assert "protein" in node
        assert "free_energy" in node
        assert "concentration" in node

        # Check edge structure
        edge = viz_model["edges"][0]
        assert "source" in edge
        assert "target" in edge
        assert "rate" in edge
        assert "delta_g" in edge

    def test_to_3d_network_spring(self):
        """Test 3D network generation with spring layout."""

        states = [
            MolecularState("S1", "ProteinA", -10.0),
            MolecularState("S2", "ProteinB", -15.0),
        ]
        transitions = [Transition("S1", "S2", 1.5)]
        mechanism = BioMechanism("MECH_001", states, transitions)

        adapter = BioMechanismAdapter(mechanism)
        viz_data = adapter.to_3d_network(layout="spring", scale=5.0)

        assert isinstance(viz_data, VisualizationData)
        assert len(viz_data.vertices) == 2
        assert viz_data.vertices.shape[1] == 3  # 3D coordinates
        assert "free_energy" in viz_data.scalar_fields
        assert len(viz_data.scalar_fields["free_energy"]) == 2

    def test_to_3d_network_circular(self):
        """Test 3D network generation with circular layout."""

        states = [MolecularState(f"S{i}", f"Protein{i}", -10.0 - i) for i in range(4)]
        transitions = [Transition(f"S{i}", f"S{i + 1}", 1.0) for i in range(3)]
        mechanism = BioMechanism("MECH_001", states, transitions)

        adapter = BioMechanismAdapter(mechanism)
        viz_data = adapter.to_3d_network(layout="circular", scale=10.0)

        assert isinstance(viz_data, VisualizationData)
        assert len(viz_data.vertices) == 4
        # Circular layout should have z=0
        assert np.allclose(viz_data.vertices[:, 2], 0.0)

    def test_to_3d_network_hierarchical(self):
        """Test 3D network generation with hierarchical layout."""

        states = [MolecularState(f"S{i}", f"Protein{i}", -10.0) for i in range(3)]
        transitions = [
            Transition("S0", "S1", 1.0),
            Transition("S1", "S2", 1.0),
        ]
        mechanism = BioMechanism("MECH_001", states, transitions)

        adapter = BioMechanismAdapter(mechanism)
        viz_data = adapter.to_3d_network(layout="hierarchical", scale=5.0)

        assert isinstance(viz_data, VisualizationData)
        assert len(viz_data.vertices) == 3

    def test_invalid_layout(self):
        """Test error handling for invalid layout."""

        states = [MolecularState("S1", "ProteinA", -10.0)]
        mechanism = BioMechanism("MECH_001", states, [])

        adapter = BioMechanismAdapter(mechanism)
        with pytest.raises(ValueError, match="Unknown layout"):
            adapter.to_3d_network(layout="invalid")


class TestMolecularStateAdapter:
    """Tests for MolecularStateAdapter."""

    def test_to_viz_model(self):
        """Test conversion to visualization model."""

        state = MolecularState("S1", "ProteinA", -10.0, 0.5)
        adapter = MolecularStateAdapter(state)
        viz_model = adapter.to_viz_model()

        assert viz_model["id"] == "S1"
        assert viz_model["protein"] == "ProteinA"
        assert viz_model["free_energy"] == -10.0
        assert viz_model["concentration"] == 0.5

    def test_to_energy_surface(self):
        """Test energy surface generation."""

        state = MolecularState("S1", "ProteinA", -10.0)
        adapter = MolecularStateAdapter(state)
        viz_data = adapter.to_energy_surface(resolution=20)

        assert isinstance(viz_data, VisualizationData)
        assert len(viz_data.vertices) == 20 * 20
        assert viz_data.vertices.shape[1] == 3
        assert "energy" in viz_data.scalar_fields
        assert viz_data.metadata["state_id"] == "S1"

    def test_to_point_cloud(self):
        """Test point cloud generation."""

        state = MolecularState("S1", "ProteinA", -15.0, 0.8)
        adapter = MolecularStateAdapter(state)
        viz_data = adapter.to_point_cloud(num_points=100)

        assert isinstance(viz_data, VisualizationData)
        assert len(viz_data.vertices) == 100
        assert viz_data.vertices.shape[1] == 3
        assert "free_energy" in viz_data.scalar_fields
        assert "concentration" in viz_data.scalar_fields
        assert viz_data.metadata["representation"] == "point_cloud"


class TestTransitionAdapter:
    """Tests for TransitionAdapter."""

    def test_to_viz_model(self):
        """Test conversion to visualization model."""

        transition = Transition("S1", "S2", 1.5, -5.0, 20.0)
        adapter = TransitionAdapter(transition)
        viz_model = adapter.to_viz_model()

        assert viz_model["source"] == "S1"
        assert viz_model["target"] == "S2"
        assert viz_model["rate"] == 1.5
        assert viz_model["delta_g"] == -5.0
        assert viz_model["activation_energy"] == 20.0

    def test_to_arrow(self):
        """Test arrow generation."""

        transition = Transition("S1", "S2", 1.5, -5.0, 20.0)
        adapter = TransitionAdapter(transition)

        source_pos = np.array([0.0, 0.0, 0.0])
        target_pos = np.array([10.0, 0.0, 0.0])

        viz_data = adapter.to_arrow(source_pos, target_pos, arrow_width=0.2)

        assert isinstance(viz_data, VisualizationData)
        assert len(viz_data.vertices) > 0
        assert viz_data.vertices.shape[1] == 3
        assert "rate_constant" in viz_data.scalar_fields
        assert viz_data.metadata["transition_type"] == "arrow"

    def test_to_energy_barrier(self):
        """Test energy barrier generation."""

        transition = Transition("S1", "S2", 1.5, -5.0, 20.0)
        adapter = TransitionAdapter(transition)
        viz_data = adapter.to_energy_barrier(num_points=30)

        assert isinstance(viz_data, VisualizationData)
        assert len(viz_data.vertices) == 30
        assert "energy" in viz_data.scalar_fields
        assert "reaction_coordinate" in viz_data.scalar_fields
        assert viz_data.metadata["transition_type"] == "energy_barrier"
        assert viz_data.metadata["activation_energy"] == 20.0

    def test_get_color_by_rate(self):
        """Test color mapping by rate constant."""

        transition = Transition("S1", "S2", 5.0)
        adapter = TransitionAdapter(transition)

        color = adapter.get_color_by_rate(min_rate=0.0, max_rate=10.0)

        assert isinstance(color, tuple)
        assert len(color) == 3
        assert all(0.0 <= c <= 1.0 for c in color)

    def test_get_thickness_by_rate(self):
        """Test thickness mapping by rate constant."""

        transition = Transition("S1", "S2", 5.0)
        adapter = TransitionAdapter(transition)

        thickness = adapter.get_thickness_by_rate()

        assert isinstance(thickness, float)
        assert 0.1 <= thickness <= 1.0
