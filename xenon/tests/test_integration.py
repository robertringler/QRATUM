"""Integration tests for XENON visualization pipeline."""

import pytest

from qubic.visualization.adapters.xenon_adapter import XenonSimulationAdapter
from qubic.visualization.core.data_model import VisualizationData
from xenon.core.mechanism import BioMechanism, MolecularState, Transition


class TestXenonIntegration:
    """Integration tests for XENON with visualization pipeline."""

    def test_xenon_adapter_with_mechanism(self):
        """Test XenonSimulationAdapter with BioMechanism."""
        # Create sample mechanism
        states = [
            MolecularState("S1", "ProteinA", -10.0),
            MolecularState("S2", "ProteinB", -15.0),
        ]
        transitions = [Transition("S1", "S2", 1.5, -5.0, 20.0)]
        mechanism = BioMechanism("MECH_001", states, transitions)

        # Load with adapter
        adapter = XenonSimulationAdapter(layout="spring", scale=10.0)
        viz_data = adapter.load_data(mechanism)

        assert isinstance(viz_data, VisualizationData)
        assert len(viz_data.vertices) == 2
        assert "free_energy" in viz_data.scalar_fields

    def test_xenon_adapter_with_state(self):
        """Test XenonSimulationAdapter with MolecularState."""
        state = MolecularState("S1", "ProteinA", -10.0, 0.5)

        adapter = XenonSimulationAdapter()
        viz_data = adapter.load_data(state)

        assert isinstance(viz_data, VisualizationData)
        assert "energy" in viz_data.scalar_fields

    def test_xenon_adapter_with_transition(self):
        """Test XenonSimulationAdapter with Transition."""
        transition = Transition("S1", "S2", 1.5, -5.0, 20.0)

        adapter = XenonSimulationAdapter(scale=5.0)
        viz_data = adapter.load_data(transition)

        assert isinstance(viz_data, VisualizationData)
        assert "rate_constant" in viz_data.scalar_fields

    def test_validate_source(self):
        """Test source validation."""
        adapter = XenonSimulationAdapter()

        # Valid sources
        state = MolecularState("S1", "ProteinA", -10.0)
        assert adapter.validate_source(state) is True

        transition = Transition("S1", "S2", 1.5)
        assert adapter.validate_source(transition) is True

        mechanism = BioMechanism("M1", [state], [transition])
        assert adapter.validate_source(mechanism) is True

        # Invalid sources
        assert adapter.validate_source("invalid") is False
        assert adapter.validate_source(123) is False
        assert adapter.validate_source(None) is False

    def test_load_mechanism_timeseries(self):
        """Test loading time series of mechanisms."""
        # Create multiple mechanism snapshots
        mechanisms = []
        for i in range(3):
            states = [MolecularState(f"S{j}", f"Protein{j}", -10.0 - i) for j in range(2)]
            transitions = [Transition("S0", "S1", 1.0 + i * 0.1)]
            mechanism = BioMechanism(f"MECH_{i}", states, transitions)
            mechanisms.append(mechanism)

        adapter = XenonSimulationAdapter()
        viz_data_list = adapter.load_mechanism_timeseries(mechanisms)

        assert len(viz_data_list) == 3
        for viz_data in viz_data_list:
            assert isinstance(viz_data, VisualizationData)

    def test_unsupported_source_type(self):
        """Test error handling for unsupported source types."""
        adapter = XenonSimulationAdapter()

        with pytest.raises(ValueError, match="Unsupported XENON source type"):
            adapter.load_data({"invalid": "dict"})
