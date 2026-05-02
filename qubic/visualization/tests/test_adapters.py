"""Tests for visualization adapters."""

import numpy as np
import pytest

from qubic.visualization.adapters.mesh import MeshAdapter
from qubic.visualization.adapters.quantum import QuantumSimulationAdapter
from qubic.visualization.adapters.timeseries import TimeSeriesAdapter
from qubic.visualization.adapters.tire import TireSimulationAdapter
from qubic.visualization.core.data_model import VisualizationData


class TestTireSimulationAdapter:
    """Tests for TireSimulationAdapter."""

    def test_create_synthetic_tire(self):
        """Test synthetic tire mesh generation."""

        adapter = TireSimulationAdapter()
        data = adapter.create_synthetic_tire(resolution=32, include_fields=True)

        assert isinstance(data, VisualizationData)
        assert len(data.vertices) > 0
        assert len(data.faces) > 0
        assert "temperature" in data.scalar_fields
        assert "stress_von_mises" in data.scalar_fields
        assert "wear_depth" in data.scalar_fields

    def test_load_from_dict(self):
        """Test loading tire data from dictionary."""

        adapter = TireSimulationAdapter()

        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        faces = np.array([[0, 1, 2]])
        temperature = np.array([300.0, 310.0, 305.0])

        data_dict = {
            "vertices": vertices,
            "faces": faces,
            "temperature": temperature,
        }

        data = adapter.load_data(data_dict)

        assert isinstance(data, VisualizationData)
        assert np.array_equal(data.vertices, vertices)
        assert np.array_equal(data.faces, faces)
        assert "temperature" in data.scalar_fields

    def test_validate_source(self):
        """Test source validation."""

        adapter = TireSimulationAdapter()

        # Valid dictionary source
        valid_dict = {"vertices": [[0, 0, 0]], "faces": [[0, 0, 0]]}
        assert adapter.validate_source(valid_dict) is True

        # Invalid source
        assert adapter.validate_source(123) is False


class TestQuantumSimulationAdapter:
    """Tests for QuantumSimulationAdapter."""

    def test_create_synthetic_state(self):
        """Test synthetic quantum state generation."""

        adapter = QuantumSimulationAdapter()

        for state_type in ["superposition", "entangled", "basis"]:
            data = adapter.create_synthetic_state(n_qubits=3, state_type=state_type)

            assert isinstance(data, VisualizationData)
            assert len(data.vertices) > 0
            assert "probability" in data.scalar_fields
            assert "phase" in data.scalar_fields

    def test_load_from_array(self):
        """Test loading from amplitude array."""

        adapter = QuantumSimulationAdapter()

        # Create simple 2-qubit state
        amplitudes = np.array([0.5, 0.5, 0.5, 0.5], dtype=complex)
        data = adapter.load_data(amplitudes)

        assert isinstance(data, VisualizationData)
        assert "probability" in data.scalar_fields

    def test_validate_source(self):
        """Test source validation."""

        adapter = QuantumSimulationAdapter()

        # Valid array
        valid_array = np.array([1.0 + 0j, 0.0 + 0j])
        assert adapter.validate_source(valid_array) is True

        # Invalid source
        assert adapter.validate_source([1, 2, 3]) is False


class TestMeshAdapter:
    """Tests for MeshAdapter."""

    def test_create_test_mesh(self):
        """Test test mesh creation."""

        adapter = MeshAdapter()

        for mesh_type in ["sphere", "cube", "cylinder"]:
            data = adapter.create_test_mesh(mesh_type=mesh_type, resolution=10)

            assert isinstance(data, VisualizationData)
            assert len(data.vertices) > 0
            assert len(data.faces) > 0

    def test_load_from_dict(self):
        """Test loading from dictionary."""

        adapter = MeshAdapter()

        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        faces = np.array([[0, 1, 2], [0, 1, 3]])

        data_dict = {"vertices": vertices, "faces": faces}
        data = adapter.load_data(data_dict)

        assert isinstance(data, VisualizationData)
        assert np.array_equal(data.vertices, vertices)
        assert np.array_equal(data.faces, faces)

    def test_validate_source(self):
        """Test source validation."""

        adapter = MeshAdapter()

        valid_dict = {"vertices": [[0, 0, 0]], "faces": [[0, 0, 0]]}
        assert adapter.validate_source(valid_dict) is True

        invalid_dict = {"vertices": [[0, 0, 0]]}  # Missing faces
        assert adapter.validate_source(invalid_dict) is False


class TestTimeSeriesAdapter:
    """Tests for TimeSeriesAdapter."""

    def test_create_synthetic_timeseries(self):
        """Test synthetic time-series generation."""

        adapter = TimeSeriesAdapter()
        data = adapter.create_synthetic_timeseries(n_steps=5)

        assert isinstance(data, VisualizationData)
        assert adapter.get_num_timesteps() == 5

    def test_get_timestep(self):
        """Test retrieving specific timesteps."""

        adapter = TimeSeriesAdapter()
        adapter.create_synthetic_timeseries(n_steps=3)

        for i in range(3):
            data = adapter.get_timestep(i)
            assert isinstance(data, VisualizationData)

        with pytest.raises(IndexError):
            adapter.get_timestep(10)

    def test_get_time_range(self):
        """Test time range retrieval."""

        adapter = TimeSeriesAdapter()
        adapter.create_synthetic_timeseries(n_steps=10)

        start_time, end_time = adapter.get_time_range()
        assert start_time <= end_time

    def test_validate_source(self):
        """Test source validation."""

        adapter = TimeSeriesAdapter()

        # Valid list of dicts
        valid_list = [{"vertices": [[0, 0, 0]], "faces": [[0, 0, 0]]}]
        assert adapter.validate_source(valid_list) is True

        # Invalid source
        assert adapter.validate_source("not a valid source") is False
