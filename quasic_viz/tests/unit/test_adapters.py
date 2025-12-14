"""Unit tests for tire data adapter."""

import numpy as np


class MockEnvironment:
    """Mock environment for testing."""

    temperature_celsius = 25.0


class MockPerformanceMetrics:
    """Mock performance metrics for testing."""

    wear_rate = 1.0


class MockSimulationResult:
    """Mock simulation result for testing."""

    tire_geometry = None
    performance_metrics = MockPerformanceMetrics()
    environment = MockEnvironment()
    load_kg = 500
    pressure_kpa = 250
    speed_kmh = 100
    simulation_id = "test-001"
    thermal_map = None
    stress_distribution = None
    wear_pattern = None


class TestTireDataAdapter:
    """Tests for TireDataAdapter class."""

    def test_extract_visualization_data(self):
        """Test data extraction from simulation result."""
        from quasic_viz.adapters.tire_data_adapter import TireDataAdapter

        result = MockSimulationResult()
        data = TireDataAdapter.extract_visualization_data(result)

        assert "thermal_map" in data
        assert "stress_distribution" in data
        assert "wear_pattern" in data
        assert data["simulation_id"] == "test-001"

    def test_extract_thermal_map_synthetic(self):
        """Test synthetic thermal map generation."""
        from quasic_viz.adapters.tire_data_adapter import TireDataAdapter

        result = MockSimulationResult()
        thermal = TireDataAdapter._extract_thermal_map(result)

        assert isinstance(thermal, np.ndarray)
        assert len(thermal) == 1000

    def test_extract_stress_distribution_synthetic(self):
        """Test synthetic stress distribution generation."""
        from quasic_viz.adapters.tire_data_adapter import TireDataAdapter

        result = MockSimulationResult()
        stress = TireDataAdapter._extract_stress_distribution(result)

        assert isinstance(stress, np.ndarray)
        assert len(stress) == 1000

    def test_normalize_field_data_upsample(self):
        """Test field data upsampling."""
        from quasic_viz.adapters.tire_data_adapter import TireDataAdapter

        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = TireDataAdapter.normalize_field_data(data, 10)

        assert len(normalized) == 10

    def test_normalize_field_data_downsample(self):
        """Test field data downsampling."""
        from quasic_viz.adapters.tire_data_adapter import TireDataAdapter

        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        normalized = TireDataAdapter.normalize_field_data(data, 5)

        assert len(normalized) == 5
