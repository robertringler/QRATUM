"""Tests for data adapters."""

import sys
from pathlib import Path

import numpy as np

# Add qubic-viz to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.tire_data_adapter import TireDataAdapter


class MockEnvironment:
    """Mock environmental conditions."""

    def __init__(self):
        self.temperature_celsius = 25.0


class MockPerformanceMetrics:
    """Mock performance metrics."""

    def __init__(self):
        self.wear_rate = 0.5


class MockTireGeometry:
    """Mock tire geometry."""

    def __init__(self):
        self.outer_diameter_mm = 700.0


class MockSimulationResult:
    """Mock simulation result for testing."""

    def __init__(self):
        self.simulation_id = "test-001"
        self.tire_geometry = MockTireGeometry()
        self.performance_metrics = MockPerformanceMetrics()
        self.environment = MockEnvironment()
        self.load_kg = 500.0
        self.pressure_kpa = 220.0
        self.speed_kmh = 100.0
        self.thermal_map = None
        self.stress_distribution = None
        self.wear_pattern = None


def test_extract_visualization_data():
    """Test visualization data extraction."""
    sim_result = MockSimulationResult()
    viz_data = TireDataAdapter.extract_visualization_data(sim_result)

    assert "geometry" in viz_data
    assert "metrics" in viz_data
    assert "thermal_map" in viz_data
    assert "stress_distribution" in viz_data
    assert "wear_pattern" in viz_data
    assert "environment" in viz_data
    assert "load_kg" in viz_data
    assert "pressure_kpa" in viz_data
    assert "speed_kmh" in viz_data
    assert "simulation_id" in viz_data

    assert viz_data["simulation_id"] == "test-001"
    assert viz_data["load_kg"] == 500.0


def test_extract_thermal_map():
    """Test thermal map extraction."""
    sim_result = MockSimulationResult()
    thermal_map = TireDataAdapter._extract_thermal_map(sim_result)

    assert isinstance(thermal_map, np.ndarray)
    assert len(thermal_map) > 0


def test_extract_stress_distribution():
    """Test stress distribution extraction."""
    sim_result = MockSimulationResult()
    stress_dist = TireDataAdapter._extract_stress_distribution(sim_result)

    assert isinstance(stress_dist, np.ndarray)
    assert len(stress_dist) > 0


def test_extract_wear_pattern():
    """Test wear pattern extraction."""
    sim_result = MockSimulationResult()
    wear_pattern = TireDataAdapter._extract_wear_pattern(sim_result)

    assert isinstance(wear_pattern, np.ndarray)
    assert len(wear_pattern) > 0


def test_normalize_field_data_same_size():
    """Test field data normalization with same size."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    normalized = TireDataAdapter.normalize_field_data(data, 5)

    assert len(normalized) == 5
    np.testing.assert_array_equal(normalized, data)


def test_normalize_field_data_upsample():
    """Test field data upsampling."""
    data = np.array([1.0, 5.0])
    normalized = TireDataAdapter.normalize_field_data(data, 5)

    assert len(normalized) == 5
    # Check interpolation
    assert normalized[0] == 1.0
    assert normalized[-1] == 5.0


def test_normalize_field_data_downsample():
    """Test field data downsampling."""
    data = np.arange(100)
    normalized = TireDataAdapter.normalize_field_data(data, 10)

    assert len(normalized) == 10
