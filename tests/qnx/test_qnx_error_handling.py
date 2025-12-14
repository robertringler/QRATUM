"""Tests for QNX error handling and canonical serialization."""

from unittest.mock import MagicMock, patch

from qnx import QNXSubstrate, SimulationConfig
from qnx.core import _canonical_serialize, _default_serializer


def test_canonical_serialize_with_dict():
    """Test canonical serialization produces deterministic output for dicts."""
    obj1 = {"b": 2, "a": 1, "c": 3}
    obj2 = {"a": 1, "c": 3, "b": 2}

    result1 = _canonical_serialize(obj1)
    result2 = _canonical_serialize(obj2)

    # Should be identical due to sort_keys=True
    assert result1 == result2
    assert result1 == '{"a":1,"b":2,"c":3}'


def test_canonical_serialize_with_nested_structures():
    """Test canonical serialization with nested structures."""
    obj = {
        "backend": "test",
        "results": {"values": [1, 2, 3], "status": "ok"},
        "seed": 42,
    }

    result = _canonical_serialize(obj)

    # Verify it's valid JSON without whitespace
    assert '"backend":"test"' in result
    assert '"seed":42' in result
    assert result.count(" ") == 0  # No spaces due to separators


def test_canonical_serialize_with_non_serializable_object():
    """Test canonical serialization falls back for non-serializable objects."""

    class CustomObject:
        def __str__(self):
            return "CustomObject"

    obj = {"data": CustomObject()}

    # Should not raise, should fall back to string representation
    result = _canonical_serialize(obj)
    assert "CustomObject" in result


def test_default_serializer():
    """Test default serializer converts objects to strings."""

    class TestClass:
        def __str__(self):
            return "test_string"

    obj = TestClass()
    result = _default_serializer(obj)

    assert result == "test_string"


def test_backend_exception_captured_in_result():
    """Test that backend exceptions are caught and returned in SubstrateResult."""
    substrate = QNXSubstrate()

    # Mock a backend that raises an exception
    with patch.object(substrate, "_resolve_backend") as mock_resolve:
        mock_backend = MagicMock()
        mock_backend.run.side_effect = RuntimeError("Backend failed")
        mock_resolve.return_value = mock_backend

        config = SimulationConfig(scenario_id="test", timesteps=1, seed=42)
        result = substrate.run_simulation(config)

        # Verify exception was caught and returned as structured result
        assert result.raw_results.get("status") == "error"
        assert "Backend failed" in result.raw_results.get("error", "")
        assert "backend_exception" in result.errors
        assert result.execution_time_ms >= 0


def test_carbon_estimation_failure_does_not_break_simulation():
    """Test that carbon estimation failures are caught and handled."""
    substrate = QNXSubstrate()

    # Mock estimate_carbon to raise an exception
    with patch("qnx.core.estimate_carbon") as mock_carbon:
        mock_carbon.side_effect = Exception("Carbon estimation failed")

        config = SimulationConfig(scenario_id="test", timesteps=1, seed=42)
        result = substrate.run_simulation(config)

        # Simulation should succeed with zero carbon emissions
        assert result.carbon_emissions_kg == 0.0
        assert result.raw_results.get("engine") == "quasim_modern"


def test_deterministic_hash_with_canonical_serialization():
    """Test that canonical serialization produces deterministic hashes."""
    substrate = QNXSubstrate()

    config1 = SimulationConfig(scenario_id="test", timesteps=2, seed=100)
    config2 = SimulationConfig(scenario_id="test", timesteps=2, seed=100)

    result1 = substrate.run_simulation(config1)
    result2 = substrate.run_simulation(config2)

    # Same config should produce same hash
    assert result1.simulation_hash == result2.simulation_hash


def test_backend_reported_error_added_to_errors():
    """Test that backend-reported errors are captured in the errors list."""
    substrate = QNXSubstrate()

    # Mock a backend that returns an error status
    with patch.object(substrate, "_resolve_backend") as mock_resolve:
        mock_backend = MagicMock()
        mock_backend.run.return_value = {"status": "error", "message": "Internal error"}
        mock_resolve.return_value = mock_backend

        config = SimulationConfig(scenario_id="test", timesteps=1, seed=42)
        result = substrate.run_simulation(config)

        # Verify error was captured
        assert "backend_reported_error" in result.errors
        assert result.raw_results.get("status") == "error"


def test_backend_warnings_extracted():
    """Test that backend warnings are extracted and added to the warnings list."""
    substrate = QNXSubstrate()

    # Mock a backend that returns warnings
    with patch.object(substrate, "_resolve_backend") as mock_resolve:
        mock_backend = MagicMock()
        mock_backend.run.return_value = {
            "status": "ok",
            "warnings": ["Warning 1", "Warning 2"],
        }
        mock_resolve.return_value = mock_backend

        config = SimulationConfig(scenario_id="test", timesteps=1, seed=42)
        result = substrate.run_simulation(config)

        # Verify warnings were extracted
        assert "Warning 1" in result.warnings
        assert "Warning 2" in result.warnings
