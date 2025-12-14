"""Integration tests for QuNimbus validation."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from quasim.io.hdf5 import write_snapshot
from quasim.validation.compare import compare_observables


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_snapshot(temp_dir):
    """Create a sample HDF5 snapshot for testing."""
    snapshot_path = temp_dir / "test_snapshot.hdf5"

    meta = {
        "version": "1.0",
        "query_id": "qid-test-123",
        "seed": 42,
    }

    arrays = {
        "agents": np.zeros((100, 5)),  # 100 agents
        "climate": np.ones((50,)) * 288.5,  # Mean temp 288.5
        "economy": np.array([100.0, 110.0, 112.0]),  # ~12% return
    }

    write_snapshot(str(snapshot_path), meta, arrays)
    return snapshot_path


@pytest.fixture
def sample_config(temp_dir):
    """Create a sample observables config."""
    config_path = temp_dir / "observables.yml"
    config_path.write_text(
        """version: 1
observables:
  population_test:
    source: "/agents"
    reduce: "count"
    expected: 100
    tolerance_abs: 10
  mean_temp_test:
    source: "/climate"
    reduce: "mean"
    expected: 288.5
    tolerance_abs: 1.0
  return_test:
    source: "/economy"
    reduce: "return_ytd"
    expected: 0.12
    tolerance_abs: 0.05
"""
    )
    return config_path


def test_compare_observables_pass(sample_snapshot, sample_config):
    """Test observable comparison with passing values."""
    results = compare_observables(str(sample_snapshot), str(sample_config), tol_default=0.03)

    assert "population_test" in results
    assert "mean_temp_test" in results
    assert "return_test" in results

    # Check population
    assert results["population_test"]["value"] == 100
    assert results["population_test"]["expected"] == 100
    assert results["population_test"]["pass"] is True

    # Check temperature
    assert abs(results["mean_temp_test"]["value"] - 288.5) < 0.01
    assert results["mean_temp_test"]["pass"] is True


def test_compare_observables_fail(sample_snapshot, temp_dir):
    """Test observable comparison with failing values."""
    config_path = temp_dir / "fail_config.yml"
    config_path.write_text(
        """version: 1
observables:
  population_test:
    source: "/agents"
    reduce: "count"
    expected: 1000  # Wrong expectation
    tolerance_abs: 10
"""
    )

    results = compare_observables(str(sample_snapshot), str(config_path), tol_default=0.03)

    assert results["population_test"]["pass"] is False
    assert results["population_test"]["delta"] > 10


def test_compare_observables_missing_snapshot(sample_config):
    """Test handling of missing snapshot file."""
    results = compare_observables("nonexistent.hdf5", str(sample_config))

    # Should return results with errors
    for _name, result in results.items():
        assert result["pass"] is False


def test_compare_observables_missing_config(sample_snapshot):
    """Test handling of missing config file."""
    results = compare_observables(str(sample_snapshot), "nonexistent.yml")

    # Should return empty results
    assert len(results) == 0


def test_compare_observables_relative_tolerance(sample_snapshot, temp_dir):
    """Test relative tolerance checking."""
    config_path = temp_dir / "rel_tol_config.yml"
    config_path.write_text(
        """version: 1
observables:
  test_observable:
    source: "/climate"
    reduce: "mean"
    expected: 288.5
    tolerance_rel: 0.01  # 1% relative tolerance
"""
    )

    results = compare_observables(str(sample_snapshot), str(config_path), tol_default=0.03)

    assert results["test_observable"]["pass"] is True
