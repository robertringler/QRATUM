"""Shared pytest fixtures for QuASIM test suite.

This module provides common fixtures for quantum simulation testing,
including configuration objects, mock data, and test utilities.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quasim import Config, Runtime


@pytest.fixture
def default_config() -> Config:
    """Provide a default QuASIM configuration for testing.

    Returns:
        Config: Default runtime configuration with fp32 precision
    """
    return Config(
        simulation_precision="fp32",
        max_workspace_mb=64,
        backend="cpu",
        seed=42,
    )


@pytest.fixture
def high_precision_config() -> Config:
    """Provide a high-precision QuASIM configuration for testing.

    Returns:
        Config: High-precision runtime configuration with fp64 precision
    """
    return Config(
        simulation_precision="fp64",
        max_workspace_mb=128,
        backend="cpu",
        seed=42,
    )


@pytest.fixture
def simple_circuit() -> list[list[complex]]:
    """Provide a simple quantum circuit for testing.

    Returns:
        list: Simple 2-gate circuit with identity-like operations
    """
    return [
        [1 + 0j, 1 + 0j, 1 + 0j, 1 + 0j],
        [1 + 0j, 0 + 0j, 0 + 0j, 1 + 0j],
    ]


@pytest.fixture
def complex_circuit() -> list[list[complex]]:
    """Provide a more complex quantum circuit for testing.

    Returns:
        list: Complex multi-gate circuit for integration testing
    """
    return [
        [1 + 0j, 0 + 0j, 0 + 0j, 1 + 0j],
        [0.707 + 0j, 0.707 + 0j, 0 + 0j, 0 + 0j],
        [1 + 0j, 0 + 0j, 0 + 0j, -1 + 0j],
        [0 + 0j, 1 + 0j, 1 + 0j, 0 + 0j],
    ]


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test outputs.

    Args:
        tmp_path: pytest's built-in temporary directory fixture

    Returns:
        Path: Temporary directory path for test outputs
    """
    output_dir = tmp_path / "test_output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture
def mock_telemetry_data() -> dict[str, Any]:
    """Provide mock telemetry data for testing.

    Returns:
        dict: Mock telemetry data structure
    """
    return {
        "timestamp": "2025-12-12T14:00:00Z",
        "simulation_id": "test-sim-001",
        "metrics": {
            "fidelity": 0.98,
            "latency_ms": 1.5,
            "memory_mb": 64,
        },
        "status": "completed",
    }


@pytest.fixture
def runtime_context(default_config: Config):
    """Provide a runtime context manager for testing.

    Args:
        default_config: Default configuration fixture

    Yields:
        Runtime: Initialized runtime instance
    """
    rt = Runtime(default_config)
    rt.__enter__()
    try:
        yield rt
    finally:
        rt.__exit__(None, None, None)
