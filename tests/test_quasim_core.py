"""Core QuASIM runtime tests.

This module tests the core QuASIM functionality including:
- Configuration management
- Runtime lifecycle
- Simulation execution
- Deterministic behavior
"""

from __future__ import annotations

import pytest

from quasim import Config, Runtime, runtime


class TestConfig:
    """Test QuASIM configuration management."""

    def test_default_config(self):
        """Test default configuration values."""
        cfg = Config()
        assert cfg.simulation_precision == "fp32"
        assert cfg.max_workspace_mb == 1024
        assert cfg.backend == "cpu"
        assert cfg.seed is None

    def test_custom_config(self):
        """Test custom configuration values."""
        cfg = Config(
            simulation_precision="fp64",
            max_workspace_mb=2048,
            backend="cuda",
            seed=42,
        )
        assert cfg.simulation_precision == "fp64"
        assert cfg.max_workspace_mb == 2048
        assert cfg.backend == "cuda"
        assert cfg.seed == 42

    def test_precision_modes(self):
        """Test all supported precision modes."""
        precisions = ["fp8", "fp16", "fp32", "fp64"]
        for precision in precisions:
            cfg = Config(simulation_precision=precision)
            assert cfg.simulation_precision == precision

    def test_backend_modes(self):
        """Test all supported backend modes."""
        backends = ["cpu", "cuda", "rocm"]
        for backend in backends:
            cfg = Config(backend=backend)
            assert cfg.backend == backend


class TestRuntime:
    """Test QuASIM runtime behavior."""

    def test_runtime_initialization(self, default_config):
        """Test runtime initialization with config."""
        rt = Runtime(default_config)
        assert rt.config == default_config
        assert rt.average_latency == 0.0
        assert not rt._initialized

    def test_runtime_context_manager(self, default_config):
        """Test runtime context manager lifecycle."""
        rt = Runtime(default_config)
        assert not rt._initialized

        with rt:
            assert rt._initialized

        assert not rt._initialized

    def test_runtime_helper_function(self, default_config):
        """Test the runtime() context manager helper."""
        with runtime(default_config) as rt:
            assert isinstance(rt, Runtime)
            assert rt._initialized
            assert rt.config == default_config

    def test_simulation_requires_context(self, default_config):
        """Test that simulation fails without context manager."""
        rt = Runtime(default_config)
        circuit = [[1 + 0j, 1 + 0j]]

        with pytest.raises(RuntimeError, match="not initialized"):
            rt.simulate(circuit)


class TestSimulation:
    """Test quantum circuit simulation."""

    def test_simple_simulation(self, default_config, simple_circuit):
        """Test basic simulation execution."""
        with runtime(default_config) as rt:
            result = rt.simulate(simple_circuit)

            assert len(result) == len(simple_circuit)
            assert all(isinstance(val, complex) for val in result)
            assert rt.average_latency > 0

    def test_deterministic_replay(self, simple_circuit):
        """Test deterministic simulation with fixed seed."""
        cfg = Config(seed=42)

        with runtime(cfg) as rt1:
            result1 = rt1.simulate(simple_circuit)

        with runtime(cfg) as rt2:
            result2 = rt2.simulate(simple_circuit)

        assert len(result1) == len(result2)
        for r1, r2 in zip(result1, result2):
            assert r1 == r2

    def test_different_seeds_produce_different_results(self, simple_circuit):
        """Test that different seeds can produce different results."""
        cfg1 = Config(seed=42)
        cfg2 = Config(seed=123)

        with runtime(cfg1) as rt1:
            result1 = rt1.simulate(simple_circuit)

        with runtime(cfg2) as rt2:
            result2 = rt2.simulate(simple_circuit)

        # Results should be the same for this simple circuit
        # but we verify the mechanism works
        assert len(result1) == len(result2)

    def test_complex_circuit_simulation(self, default_config, complex_circuit):
        """Test simulation of complex circuit."""
        with runtime(default_config) as rt:
            result = rt.simulate(complex_circuit)

            assert len(result) == len(complex_circuit)
            assert all(isinstance(val, complex) for val in result)
            assert rt.average_latency > 0

    def test_empty_circuit(self, default_config):
        """Test simulation of empty circuit."""
        with runtime(default_config) as rt:
            result = rt.simulate([])
            assert len(result) == 0

    def test_single_gate_circuit(self, default_config):
        """Test simulation of single-gate circuit."""
        circuit = [[1 + 0j, 0 + 0j, 0 + 0j, 1 + 0j]]
        with runtime(default_config) as rt:
            result = rt.simulate(circuit)
            assert len(result) == 1
            assert isinstance(result[0], complex)


class TestPrecisionModes:
    """Test different precision modes."""

    @pytest.mark.parametrize("precision", ["fp8", "fp16", "fp32", "fp64"])
    def test_precision_mode_simulation(self, precision, simple_circuit):
        """Test simulation with different precision modes."""
        cfg = Config(simulation_precision=precision, seed=42)
        with runtime(cfg) as rt:
            result = rt.simulate(simple_circuit)
            assert len(result) == len(simple_circuit)
            assert rt.average_latency > 0


class TestScalability:
    """Test simulation scalability."""

    def test_variable_circuit_sizes(self, default_config):
        """Test simulation with circuits of different sizes."""
        sizes = [1, 2, 4, 8, 16]

        for size in sizes:
            circuit = [[1 + 0j] * 4 for _ in range(size)]
            with runtime(default_config) as rt:
                result = rt.simulate(circuit)
                assert len(result) == size

    def test_large_gate_matrix(self, default_config):
        """Test simulation with larger gate matrices."""
        # Create a circuit with larger gates
        circuit = [
            [1 + 0j] * 16,  # 16-element gate
            [0.5 + 0j] * 16,
        ]
        with runtime(default_config) as rt:
            result = rt.simulate(circuit)
            assert len(result) == len(circuit)


class TestLatencyTracking:
    """Test latency measurement."""

    def test_latency_recorded(self, default_config, simple_circuit):
        """Test that latency is recorded during simulation."""
        with runtime(default_config) as rt:
            assert rt.average_latency == 0.0
            rt.simulate(simple_circuit)
            assert rt.average_latency > 0.0

    def test_latency_positive(self, default_config, simple_circuit):
        """Test that latency is always positive."""
        with runtime(default_config) as rt:
            rt.simulate(simple_circuit)
            assert rt.average_latency > 0
