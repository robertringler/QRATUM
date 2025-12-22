"""Integration tests for QRATUM platform integration layer.

Tests platform initialization, deterministic execution, backend selection,
and workflow integration.
"""

import pytest
from qratum.core.exceptions import (
    PlatformConfigError,
)

from qratum import PlatformConfig, QRATUMPlatform, create_platform


class TestPlatformIntegration:
    """Test suite for QRATUM platform integration."""

    def test_platform_initialization(self):
        """Test platform initializes with valid config."""
        config = PlatformConfig(
            quantum_backend="simulator",
            seed=42,
            do178c_enabled=True,
            audit_enabled=True,
        )

        platform = QRATUMPlatform(config)

        assert platform is not None
        assert platform.config.quantum_backend == "simulator"
        assert platform.config.seed == 42
        assert platform.config.do178c_enabled is True

    def test_platform_initialization_invalid_config(self):
        """Test platform initialization fails with invalid config."""
        config = PlatformConfig(
            quantum_backend="invalid_backend",
            seed=42,
        )

        with pytest.raises(PlatformConfigError):
            config.validate()

    def test_deterministic_execution(self):
        """Test same seed produces identical execution_id."""
        platform1 = create_platform(quantum_backend="simulator", seed=42)
        platform2 = create_platform(quantum_backend="simulator", seed=42)

        # Both platforms should have the same seed
        assert platform1.config.seed == platform2.config.seed
        assert platform1.config.seed == 42

    def test_backend_selection(self):
        """Test intelligent backend selection logic."""
        platform = create_platform(quantum_backend="simulator", seed=42)

        # 2-10 qubits: quantum
        backend = platform.select_backend("VQE", 8)
        assert backend == "quantum"

        # 10-20 qubits: hybrid
        backend = platform.select_backend("VQE", 15)
        assert backend == "hybrid"

        # >20 qubits: classical
        backend = platform.select_backend("VQE", 25)
        assert backend == "classical"

    def test_execution_context(self):
        """Test execution context provides audit logging."""
        platform = create_platform(quantum_backend="simulator", seed=42)

        with platform.execution_context("TestWorkflow") as ctx:
            assert ctx.workflow_name == "TestWorkflow"
            assert ctx.seed == 42
            assert ctx.execution_id is not None
            assert ctx.audit_enabled is True

    def test_vqe_workflow_integration(self):
        """Test VQE workflow with compliance hooks."""
        platform = create_platform(
            quantum_backend="simulator",
            seed=42,
            do178c_enabled=True,
            audit_enabled=True,
        )

        result = platform.run_vqe(molecule="H2", bond_length=0.735, basis="sto3g")

        assert result is not None
        assert "molecule" in result
        assert result["molecule"] == "H2"
        assert "energy" in result
        assert "execution_id" in result
        assert "compliance" in result
        assert result["compliance"]["do178c_enabled"] is True

    def test_qaoa_workflow_integration(self):
        """Test QAOA workflow with compliance hooks."""
        platform = create_platform(
            quantum_backend="simulator",
            seed=42,
            do178c_enabled=True,
            audit_enabled=True,
        )

        problem_data = {"edges": [[0, 1], [1, 2], [2, 3]]}
        result = platform.run_qaoa(problem_type="maxcut", problem_data=problem_data, p_layers=3)

        assert result is not None
        assert "problem_type" in result
        assert result["problem_type"] == "maxcut"
        assert "solution" in result
        assert "execution_id" in result
        assert "compliance" in result

    def test_hybrid_optimization(self):
        """Test hybrid quantum-classical optimization."""
        platform = create_platform(quantum_backend="simulator", seed=42)

        def objective(x):
            return x**2

        result = platform.run_hybrid_optimization(
            objective_function=objective, constraints={}, initial_guess=1.0
        )

        assert result is not None
        assert "execution_id" in result
        assert "method" in result

    def test_compliance_report_generation(self):
        """Test compliance report generation."""
        platform = create_platform(
            quantum_backend="simulator",
            seed=42,
            do178c_enabled=True,
            audit_enabled=True,
        )

        # Run a workflow first
        platform.run_vqe(molecule="H2", bond_length=0.735)

        # Generate report
        report = platform.generate_compliance_report("/tmp/test_compliance_report.json")

        assert report is not None
        assert "report_id" in report
        assert "generated_at" in report
        assert "configuration" in report
        assert "audit_trail" in report
        assert "seed_manifest" in report

    def test_factory_function(self):
        """Test create_platform factory function."""
        platform = create_platform(
            quantum_backend="simulator",
            seed=42,
            shots=2048,
            do178c_enabled=True,
        )

        assert platform is not None
        assert platform.config.quantum_backend == "simulator"
        assert platform.config.seed == 42
        assert platform.config.shots == 2048
        assert platform.config.do178c_enabled is True

    def test_seed_requirement_for_do178c(self):
        """Test seed is required when DO-178C is enabled."""
        config = PlatformConfig(
            quantum_backend="simulator",
            seed=None,  # No seed
            do178c_enabled=True,  # DO-178C enabled
        )

        with pytest.raises(PlatformConfigError, match="seed is required"):
            config.validate()

    def test_ibmq_token_requirement(self):
        """Test IBMQ token is required for ibmq backend."""
        config = PlatformConfig(
            quantum_backend="ibmq",
            seed=42,
            ibmq_token=None,  # No token
        )

        with pytest.raises(PlatformConfigError, match="ibmq_token is required"):
            config.validate()

    def test_invalid_precision(self):
        """Test invalid simulation precision raises error."""
        config = PlatformConfig(
            quantum_backend="simulator",
            seed=42,
            simulation_precision="invalid",
        )

        with pytest.raises(PlatformConfigError, match="Invalid simulation_precision"):
            config.validate()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
