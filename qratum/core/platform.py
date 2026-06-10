"""QRATUM Platform Integration.

Main platform class integrating all layers.

Classification: UNCLASSIFIED // CUI
"""

from __future__ import annotations

import hashlib
import json
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Generator

from ..compliance import AuditWrapper, SeedManagerWrapper
from ..observability import MetricsCollector, StructuredLogger
from ..opt import OptimizationAdapter
from ..quantum import QuantumBackendAdapter
from ..workflows import QAOAWorkflow, VQEWorkflow
from .context import ExecutionContext
from .exceptions import BackendSelectionError, WorkflowExecutionError
from .platform_config import PlatformConfig

__all__ = ["QRATUMPlatform", "create_platform"]


class QRATUMPlatform:
    """Main QRATUM platform class integrating all layers.

    Layers:
    1. Compliance & Seed Management (SeedManagerWrapper, AuditWrapper, DO178CCompliance)
    2. Observability (StructuredLogger, MetricsCollector)
    3. Backend Abstraction (QuantumBackendAdapter, ClassicalFallback)
    4. Algorithms (VQE, QAOA, HybridOptimizer)
    5. Infrastructure (GPUScheduler)
    """

    def __init__(self, config: PlatformConfig):
        """Initialize QRATUM platform.

        Args:
            config: Platform configuration

        Raises:
            PlatformConfigError: If configuration is invalid
        """
        # Validate configuration
        config.validate()
        self.config = config

        # Initialize compliance layer
        self._seed_manager = SeedManagerWrapper(
            base_seed=config.seed or 42, environment="production"
        )
        self._audit = AuditWrapper(enabled=config.audit_enabled)

        # Initialize observability layer
        self._logger = StructuredLogger(name="qratum.platform", level=config.log_level)
        self._metrics = MetricsCollector(enabled=config.prometheus_enabled)

        # Initialize backend layer
        self._quantum_backend = QuantumBackendAdapter(
            backend_type=config.quantum_backend,
            seed=config.seed,
            shots=config.shots,
        )
        self._opt_backend = OptimizationAdapter(method="hybrid", seed=config.seed)

        # Initialize workflow layer
        self._vqe_workflow = VQEWorkflow(
            seed=config.seed, shots=config.shots, backend=config.quantum_backend
        )
        self._qaoa_workflow = QAOAWorkflow(
            seed=config.seed, shots=config.shots, backend=config.quantum_backend
        )

        self._execution_logger = logging.getLogger("qratum.platform.execution")

    def _generate_execution_hash(self) -> str:
        """Generate SHA-256 execution hash for DO-178C traceability.

        Returns:
            SHA-256 hash of execution metadata
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        seed = self.config.seed or 42
        data = f"qratum|{seed}|{timestamp}|{self.config.quantum_backend}"
        return hashlib.sha256(data.encode()).hexdigest()

    def select_backend(self, problem_type: str, problem_size: int) -> str:
        """Intelligent backend selection.

        Selection logic:
        - 2-10 qubits: Quantum simulator/hardware
        - 10-20 qubits: Hybrid
        - >20 qubits: Classical fallback

        Args:
            problem_type: Problem type
            problem_size: Problem size (number of qubits/variables)

        Returns:
            Selected backend name

        Raises:
            BackendSelectionError: If backend selection fails
        """
        try:
            if problem_size <= 10:
                backend = "quantum"
            elif problem_size <= 20:
                backend = "hybrid"
            else:
                backend = "classical"

            self._metrics.record_backend_selection(backend)
            return backend

        except Exception as e:
            raise BackendSelectionError(f"Backend selection failed: {e}") from e

    @contextmanager
    def execution_context(self, workflow_name: str) -> Generator[ExecutionContext, None, None]:
        """Context manager for workflow execution.

        Provides:
        - Deterministic seed management
        - Audit logging with SHA-256 traceability
        - Prometheus metrics collection
        - DO-178C compliance validation

        Args:
            workflow_name: Name of the workflow

        Yields:
            ExecutionContext instance
        """
        ctx = ExecutionContext(
            workflow_name=workflow_name,
            seed=self.config.seed,
            audit_enabled=self.config.audit_enabled,
            metrics_enabled=self.config.prometheus_enabled,
            do178c_enabled=self.config.do178c_enabled,
        )

        with ctx:
            yield ctx

    def run_vqe(self, molecule: str, bond_length: float, basis: str = "sto3g") -> dict[str, Any]:
        """Execute VQE workflow with hybrid preprocessing/postprocessing.

        Args:
            molecule: Molecule name (e.g., 'H2', 'LiH')
            bond_length: Bond length in Angstroms
            basis: Basis set

        Returns:
            Result dictionary with energy and execution metadata

        Raises:
            WorkflowExecutionError: If VQE execution fails
        """
        try:
            with self.execution_context("VQE") as ctx:
                # Log start
                execution_id = ctx.execution_id or "unknown"
                self._audit.log_start("VQE", execution_id, {"molecule": molecule})

                # Execute VQE
                result = self._vqe_workflow.run(molecule, bond_length, basis)

                # Add execution metadata
                result["execution_id"] = execution_id
                result["compliance"] = {
                    "do178c_enabled": self.config.do178c_enabled,
                    "audit_enabled": self.config.audit_enabled,
                }

                # Log result
                self._audit.log_result("VQE", execution_id, result)
                ctx.set_result(result)

                return result

        except Exception as e:
            raise WorkflowExecutionError(f"VQE execution failed: {e}") from e

    def run_qaoa(
        self, problem_type: str, problem_data: dict[str, Any], p_layers: int = 3
    ) -> dict[str, Any]:
        """Execute QAOA workflow for combinatorial optimization.

        Args:
            problem_type: Problem type (e.g., 'maxcut', 'ising')
            problem_data: Problem data dictionary
            p_layers: Number of QAOA layers

        Returns:
            Result dictionary with solution and execution metadata

        Raises:
            WorkflowExecutionError: If QAOA execution fails
        """
        try:
            with self.execution_context("QAOA") as ctx:
                # Log start
                execution_id = ctx.execution_id or "unknown"
                self._audit.log_start("QAOA", execution_id, {"problem_type": problem_type})

                # Execute QAOA
                result = self._qaoa_workflow.run(problem_type, problem_data, p_layers)

                # Add execution metadata
                result["execution_id"] = execution_id
                result["compliance"] = {
                    "do178c_enabled": self.config.do178c_enabled,
                    "audit_enabled": self.config.audit_enabled,
                }

                # Log result
                self._audit.log_result("QAOA", execution_id, result)
                ctx.set_result(result)

                return result

        except Exception as e:
            raise WorkflowExecutionError(f"QAOA execution failed: {e}") from e

    def run_hybrid_optimization(
        self,
        objective_function: Callable,
        constraints: dict[str, Any],
        initial_guess: Any,
        method: str = "hybrid",
    ) -> dict[str, Any]:
        """Execute hybrid quantum-classical optimization.

        Args:
            objective_function: Objective function to minimize
            constraints: Optimization constraints
            initial_guess: Initial parameter guess
            method: Optimization method

        Returns:
            Optimization result dictionary

        Raises:
            WorkflowExecutionError: If optimization fails
        """
        try:
            with self.execution_context("HybridOptimization") as ctx:
                # Log start
                execution_id = ctx.execution_id or "unknown"
                self._audit.log_start("HybridOptimization", execution_id, {"method": method})

                # Execute optimization
                result = self._opt_backend.optimize(objective_function, initial_guess, constraints)

                # Add execution metadata
                result["execution_id"] = execution_id
                result["method"] = method

                # Log result
                self._audit.log_result("HybridOptimization", execution_id, result)
                ctx.set_result(result)

                return result

        except Exception as e:
            raise WorkflowExecutionError(f"Hybrid optimization failed: {e}") from e

    def generate_compliance_report(
        self, output_path: str = "compliance_report.json"
    ) -> dict[str, Any]:
        """Generate DO-178C compliance report with all artifacts.

        Args:
            output_path: Path to save compliance report

        Returns:
            Compliance report dictionary
        """
        report = {
            "report_id": self._generate_execution_hash(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "platform_version": "2.0.0",
            "configuration": {
                "quantum_backend": self.config.quantum_backend,
                "do178c_enabled": self.config.do178c_enabled,
                "audit_enabled": self.config.audit_enabled,
                "seed": self.config.seed,
            },
            "audit_trail": self._audit.get_audit_trail(),
            "seed_manifest": self._seed_manager.export_manifest(),
            "metrics": self._metrics.get_metrics(),
        }

        # Write report to file
        try:
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)
            self._execution_logger.info(f"Compliance report written to {output_path}")
        except Exception as e:
            self._execution_logger.error(f"Failed to write compliance report: {e}")

        return report


def create_platform(quantum_backend: str = "simulator", seed: int = 42, **kwargs) -> QRATUMPlatform:
    """Factory function for creating QRATUM platform instance.

    Args:
        quantum_backend: Quantum backend type
        seed: Random seed for reproducibility
        **kwargs: Additional configuration parameters

    Returns:
        QRATUMPlatform instance

    Example:
        >>> platform = create_platform(quantum_backend="simulator", seed=42)
        >>> result = platform.run_vqe("H2", bond_length=0.735)
    """
    config = PlatformConfig(quantum_backend=quantum_backend, seed=seed, **kwargs)
    return QRATUMPlatform(config)
