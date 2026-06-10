"""QRATUM Platform Configuration.

Extended configuration for platform integration layer.

Classification: UNCLASSIFIED // CUI
"""

from __future__ import annotations

from dataclasses import dataclass

from .exceptions import PlatformConfigError

__all__ = ["PlatformConfig"]


@dataclass
class PlatformConfig:
    """Platform configuration for QRATUM integration layer.

    Attributes:
        quantum_backend: Quantum backend selection ('simulator' | 'ibmq' | 'cuquantum')
        ibmq_token: IBM Quantum token (required for ibmq backend)
        device_name: Device name for quantum backend
        shots: Number of shots for quantum circuit execution
        seed: Random seed for reproducibility (REQUIRED for DO-178C)
        max_qubits: Maximum number of qubits
        do178c_enabled: Enable DO-178C compliance mode
        audit_enabled: Enable audit logging
        nist_controls_enabled: Enable NIST controls
        prometheus_enabled: Enable Prometheus metrics
        grafana_enabled: Enable Grafana dashboards
        log_level: Logging level
        kubernetes_namespace: Kubernetes namespace for deployment
        gpu_enabled: Enable GPU acceleration
        auto_scaling: Enable auto-scaling
        simulation_precision: Simulation precision ('fp8' | 'fp16' | 'fp32' | 'fp64')
        max_workspace_mb: Maximum workspace size in MB
    """

    # Quantum backend selection
    quantum_backend: str = "simulator"
    ibmq_token: str | None = None
    device_name: str | None = None

    # Execution parameters
    shots: int = 1024
    seed: int | None = None
    max_qubits: int = 20

    # Compliance settings
    do178c_enabled: bool = True
    audit_enabled: bool = True
    nist_controls_enabled: bool = True

    # Observability
    prometheus_enabled: bool = True
    grafana_enabled: bool = True
    log_level: str = "INFO"

    # Infrastructure
    kubernetes_namespace: str = "quasim-prod"
    gpu_enabled: bool = False
    auto_scaling: bool = True

    # Precision and performance
    simulation_precision: str = "fp32"
    max_workspace_mb: int = 512

    def validate(self) -> None:
        """Validate configuration (DO-178C requirement).

        Raises:
            PlatformConfigError: If configuration is invalid
        """
        # Validate quantum backend
        valid_backends = ["simulator", "ibmq", "cuquantum"]
        if self.quantum_backend not in valid_backends:
            raise PlatformConfigError(
                f"Invalid quantum_backend: {self.quantum_backend}. "
                f"Must be one of {valid_backends}"
            )

        # Validate IBMQ token requirement
        if self.quantum_backend == "ibmq" and not self.ibmq_token:
            raise PlatformConfigError("ibmq_token is required when quantum_backend='ibmq'")

        # Validate seed requirement for DO-178C
        if self.do178c_enabled and self.seed is None:
            raise PlatformConfigError("seed is required when do178c_enabled=True")

        # Validate shots
        if self.shots <= 0:
            raise PlatformConfigError(f"shots must be positive, got {self.shots}")

        # Validate max_qubits
        if self.max_qubits <= 0 or self.max_qubits > 100:
            raise PlatformConfigError(
                f"max_qubits must be between 1 and 100, got {self.max_qubits}"
            )

        # Validate simulation precision
        valid_precisions = ["fp8", "fp16", "fp32", "fp64"]
        if self.simulation_precision not in valid_precisions:
            raise PlatformConfigError(
                f"Invalid simulation_precision: {self.simulation_precision}. "
                f"Must be one of {valid_precisions}"
            )

        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_log_levels:
            raise PlatformConfigError(
                f"Invalid log_level: {self.log_level}. " f"Must be one of {valid_log_levels}"
            )

        # Validate workspace size
        if self.max_workspace_mb <= 0:
            raise PlatformConfigError(
                f"max_workspace_mb must be positive, got {self.max_workspace_mb}"
            )
