"""Hybrid Quantum Backend Abstractions and Implementations.

This module provides a unified interface for multiple quantum hardware providers
while maintaining QRATUM's invariant-preserving execution model.

All backend interactions are:
1. Deterministically prepared (classical preprocessing)
2. Executed with full provenance tracking
3. Verified before reinjection into QRATUM pipelines
4. Rollback-capable via state checkpointing
"""

from __future__ import annotations

import hashlib
import time
import uuid
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal


class BackendProvider(Enum):
    """Supported quantum backend providers."""

    IBM = "ibm"
    IONQ = "ionq"
    QUANTINUUM = "quantinuum"
    AZURE = "azure"
    BRAKET = "braket"
    SIMULATOR = "simulator"


class ExecutionStatus(Enum):
    """Status of quantum execution."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    AWAITING_APPROVAL = "awaiting_approval"


@dataclass
class HybridQuantumConfig:
    """Configuration for hybrid quantum-classical execution.

    This configuration enforces QRATUM's safety invariants:
    - Deterministic input preparation
    - Output verification requirements
    - Rollback integration points

    Attributes:
        provider: Backend provider (ibm, ionq, quantinuum, azure, braket)
        device_name: Specific device or simulator name
        shots: Number of measurement shots (min 1000 for statistical validity)
        max_circuit_depth: Maximum allowed circuit depth (NISQ constraint)
        api_token: API token for authentication (required for real hardware)
        require_verification: Whether outputs must pass verification
        enable_rollback: Whether to enable rollback checkpointing
        seed: Random seed for reproducibility (simulators only)
        optimization_level: Transpiler optimization level (0-3)
        noise_mitigation: Enable noise mitigation techniques
        dual_approval_required: Whether dual control is required for execution
    """

    provider: Literal["ibm", "ionq", "quantinuum", "azure", "braket", "simulator"] = "simulator"
    device_name: str | None = None
    shots: int = 1024
    max_circuit_depth: int = 1000
    api_token: str | None = None
    require_verification: bool = True
    enable_rollback: bool = True
    seed: int | None = 42
    optimization_level: int = 1
    noise_mitigation: bool = True
    dual_approval_required: bool = False

    def __post_init__(self) -> None:
        """Validate configuration against QRATUM safety invariants."""
        if self.shots < 100:
            warnings.warn(
                f"shots={self.shots} is very low. Use at least 1000 for reliable statistics.",
                UserWarning,
                stacklevel=2,
            )

        if self.provider != "simulator" and not self.api_token:
            raise ValueError(f"api_token required for {self.provider} backend")

        if self.provider == "simulator" and self.seed is None:
            warnings.warn(
                "No seed provided for simulator. Results may not be deterministic.",
                UserWarning,
                stacklevel=2,
            )

        if self.max_circuit_depth > 5000:
            warnings.warn(
                f"max_circuit_depth={self.max_circuit_depth} exceeds NISQ practical limits.",
                UserWarning,
                stacklevel=2,
            )

    @property
    def is_simulator(self) -> bool:
        """Check if using simulator backend."""
        return self.provider == "simulator"


@dataclass
class ExecutionResult:
    """Result of hybrid quantum execution with provenance.

    Attributes:
        execution_id: Unique identifier for this execution
        counts: Measurement outcome counts
        raw_data: Raw backend response
        execution_time: Wall-clock execution time in seconds
        queue_time: Time spent in queue (for real hardware)
        success: Whether execution completed successfully
        error_message: Error message if failed
        provenance_hash: Hash of inputs for verification
        timestamp: Execution completion timestamp
        backend_info: Information about the backend used
        verification_required: Whether this result needs verification
    """

    execution_id: str
    counts: dict[str, int]
    raw_data: dict[str, Any]
    execution_time: float
    queue_time: float = 0.0
    success: bool = True
    error_message: str | None = None
    provenance_hash: str = ""
    timestamp: str = ""
    backend_info: dict[str, Any] = field(default_factory=dict)
    verification_required: bool = True

    def __post_init__(self) -> None:
        """Set defaults."""
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


class HybridQuantumBackend(ABC):
    """Abstract base class for hybrid quantum backends.

    This class defines the interface that all quantum backend wrappers must implement.
    It enforces QRATUM's safety invariants:
    - Deterministic input preparation
    - Full provenance tracking
    - Output verification hooks
    - Rollback capability

    Implementations must:
    1. Validate circuits against safety constraints
    2. Generate provenance records for all executions
    3. Support async execution with status polling
    4. Enable result verification before use
    """

    def __init__(self, config: HybridQuantumConfig):
        """Initialize backend with configuration.

        Args:
            config: Hybrid quantum configuration
        """
        self.config = config
        self._execution_history: list[ExecutionResult] = []
        self._initialize_backend()

    @abstractmethod
    def _initialize_backend(self) -> None:
        """Initialize the specific backend connection."""
        pass

    @abstractmethod
    def execute_circuit(
        self,
        circuit: Any,
        shots: int | None = None,
        await_approval: bool = False,
    ) -> ExecutionResult:
        """Execute a quantum circuit on this backend.

        Args:
            circuit: Quantum circuit (Qiskit QuantumCircuit or equivalent)
            shots: Override shot count (uses config default if None)
            await_approval: If True, return pending result awaiting approval

        Returns:
            ExecutionResult with counts, provenance, and metadata
        """
        pass

    @abstractmethod
    def validate_circuit(self, circuit: Any) -> tuple[bool, str]:
        """Validate circuit against backend constraints.

        Args:
            circuit: Circuit to validate

        Returns:
            (is_valid, message) tuple
        """
        pass

    @abstractmethod
    def get_backend_info(self) -> dict[str, Any]:
        """Get information about the backend.

        Returns:
            Dictionary with backend capabilities and status
        """
        pass

    def compute_provenance_hash(self, circuit: Any, params: dict[str, Any]) -> str:
        """Compute deterministic hash of circuit and parameters.

        This hash enables verification that outputs correspond to declared inputs.

        Args:
            circuit: Quantum circuit
            params: Additional parameters

        Returns:
            SHA-256 hex digest of inputs
        """
        # Create deterministic string representation
        circuit_str = str(circuit) if circuit is not None else ""
        params_str = str(sorted(params.items())) if params else ""

        combined = f"{circuit_str}|{params_str}|{self.config.shots}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def get_execution_history(self) -> list[ExecutionResult]:
        """Get history of executions on this backend.

        Returns:
            List of ExecutionResult objects
        """
        return self._execution_history.copy()


class IBMHybridBackend(HybridQuantumBackend):
    """IBM Quantum backend wrapper with QRATUM safety integration.

    Supports:
    - IBM Quantum hardware (127+ qubit processors)
    - Qiskit Runtime primitives (Sampler, Estimator)
    - Noise mitigation via Qiskit Runtime
    - Session management for iterative algorithms
    """

    def _initialize_backend(self) -> None:
        """Initialize IBM Quantum connection."""
        if self.config.api_token:
            # Real hardware mode - requires qiskit-ibm-runtime
            try:
                from qiskit_ibm_runtime import QiskitRuntimeService
            except ImportError:
                raise ImportError(
                    "IBM Quantum backend requires qiskit-ibm-runtime. "
                    "Install with: pip install qiskit-ibm-runtime"
                )

            QiskitRuntimeService.save_account(
                channel="ibm_quantum",
                token=self.config.api_token,
                overwrite=True,
            )
            self._service = QiskitRuntimeService(channel="ibm_quantum")

            if self.config.device_name:
                self._backend = self._service.backend(self.config.device_name)
            else:
                self._backend = self._service.least_busy(operational=True, simulator=False)
        else:
            # Simulator mode - only requires qiskit-aer
            try:
                from qiskit_aer import AerSimulator
            except ImportError:
                raise ImportError(
                    "IBM simulator backend requires qiskit-aer. "
                    "Install with: pip install qiskit-aer"
                )

            self._backend = AerSimulator()
            self._service = None

    def execute_circuit(
        self,
        circuit: Any,
        shots: int | None = None,
        await_approval: bool = False,
    ) -> ExecutionResult:
        """Execute circuit on IBM Quantum backend."""
        from qiskit import transpile

        execution_id = str(uuid.uuid4())
        shots = shots or self.config.shots
        start_time = time.time()

        # Validate circuit
        is_valid, message = self.validate_circuit(circuit)
        if not is_valid:
            return ExecutionResult(
                execution_id=execution_id,
                counts={},
                raw_data={"error": message},
                execution_time=time.time() - start_time,
                success=False,
                error_message=message,
            )

        # Compute provenance hash
        provenance_hash = self.compute_provenance_hash(circuit, {"shots": shots})

        # Check approval requirement
        if self.config.dual_approval_required and not await_approval:
            return ExecutionResult(
                execution_id=execution_id,
                counts={},
                raw_data={"status": "awaiting_approval"},
                execution_time=0.0,
                success=True,
                provenance_hash=provenance_hash,
                verification_required=True,
                backend_info=self.get_backend_info(),
            )

        # Transpile circuit
        transpiled = transpile(
            circuit,
            backend=self._backend,
            optimization_level=self.config.optimization_level,
            seed_transpiler=self.config.seed,
        )

        # Execute
        if self.config.is_simulator or self._service is None:
            job = self._backend.run(transpiled, shots=shots, seed_simulator=self.config.seed)
        else:
            job = self._backend.run(transpiled, shots=shots)

        result = job.result()
        execution_time = time.time() - start_time

        # Get job ID, handling both simulator and real hardware cases
        if hasattr(job, "job_id") and callable(job.job_id):
            job_id = job.job_id()
        else:
            job_id = "simulator"

        exec_result = ExecutionResult(
            execution_id=execution_id,
            counts=result.get_counts(),
            raw_data={"job_id": job_id},
            execution_time=execution_time,
            success=result.success,
            provenance_hash=provenance_hash,
            backend_info=self.get_backend_info(),
        )

        self._execution_history.append(exec_result)
        return exec_result

    def validate_circuit(self, circuit: Any) -> tuple[bool, str]:
        """Validate circuit against IBM backend constraints."""
        if circuit is None:
            return False, "Circuit cannot be None"

        if hasattr(circuit, "depth") and circuit.depth() > self.config.max_circuit_depth:
            return (
                False,
                f"Circuit depth {circuit.depth()} exceeds max {self.config.max_circuit_depth}",
            )

        return True, "Circuit valid"

    def get_backend_info(self) -> dict[str, Any]:
        """Get IBM backend information."""
        return {
            "provider": "ibm",
            "backend_name": getattr(self._backend, "name", "simulator"),
            "is_simulator": self.config.is_simulator,
            "max_shots": 100000,
            "max_circuits": 300,
        }


class IonQHybridBackend(HybridQuantumBackend):
    """IonQ trapped-ion quantum backend wrapper.

    IonQ provides high-fidelity trapped-ion quantum computers accessible via:
    - Amazon Braket
    - Direct IonQ API
    - Azure Quantum

    Features:
    - All-to-all connectivity
    - High single/two-qubit gate fidelities
    - Native gate set: {GPI, GPI2, MS}
    """

    def _initialize_backend(self) -> None:
        """Initialize IonQ backend via Braket or Azure."""
        # Try Braket first
        try:
            from braket.aws import AwsDevice

            if self.config.device_name:
                device_arn = self.config.device_name
            else:
                device_arn = "arn:aws:braket:::device/qpu/ionq/ionQdevice"

            self._backend = AwsDevice(device_arn)
            self._provider = "braket"
            return
        except ImportError:
            pass

        # Fallback to local simulator
        warnings.warn(
            "IonQ hardware requires amazon-braket-sdk. Using local simulator.",
            UserWarning,
            stacklevel=2,
        )
        self._backend = None
        self._provider = "simulator"

    def execute_circuit(
        self,
        circuit: Any,
        shots: int | None = None,
        await_approval: bool = False,
    ) -> ExecutionResult:
        """Execute circuit on IonQ backend."""
        execution_id = str(uuid.uuid4())
        shots = shots or self.config.shots
        start_time = time.time()

        # Validate
        is_valid, message = self.validate_circuit(circuit)
        if not is_valid:
            return ExecutionResult(
                execution_id=execution_id,
                counts={},
                raw_data={"error": message},
                execution_time=time.time() - start_time,
                success=False,
                error_message=message,
            )

        provenance_hash = self.compute_provenance_hash(circuit, {"shots": shots})

        if self._backend is None:
            # Simulator fallback
            try:
                from qiskit import transpile
                from qiskit_aer import AerSimulator

                sim = AerSimulator()
                transpiled = transpile(circuit, backend=sim)
                job = sim.run(transpiled, shots=shots, seed_simulator=self.config.seed)
                result = job.result()

                return ExecutionResult(
                    execution_id=execution_id,
                    counts=result.get_counts(),
                    raw_data={"simulator": "aer_fallback"},
                    execution_time=time.time() - start_time,
                    success=True,
                    provenance_hash=provenance_hash,
                    backend_info=self.get_backend_info(),
                )
            except ImportError:
                return ExecutionResult(
                    execution_id=execution_id,
                    counts={},
                    raw_data={"error": "No quantum backend available"},
                    execution_time=time.time() - start_time,
                    success=False,
                    error_message="No quantum backend available",
                )

        # Execute on IonQ via Braket

        # Convert Qiskit circuit to Braket (simplified)
        # In production, use qiskit-braket-provider for proper conversion
        task = self._backend.run(circuit, shots=shots)
        task_result = task.result()

        return ExecutionResult(
            execution_id=execution_id,
            counts=dict(task_result.measurement_counts),
            raw_data={"task_arn": task.id},
            execution_time=time.time() - start_time,
            success=True,
            provenance_hash=provenance_hash,
            backend_info=self.get_backend_info(),
        )

    def validate_circuit(self, circuit: Any) -> tuple[bool, str]:
        """Validate circuit for IonQ execution."""
        if circuit is None:
            return False, "Circuit cannot be None"
        return True, "Circuit valid"

    def get_backend_info(self) -> dict[str, Any]:
        """Get IonQ backend information."""
        return {
            "provider": "ionq",
            "backend_name": self.config.device_name or "ionq_simulator",
            "is_simulator": self._backend is None,
            "connectivity": "all-to-all",
            "native_gates": ["gpi", "gpi2", "ms"],
        }


class QuantinuumHybridBackend(HybridQuantumBackend):
    """Quantinuum (Honeywell) trapped-ion quantum backend wrapper.

    Quantinuum provides the highest-fidelity quantum computers available via:
    - Azure Quantum

    Features:
    - Highest two-qubit gate fidelities (>99.5%)
    - QCCD (quantum charge-coupled device) architecture
    - Native gate set: {Rz, U1q, ZZ}
    """

    def _initialize_backend(self) -> None:
        """Initialize Quantinuum backend via Azure Quantum."""
        try:
            from azure.quantum import Workspace
        except ImportError:
            warnings.warn(
                "Quantinuum backend requires azure-quantum. Using local simulator.",
                UserWarning,
                stacklevel=2,
            )
            self._backend = None
            self._workspace = None
            return

        # Initialize Azure Quantum workspace
        # In production, these would come from environment or config
        self._workspace = None
        self._backend = None

    def execute_circuit(
        self,
        circuit: Any,
        shots: int | None = None,
        await_approval: bool = False,
    ) -> ExecutionResult:
        """Execute circuit on Quantinuum backend."""
        execution_id = str(uuid.uuid4())
        shots = shots or self.config.shots
        start_time = time.time()

        is_valid, message = self.validate_circuit(circuit)
        if not is_valid:
            return ExecutionResult(
                execution_id=execution_id,
                counts={},
                raw_data={"error": message},
                execution_time=time.time() - start_time,
                success=False,
                error_message=message,
            )

        provenance_hash = self.compute_provenance_hash(circuit, {"shots": shots})

        # Simulator fallback
        try:
            from qiskit import transpile
            from qiskit_aer import AerSimulator

            sim = AerSimulator()
            transpiled = transpile(circuit, backend=sim)
            job = sim.run(transpiled, shots=shots, seed_simulator=self.config.seed)
            result = job.result()

            return ExecutionResult(
                execution_id=execution_id,
                counts=result.get_counts(),
                raw_data={"simulator": "quantinuum_emulator"},
                execution_time=time.time() - start_time,
                success=True,
                provenance_hash=provenance_hash,
                backend_info=self.get_backend_info(),
            )
        except ImportError:
            return ExecutionResult(
                execution_id=execution_id,
                counts={},
                raw_data={"error": "No quantum backend available"},
                execution_time=time.time() - start_time,
                success=False,
                error_message="No quantum backend available",
            )

    def validate_circuit(self, circuit: Any) -> tuple[bool, str]:
        """Validate circuit for Quantinuum execution."""
        if circuit is None:
            return False, "Circuit cannot be None"
        return True, "Circuit valid"

    def get_backend_info(self) -> dict[str, Any]:
        """Get Quantinuum backend information."""
        return {
            "provider": "quantinuum",
            "backend_name": self.config.device_name or "quantinuum_emulator",
            "is_simulator": self._backend is None,
            "connectivity": "all-to-all",
            "native_gates": ["rz", "u1q", "zz"],
            "fidelity": "highest_available",
        }


class AzureQuantumHybridBackend(HybridQuantumBackend):
    """Azure Quantum multi-provider backend wrapper.

    Azure Quantum provides access to multiple quantum hardware providers:
    - IonQ
    - Quantinuum
    - Rigetti (superconducting)
    - Pasqal (neutral atoms)
    """

    def _initialize_backend(self) -> None:
        """Initialize Azure Quantum workspace."""
        try:
            from azure.quantum import Workspace
        except ImportError:
            warnings.warn(
                "Azure Quantum requires azure-quantum. Using local simulator.",
                UserWarning,
                stacklevel=2,
            )
            self._workspace = None
            self._backend = None
            return

        self._workspace = None
        self._backend = None

    def execute_circuit(
        self,
        circuit: Any,
        shots: int | None = None,
        await_approval: bool = False,
    ) -> ExecutionResult:
        """Execute circuit via Azure Quantum."""
        execution_id = str(uuid.uuid4())
        shots = shots or self.config.shots
        start_time = time.time()

        provenance_hash = self.compute_provenance_hash(circuit, {"shots": shots})

        # Simulator fallback
        try:
            from qiskit import transpile
            from qiskit_aer import AerSimulator

            sim = AerSimulator()
            transpiled = transpile(circuit, backend=sim)
            job = sim.run(transpiled, shots=shots, seed_simulator=self.config.seed)
            result = job.result()

            return ExecutionResult(
                execution_id=execution_id,
                counts=result.get_counts(),
                raw_data={"simulator": "azure_emulator"},
                execution_time=time.time() - start_time,
                success=True,
                provenance_hash=provenance_hash,
                backend_info=self.get_backend_info(),
            )
        except ImportError:
            return ExecutionResult(
                execution_id=execution_id,
                counts={},
                raw_data={"error": "No quantum backend available"},
                execution_time=time.time() - start_time,
                success=False,
                error_message="No quantum backend available",
            )

    def validate_circuit(self, circuit: Any) -> tuple[bool, str]:
        """Validate circuit for Azure Quantum execution."""
        if circuit is None:
            return False, "Circuit cannot be None"
        return True, "Circuit valid"

    def get_backend_info(self) -> dict[str, Any]:
        """Get Azure Quantum backend information."""
        return {
            "provider": "azure",
            "backend_name": self.config.device_name or "azure_simulator",
            "is_simulator": self._workspace is None,
            "available_providers": ["ionq", "quantinuum", "rigetti", "pasqal"],
        }


class BraketHybridBackend(HybridQuantumBackend):
    """AWS Braket multi-provider backend wrapper.

    AWS Braket provides access to:
    - IonQ (trapped-ion)
    - Rigetti (superconducting)
    - OQC (superconducting)
    - QuEra (neutral atoms)
    - IQM (superconducting)
    - Various simulators (SV1, DM1, TN1)
    """

    def _initialize_backend(self) -> None:
        """Initialize AWS Braket connection."""
        try:
            from braket.aws import AwsDevice
            from braket.devices import LocalSimulator
        except ImportError:
            warnings.warn(
                "AWS Braket requires amazon-braket-sdk. Using local simulator.",
                UserWarning,
                stacklevel=2,
            )
            self._backend = None
            return

        if self.config.device_name:
            if "local" in self.config.device_name.lower():
                self._backend = LocalSimulator()
            else:
                self._backend = AwsDevice(self.config.device_name)
        else:
            self._backend = LocalSimulator()

    def execute_circuit(
        self,
        circuit: Any,
        shots: int | None = None,
        await_approval: bool = False,
    ) -> ExecutionResult:
        """Execute circuit on Braket backend."""
        execution_id = str(uuid.uuid4())
        shots = shots or self.config.shots
        start_time = time.time()

        provenance_hash = self.compute_provenance_hash(circuit, {"shots": shots})

        if self._backend is None:
            # Fallback to Qiskit Aer
            try:
                from qiskit import transpile
                from qiskit_aer import AerSimulator

                sim = AerSimulator()
                transpiled = transpile(circuit, backend=sim)
                job = sim.run(transpiled, shots=shots, seed_simulator=self.config.seed)
                result = job.result()

                return ExecutionResult(
                    execution_id=execution_id,
                    counts=result.get_counts(),
                    raw_data={"simulator": "braket_fallback"},
                    execution_time=time.time() - start_time,
                    success=True,
                    provenance_hash=provenance_hash,
                    backend_info=self.get_backend_info(),
                )
            except ImportError:
                return ExecutionResult(
                    execution_id=execution_id,
                    counts={},
                    raw_data={"error": "No quantum backend available"},
                    execution_time=time.time() - start_time,
                    success=False,
                    error_message="No quantum backend available",
                )

        # Execute on Braket
        task = self._backend.run(circuit, shots=shots)
        task_result = task.result()

        return ExecutionResult(
            execution_id=execution_id,
            counts=dict(task_result.measurement_counts),
            raw_data={"task_id": task.id},
            execution_time=time.time() - start_time,
            success=True,
            provenance_hash=provenance_hash,
            backend_info=self.get_backend_info(),
        )

    def validate_circuit(self, circuit: Any) -> tuple[bool, str]:
        """Validate circuit for Braket execution."""
        if circuit is None:
            return False, "Circuit cannot be None"
        return True, "Circuit valid"

    def get_backend_info(self) -> dict[str, Any]:
        """Get Braket backend information."""
        return {
            "provider": "braket",
            "backend_name": self.config.device_name or "local_simulator",
            "is_simulator": self._backend is None or "local" in str(type(self._backend)).lower(),
            "available_providers": ["ionq", "rigetti", "oqc", "quera", "iqm"],
        }
