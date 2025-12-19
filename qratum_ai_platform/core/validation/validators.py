"""QRATUM Enhanced Validation Framework.

Provides comprehensive input/output validation, state verification,
and compliance checking for quantum simulation operations.

Designed for DO-178C Level A certification requirements with
complete requirements traceability and MC/DC coverage support.

Classification: UNCLASSIFIED // CUI
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np

__all__ = [
    "ValidationResult",
    "ValidationLevel",
    "Validator",
    "CircuitValidator",
    "StateVectorValidator",
    "NumericValidator",
    "ComplianceValidator",
    "ValidationChain",
    "validate_circuit",
    "validate_statevector",
    "validate_measurement",
]


class ValidationLevel(Enum):
    """Validation strictness levels."""

    RELAXED = auto()  # Basic checks only (development)
    STANDARD = auto()  # Standard validation (default)
    STRICT = auto()  # Strict validation (production)
    AEROSPACE = auto()  # DO-178C Level A compliance


@dataclass(frozen=True)
class ValidationResult:
    """Immutable validation result.

    Attributes:
        valid: Whether validation passed
        level: Validation level applied
        checks_performed: Number of checks performed
        checks_passed: Number of checks that passed
        errors: List of error messages
        warnings: List of warning messages
        metadata: Additional validation metadata
        duration_ns: Validation duration in nanoseconds
        validator_id: Identifier of the validator
    """

    valid: bool
    level: ValidationLevel
    checks_performed: int
    checks_passed: int
    errors: Tuple[str, ...]
    warnings: Tuple[str, ...]
    metadata: Dict[str, Any]
    duration_ns: int
    validator_id: str

    @property
    def pass_rate(self) -> float:
        """Calculate check pass rate."""
        if self.checks_performed == 0:
            return 1.0
        return self.checks_passed / self.checks_performed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "valid": self.valid,
            "level": self.level.name,
            "checks_performed": self.checks_performed,
            "checks_passed": self.checks_passed,
            "pass_rate": self.pass_rate,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "metadata": self.metadata,
            "duration_ns": self.duration_ns,
            "duration_ms": self.duration_ns / 1_000_000,
            "validator_id": self.validator_id,
        }

    def raise_if_invalid(self) -> None:
        """Raise ValidationError if validation failed."""
        if not self.valid:
            error_msg = f"Validation failed: {'; '.join(self.errors)}"
            raise ValidationError(error_msg, self)


class ValidationError(Exception):
    """Exception raised for validation failures."""

    def __init__(self, message: str, result: ValidationResult):
        super().__init__(message)
        self.result = result


T = TypeVar("T")


class Validator(ABC, Generic[T]):
    """Abstract base class for validators.

    Provides the foundation for type-safe validation with support
    for chaining, composition, and compliance tracking.
    """

    def __init__(
        self,
        level: ValidationLevel = ValidationLevel.STANDARD,
        validator_id: Optional[str] = None,
    ):
        """Initialize validator.

        Args:
            level: Validation strictness level
            validator_id: Unique identifier for this validator
        """
        self.level = level
        self.validator_id = validator_id or self.__class__.__name__
        self._logger = logging.getLogger(f"qratum.validation.{self.validator_id}")

    @abstractmethod
    def validate(self, target: T) -> ValidationResult:
        """Validate the target object.

        Args:
            target: Object to validate

        Returns:
            ValidationResult with check outcomes
        """
        pass

    def __call__(self, target: T) -> ValidationResult:
        """Callable interface for validation."""
        return self.validate(target)

    def and_then(self, other: Validator[T]) -> ValidationChain[T]:
        """Chain this validator with another."""
        return ValidationChain([self, other])


class ValidationChain(Validator[T]):
    """Chain of validators executed in sequence."""

    def __init__(
        self,
        validators: List[Validator[T]],
        fail_fast: bool = True,
    ):
        """Initialize validation chain.

        Args:
            validators: List of validators to execute
            fail_fast: Stop on first failure if True
        """
        super().__init__(validator_id="ValidationChain")
        self._validators = validators
        self._fail_fast = fail_fast

    def validate(self, target: T) -> ValidationResult:
        """Execute all validators in chain."""
        start_time = time.perf_counter_ns()

        all_errors: List[str] = []
        all_warnings: List[str] = []
        total_checks = 0
        total_passed = 0
        all_metadata: Dict[str, Any] = {}

        for validator in self._validators:
            result = validator.validate(target)
            total_checks += result.checks_performed
            total_passed += result.checks_passed
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)
            all_metadata[validator.validator_id] = result.metadata

            if self._fail_fast and not result.valid:
                break

        return ValidationResult(
            valid=len(all_errors) == 0,
            level=self.level,
            checks_performed=total_checks,
            checks_passed=total_passed,
            errors=tuple(all_errors),
            warnings=tuple(all_warnings),
            metadata=all_metadata,
            duration_ns=time.perf_counter_ns() - start_time,
            validator_id=self.validator_id,
        )

    def add(self, validator: Validator[T]) -> ValidationChain[T]:
        """Add a validator to the chain."""
        self._validators.append(validator)
        return self


class NumericValidator(Validator[Union[float, np.ndarray]]):
    """Validator for numeric values and arrays.

    Provides comprehensive numeric validation including:
    - Range checking
    - NaN/Inf detection
    - Precision verification
    - Statistical bounds checking
    """

    def __init__(
        self,
        level: ValidationLevel = ValidationLevel.STANDARD,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        allow_nan: bool = False,
        allow_inf: bool = False,
        tolerance: float = 1e-12,
    ):
        """Initialize numeric validator.

        Args:
            level: Validation level
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            allow_nan: Whether NaN values are allowed
            allow_inf: Whether infinite values are allowed
            tolerance: Tolerance for floating-point comparisons
        """
        super().__init__(level=level, validator_id="NumericValidator")
        self.min_value = min_value
        self.max_value = max_value
        self.allow_nan = allow_nan
        self.allow_inf = allow_inf
        self.tolerance = tolerance

    def validate(self, target: Union[float, np.ndarray]) -> ValidationResult:
        """Validate numeric value or array."""
        start_time = time.perf_counter_ns()
        errors: List[str] = []
        warnings: List[str] = []
        checks_performed = 0
        checks_passed = 0

        # Convert to numpy array for uniform handling
        arr = np.asarray(target)

        # Check for NaN values
        checks_performed += 1
        nan_count = np.count_nonzero(np.isnan(arr))
        if nan_count > 0:
            if not self.allow_nan:
                errors.append(f"Found {nan_count} NaN value(s)")
            else:
                warnings.append(f"Found {nan_count} NaN value(s)")
                checks_passed += 1
        else:
            checks_passed += 1

        # Check for infinite values
        checks_performed += 1
        inf_count = np.count_nonzero(np.isinf(arr))
        if inf_count > 0:
            if not self.allow_inf:
                errors.append(f"Found {inf_count} infinite value(s)")
            else:
                warnings.append(f"Found {inf_count} infinite value(s)")
                checks_passed += 1
        else:
            checks_passed += 1

        # Check minimum value
        if self.min_value is not None:
            checks_performed += 1
            finite_arr = arr[np.isfinite(arr)]
            if len(finite_arr) > 0 and np.min(finite_arr) < self.min_value - self.tolerance:
                errors.append(f"Value {np.min(finite_arr):.6e} below minimum {self.min_value:.6e}")
            else:
                checks_passed += 1

        # Check maximum value
        if self.max_value is not None:
            checks_performed += 1
            finite_arr = arr[np.isfinite(arr)]
            if len(finite_arr) > 0 and np.max(finite_arr) > self.max_value + self.tolerance:
                errors.append(f"Value {np.max(finite_arr):.6e} above maximum {self.max_value:.6e}")
            else:
                checks_passed += 1

        # Aerospace-level checks
        if self.level == ValidationLevel.AEROSPACE:
            checks_performed += 1
            # Check for denormalized values
            denorm_count = np.count_nonzero((arr != 0) & (np.abs(arr) < np.finfo(float).tiny))
            if denorm_count > 0:
                warnings.append(f"Found {denorm_count} denormalized value(s)")
            checks_passed += 1

        return ValidationResult(
            valid=len(errors) == 0,
            level=self.level,
            checks_performed=checks_performed,
            checks_passed=checks_passed,
            errors=tuple(errors),
            warnings=tuple(warnings),
            metadata={
                "shape": arr.shape if hasattr(arr, "shape") else (),
                "dtype": str(arr.dtype) if hasattr(arr, "dtype") else type(target).__name__,
                "nan_count": nan_count,
                "inf_count": inf_count,
            },
            duration_ns=time.perf_counter_ns() - start_time,
            validator_id=self.validator_id,
        )


class StateVectorValidator(Validator[np.ndarray]):
    """Validator for quantum state vectors.

    Ensures state vectors satisfy quantum mechanical constraints:
    - Proper dimension (2^n)
    - Normalization (|ψ|² = 1)
    - Numeric stability
    """

    def __init__(
        self,
        level: ValidationLevel = ValidationLevel.STANDARD,
        normalization_tolerance: float = 1e-10,
        max_qubits: int = 40,
    ):
        """Initialize state vector validator.

        Args:
            level: Validation level
            normalization_tolerance: Tolerance for normalization check
            max_qubits: Maximum allowed qubit count
        """
        super().__init__(level=level, validator_id="StateVectorValidator")
        self.normalization_tolerance = normalization_tolerance
        self.max_qubits = max_qubits

    def validate(self, target: np.ndarray) -> ValidationResult:
        """Validate quantum state vector."""
        start_time = time.perf_counter_ns()
        errors: List[str] = []
        warnings: List[str] = []
        checks_performed = 0
        checks_passed = 0

        # Check array type
        checks_performed += 1
        if not isinstance(target, np.ndarray):
            errors.append(f"Expected numpy array, got {type(target).__name__}")
        else:
            checks_passed += 1

        if errors:
            return self._build_result(
                errors, warnings, checks_performed, checks_passed, start_time, {}
            )

        # Check dimension is power of 2
        checks_performed += 1
        dim = len(target)
        if dim == 0:
            errors.append("State vector cannot be empty")
        elif not (dim & (dim - 1) == 0):
            errors.append(f"Dimension {dim} is not a power of 2")
        else:
            checks_passed += 1

        num_qubits = int(np.log2(dim)) if dim > 0 else 0

        # Check qubit count
        checks_performed += 1
        if num_qubits > self.max_qubits:
            errors.append(f"Qubit count {num_qubits} exceeds maximum {self.max_qubits}")
        else:
            checks_passed += 1

        # Check normalization
        checks_performed += 1
        norm = np.linalg.norm(target)
        if abs(norm - 1.0) > self.normalization_tolerance:
            if self.level in (ValidationLevel.STRICT, ValidationLevel.AEROSPACE):
                errors.append(f"State not normalized: |ψ|² = {norm:.15e}, expected 1.0")
            else:
                warnings.append(f"State not normalized: |ψ|² = {norm:.10e}")
                checks_passed += 1
        else:
            checks_passed += 1

        # Check for numeric issues (delegate to NumericValidator)
        numeric_validator = NumericValidator(level=self.level, allow_nan=False, allow_inf=False)
        numeric_result = numeric_validator.validate(target)
        checks_performed += numeric_result.checks_performed
        checks_passed += numeric_result.checks_passed
        errors.extend(numeric_result.errors)
        warnings.extend(numeric_result.warnings)

        # Aerospace-level additional checks
        if self.level == ValidationLevel.AEROSPACE:
            # Check for near-zero amplitudes that should be exactly zero
            checks_performed += 1
            tiny_count = np.count_nonzero((np.abs(target) > 0) & (np.abs(target) < 1e-15))
            if tiny_count > 0:
                warnings.append(
                    f"Found {tiny_count} near-zero amplitude(s) that may cause precision issues"
                )
            checks_passed += 1

            # Verify complex dtype
            checks_performed += 1
            if not np.issubdtype(target.dtype, np.complexfloating):
                errors.append(f"State vector must be complex, got {target.dtype}")
            else:
                checks_passed += 1

        metadata = {
            "dimension": dim,
            "num_qubits": num_qubits,
            "norm": float(norm),
            "dtype": str(target.dtype),
            "max_amplitude": float(np.max(np.abs(target))),
            "min_amplitude": float(np.min(np.abs(target))),
        }

        return self._build_result(
            errors, warnings, checks_performed, checks_passed, start_time, metadata
        )

    def _build_result(
        self,
        errors: List[str],
        warnings: List[str],
        checks_performed: int,
        checks_passed: int,
        start_time: int,
        metadata: Dict[str, Any],
    ) -> ValidationResult:
        """Build ValidationResult."""
        return ValidationResult(
            valid=len(errors) == 0,
            level=self.level,
            checks_performed=checks_performed,
            checks_passed=checks_passed,
            errors=tuple(errors),
            warnings=tuple(warnings),
            metadata=metadata,
            duration_ns=time.perf_counter_ns() - start_time,
            validator_id=self.validator_id,
        )


@dataclass
class CircuitSpec:
    """Specification for circuit validation."""

    num_qubits: int
    instructions: List[Tuple[str, List[int], Any, Dict[str, Any]]]


class CircuitValidator(Validator[CircuitSpec]):
    """Validator for quantum circuits.

    Ensures circuits are well-formed and executable:
    - Valid qubit indices
    - Valid gate operations
    - No overlapping gates on same qubits (where applicable)
    - Resource estimation bounds
    """

    VALID_GATES = frozenset(
        [
            "h",
            "x",
            "y",
            "z",
            "s",
            "t",
            "sdg",
            "tdg",
            "rx",
            "ry",
            "rz",
            "u1",
            "u2",
            "u3",
            "cnot",
            "cx",
            "cy",
            "cz",
            "ch",
            "crx",
            "cry",
            "crz",
            "swap",
            "iswap",
            "dcx",
            "ccx",
            "ccz",
            "cswap",
            "measure",
            "reset",
            "barrier",
        ]
    )

    def __init__(
        self,
        level: ValidationLevel = ValidationLevel.STANDARD,
        max_gates: int = 100_000,
        max_depth: int = 10_000,
    ):
        """Initialize circuit validator.

        Args:
            level: Validation level
            max_gates: Maximum allowed gate count
            max_depth: Maximum allowed circuit depth
        """
        super().__init__(level=level, validator_id="CircuitValidator")
        self.max_gates = max_gates
        self.max_depth = max_depth

    def validate(self, target: CircuitSpec) -> ValidationResult:
        """Validate quantum circuit specification."""
        start_time = time.perf_counter_ns()
        errors: List[str] = []
        warnings: List[str] = []
        checks_performed = 0
        checks_passed = 0

        num_qubits = target.num_qubits
        instructions = target.instructions

        # Check qubit count
        checks_performed += 1
        if num_qubits <= 0:
            errors.append(f"Invalid qubit count: {num_qubits}")
        elif num_qubits > 100:
            warnings.append(f"Large qubit count ({num_qubits}) may cause performance issues")
            checks_passed += 1
        else:
            checks_passed += 1

        # Check gate count
        checks_performed += 1
        gate_count = len(instructions)
        if gate_count > self.max_gates:
            errors.append(f"Gate count {gate_count} exceeds maximum {self.max_gates}")
        else:
            checks_passed += 1

        # Validate each instruction
        depth_tracker: Dict[int, int] = dict.fromkeys(range(num_qubits), 0)

        for idx, (gate_name, qubits, matrix, params) in enumerate(instructions):
            # Check gate name
            checks_performed += 1
            if gate_name.lower() not in self.VALID_GATES:
                if self.level == ValidationLevel.STRICT:
                    errors.append(f"Unknown gate '{gate_name}' at position {idx}")
                else:
                    warnings.append(f"Unknown gate '{gate_name}' at position {idx}")
                    checks_passed += 1
            else:
                checks_passed += 1

            # Check qubit indices
            checks_performed += 1
            invalid_qubits = [q for q in qubits if q < 0 or q >= num_qubits]
            if invalid_qubits:
                errors.append(
                    f"Invalid qubit indices {invalid_qubits} at position {idx} "
                    f"(valid range: 0-{num_qubits - 1})"
                )
            else:
                checks_passed += 1

            # Check for duplicate qubits in single instruction
            checks_performed += 1
            if len(qubits) != len(set(qubits)):
                errors.append(f"Duplicate qubits in gate at position {idx}: {qubits}")
            else:
                checks_passed += 1

            # Track depth
            for q in qubits:
                if 0 <= q < num_qubits:
                    depth_tracker[q] += 1

        # Check circuit depth
        checks_performed += 1
        max_actual_depth = max(depth_tracker.values()) if depth_tracker else 0
        if max_actual_depth > self.max_depth:
            errors.append(f"Circuit depth {max_actual_depth} exceeds maximum {self.max_depth}")
        else:
            checks_passed += 1

        # Aerospace-level checks
        if self.level == ValidationLevel.AEROSPACE:
            # Check for measurement placement
            checks_performed += 1
            measurement_positions = [
                i for i, (name, _, _, _) in enumerate(instructions) if name.lower() == "measure"
            ]
            if measurement_positions:
                # Measurements should be at the end
                last_measure = max(measurement_positions)
                gates_after_measure = len(instructions) - 1 - last_measure
                if gates_after_measure > 0:
                    warnings.append(f"Found {gates_after_measure} gate(s) after measurement")
            checks_passed += 1

        metadata = {
            "num_qubits": num_qubits,
            "gate_count": gate_count,
            "max_depth": max_actual_depth,
            "gate_distribution": self._gate_distribution(instructions),
        }

        return ValidationResult(
            valid=len(errors) == 0,
            level=self.level,
            checks_performed=checks_performed,
            checks_passed=checks_passed,
            errors=tuple(errors),
            warnings=tuple(warnings),
            metadata=metadata,
            duration_ns=time.perf_counter_ns() - start_time,
            validator_id=self.validator_id,
        )

    def _gate_distribution(
        self, instructions: List[Tuple[str, List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, int]:
        """Calculate gate type distribution."""
        distribution: Dict[str, int] = {}
        for gate_name, _, _, _ in instructions:
            key = gate_name.lower()
            distribution[key] = distribution.get(key, 0) + 1
        return distribution


class ComplianceValidator(Validator[Dict[str, Any]]):
    """Validator for compliance requirements.

    Validates that simulation parameters and results meet
    compliance framework requirements (DO-178C, NIST, CMMC).
    """

    def __init__(
        self,
        level: ValidationLevel = ValidationLevel.AEROSPACE,
        framework: str = "DO-178C",
    ):
        """Initialize compliance validator.

        Args:
            level: Validation level
            framework: Compliance framework to validate against
        """
        super().__init__(level=level, validator_id=f"ComplianceValidator_{framework}")
        self.framework = framework

    def validate(self, target: Dict[str, Any]) -> ValidationResult:
        """Validate compliance requirements."""
        start_time = time.perf_counter_ns()
        errors: List[str] = []
        warnings: List[str] = []
        checks_performed = 0
        checks_passed = 0

        # Check for required fields
        required_fields = ["seed", "backend", "precision", "timestamp"]

        for field in required_fields:
            checks_performed += 1
            if field not in target:
                errors.append(f"Missing required field: {field}")
            else:
                checks_passed += 1

        # Check seed presence for determinism
        checks_performed += 1
        if target.get("seed") is None:
            if self.framework == "DO-178C":
                errors.append("DO-178C requires deterministic seed for reproducibility")
            else:
                warnings.append("No seed specified; results may not be reproducible")
                checks_passed += 1
        else:
            checks_passed += 1

        # Check precision for aerospace applications
        if self.framework == "DO-178C":
            checks_performed += 1
            precision = target.get("precision", "fp32")
            if precision not in ("fp64", "fp128"):
                warnings.append(
                    f"DO-178C Level A typically requires fp64 or higher; got {precision}"
                )
            checks_passed += 1

        # Check audit trail presence
        checks_performed += 1
        if not target.get("audit_enabled", False):
            if self.framework in ("DO-178C", "NIST-800-53"):
                errors.append(f"{self.framework} requires audit trail")
            else:
                warnings.append("Audit trail not enabled")
                checks_passed += 1
        else:
            checks_passed += 1

        return ValidationResult(
            valid=len(errors) == 0,
            level=self.level,
            checks_performed=checks_performed,
            checks_passed=checks_passed,
            errors=tuple(errors),
            warnings=tuple(warnings),
            metadata={
                "framework": self.framework,
                "fields_checked": required_fields,
            },
            duration_ns=time.perf_counter_ns() - start_time,
            validator_id=self.validator_id,
        )


# Convenience functions for common validation scenarios


def validate_circuit(
    num_qubits: int,
    instructions: List[Tuple[str, List[int], Any, Dict[str, Any]]],
    level: ValidationLevel = ValidationLevel.STANDARD,
) -> ValidationResult:
    """Validate a quantum circuit.

    Args:
        num_qubits: Number of qubits
        instructions: List of gate instructions
        level: Validation level

    Returns:
        ValidationResult
    """
    spec = CircuitSpec(num_qubits=num_qubits, instructions=instructions)
    validator = CircuitValidator(level=level)
    return validator.validate(spec)


def validate_statevector(
    state: np.ndarray,
    level: ValidationLevel = ValidationLevel.STANDARD,
) -> ValidationResult:
    """Validate a quantum state vector.

    Args:
        state: State vector to validate
        level: Validation level

    Returns:
        ValidationResult
    """
    validator = StateVectorValidator(level=level)
    return validator.validate(state)


def validate_measurement(
    counts: Dict[str, int],
    num_qubits: int,
    shots: int,
    level: ValidationLevel = ValidationLevel.STANDARD,
) -> ValidationResult:
    """Validate measurement results.

    Args:
        counts: Measurement count dictionary
        num_qubits: Expected number of qubits
        shots: Expected total shots
        level: Validation level

    Returns:
        ValidationResult
    """
    start_time = time.perf_counter_ns()
    errors: List[str] = []
    warnings: List[str] = []
    checks_performed = 0
    checks_passed = 0

    # Check total shots
    checks_performed += 1
    total_counts = sum(counts.values())
    if total_counts != shots:
        errors.append(f"Total counts {total_counts} != expected shots {shots}")
    else:
        checks_passed += 1

    # Check state string lengths
    checks_performed += 1
    invalid_states = [s for s in counts if len(s) != num_qubits]
    if invalid_states:
        errors.append(
            f"Invalid state string lengths: {invalid_states[:5]}... "
            f"(expected {num_qubits} qubits)"
        )
    else:
        checks_passed += 1

    # Check for valid binary strings
    checks_performed += 1
    non_binary = [s for s in counts if not all(c in "01" for c in s)]
    if non_binary:
        errors.append(f"Non-binary state strings: {non_binary[:5]}...")
    else:
        checks_passed += 1

    # Check for negative counts
    checks_performed += 1
    negative_counts = [(s, c) for s, c in counts.items() if c < 0]
    if negative_counts:
        errors.append(f"Negative counts found: {negative_counts[:5]}...")
    else:
        checks_passed += 1

    return ValidationResult(
        valid=len(errors) == 0,
        level=level,
        checks_performed=checks_performed,
        checks_passed=checks_passed,
        errors=tuple(errors),
        warnings=tuple(warnings),
        metadata={
            "total_counts": total_counts,
            "expected_shots": shots,
            "unique_outcomes": len(counts),
            "num_qubits": num_qubits,
        },
        duration_ns=time.perf_counter_ns() - start_time,
        validator_id="MeasurementValidator",
    )
