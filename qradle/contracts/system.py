"""
Contract Execution and Validation System

Provides contract executor and validator for QRADLE.
All contracts must be validated before execution.

Version: 1.0.0
"""

from dataclasses import dataclass
from typing import Any, Optional

from qradle.core.engine import DeterministicEngine, ExecutionContext


@dataclass
class ContractValidationResult:
    """Result of contract validation.

    Attributes:
        valid: Whether contract is valid
        errors: List of validation errors
        warnings: List of validation warnings
    """

    valid: bool
    errors: list[str]
    warnings: list[str]


class ContractValidator:
    """Validates contracts before execution.

    All contracts must pass validation before they can be executed.
    Validation checks:
    - Contract structure
    - Required fields
    - Safety level classification
    - Authorization requirements
    """

    def validate_contract(self, contract: dict[str, Any]) -> ContractValidationResult:
        """Validate a contract.

        Args:
            contract: Contract dictionary to validate

        Returns:
            ContractValidationResult
        """
        errors = []
        warnings = []

        # Check required fields
        required_fields = ["contract_id", "contract_type", "parameters"]
        for field in required_fields:
            if field not in contract:
                errors.append(f"Missing required field: {field}")

        # Check safety level
        if "safety_level" not in contract:
            warnings.append("No safety_level specified, defaulting to ROUTINE")
        else:
            valid_levels = ["ROUTINE", "ELEVATED", "SENSITIVE", "CRITICAL", "EXISTENTIAL"]
            if contract["safety_level"] not in valid_levels:
                errors.append(f"Invalid safety_level: {contract['safety_level']}")

        # Check authorization for sensitive operations
        safety_level = contract.get("safety_level", "ROUTINE")
        if safety_level in ["SENSITIVE", "CRITICAL", "EXISTENTIAL"]:
            if not contract.get("authorized", False):
                errors.append(f"Operation at {safety_level} level requires authorization")

        return ContractValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


class ContractExecutor:
    """Executes validated contracts using the deterministic engine.

    The executor ensures all contracts are validated and executed
    with full invariant enforcement.
    """

    def __init__(self, engine: Optional[DeterministicEngine] = None):
        """Initialize contract executor.

        Args:
            engine: Optional DeterministicEngine instance
        """
        self.engine = engine or DeterministicEngine()
        self.validator = ContractValidator()

    def execute(
        self, contract: dict[str, Any], executor_func: Any, create_checkpoint: bool = True
    ) -> dict[str, Any]:
        """Execute a contract.

        Args:
            contract: Contract to execute
            executor_func: Function to execute
            create_checkpoint: Whether to create checkpoint

        Returns:
            Execution result dictionary

        Raises:
            ValueError: If contract validation fails
        """
        # Validate contract
        validation = self.validator.validate_contract(contract)
        if not validation.valid:
            raise ValueError(f"Contract validation failed: {validation.errors}")

        # Create execution context
        context = ExecutionContext(
            contract_id=contract["contract_id"],
            parameters=contract.get("parameters", {}),
            timestamp=contract.get("timestamp", ""),
            safety_level=contract.get("safety_level", "ROUTINE"),
            authorized=contract.get("authorized", False),
            metadata=contract.get("metadata", {}),
        )

        # Execute with deterministic engine
        result = self.engine.execute_contract(
            context=context, executor_func=executor_func, create_checkpoint=create_checkpoint
        )

        return {
            "success": result.success,
            "output": result.output,
            "output_hash": result.output_hash,
            "execution_time": result.execution_time,
            "events_emitted": result.events_emitted,
            "checkpoint_id": result.checkpoint_id,
            "error": result.error,
        }

    def get_engine_stats(self) -> dict[str, Any]:
        """Get engine statistics.

        Returns:
            Statistics dictionary
        """
        return self.engine.get_stats()
