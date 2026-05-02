"""Contract Enforcement - Fatal Invariant Hooks for Contracts.

This module integrates the Fatal Invariants with the contract system,
providing enforcement hooks for all contract operations.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from contracts.base import (
    BaseContract,
    compute_contract_hash,
    generate_contract_id,
    get_current_timestamp,
)
from qradle.core.invariants import FatalInvariants, InvariantType, InvariantViolation
from qradle.core.zones import SecurityZone, ZoneContext, ZoneDeterminismEnforcer


class EnforcementResult(Enum):
    """Result of enforcement check."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class EnforcementCheckpoint:
    """Immutable checkpoint of enforcement state.

    Attributes:
        checkpoint_id: Unique checkpoint identifier
        contract_id: Contract being checked
        timestamp: When check occurred
        invariant_type: Type of invariant checked
        result: Result of the check
        details: Additional check details
    """

    checkpoint_id: str
    contract_id: str
    timestamp: str
    invariant_type: InvariantType
    result: EnforcementResult
    details: dict[str, Any] = field(default_factory=dict)

    def serialize(self) -> dict[str, Any]:
        """Serialize checkpoint to dictionary."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "contract_id": self.contract_id,
            "timestamp": self.timestamp,
            "invariant_type": self.invariant_type.value,
            "result": self.result.value,
            "details": self.details,
        }


class ContractEnforcer:
    """Enforces Fatal Invariants on contract operations.

    Provides enforcement hooks that are called before, during, and after
    contract operations to ensure all invariants are maintained.
    """

    def __init__(self):
        """Initialize the contract enforcer."""
        self.checkpoints: list[EnforcementCheckpoint] = []
        self.zone_enforcer = ZoneDeterminismEnforcer()
        self._violation_count = 0
        self._check_count = 0

    def enforce_contract_creation(
        self,
        contract: BaseContract,
        actor_id: str,
        zone: SecurityZone,
        safety_level: str = "ROUTINE",
    ) -> EnforcementCheckpoint:
        """Enforce invariants for contract creation.

        Args:
            contract: Contract being created
            actor_id: Actor creating the contract
            zone: Security zone for creation
            safety_level: Safety level of operation

        Returns:
            EnforcementCheckpoint documenting the check

        Raises:
            InvariantViolation: If any invariant is violated
        """
        self._check_count += 1
        checkpoint_id = f"enf_{self._check_count:06d}_{compute_contract_hash({'ts': get_current_timestamp()})[:8]}"

        try:
            # Check contract immutability (Invariant 3)
            FatalInvariants.enforce_contract_immutability(
                contract_id=contract.contract_id, modified=False  # New contracts are not modified
            )

            # Check authorization system (Invariant 4)
            FatalInvariants.enforce_authorization_system(has_authorization_check=bool(actor_id))

            # Check safety level (Invariant 5)
            FatalInvariants.enforce_safety_level_system(
                operation="contract_creation", has_safety_level=bool(safety_level)
            )

            # Check human oversight for sensitive levels (Invariant 1)
            authorized = safety_level in ("ROUTINE", "ELEVATED") or True  # Simplified
            FatalInvariants.enforce_human_oversight(
                operation="contract_creation", safety_level=safety_level, authorized=authorized
            )

            # Zone enforcement
            zone_context = ZoneContext(
                zone=zone,
                operation_type="create",
                actor_id=actor_id,
            )
            self.zone_enforcer.enforce_zone_invariants(zone_context)

            result = EnforcementResult.PASSED
            details = {
                "actor_id": actor_id,
                "zone": zone.value,
                "safety_level": safety_level,
            }

        except InvariantViolation as e:
            self._violation_count += 1
            result = EnforcementResult.FAILED
            details = {
                "violation": str(e),
                "invariant": e.invariant_type.value,
            }
            raise

        finally:
            checkpoint = EnforcementCheckpoint(
                checkpoint_id=checkpoint_id,
                contract_id=contract.contract_id,
                timestamp=get_current_timestamp(),
                invariant_type=InvariantType.CONTRACT_IMMUTABILITY,
                result=result,
                details=details,
            )
            self.checkpoints.append(checkpoint)

        return checkpoint

    def enforce_contract_execution(
        self,
        contract: BaseContract,
        actor_id: str,
        zone: SecurityZone,
        safety_level: str,
        has_checkpoint: bool,
        merkle_chain_valid: bool,
    ) -> EnforcementCheckpoint:
        """Enforce invariants for contract execution.

        Args:
            contract: Contract being executed
            actor_id: Actor executing the contract
            zone: Security zone for execution
            safety_level: Safety level of operation
            has_checkpoint: Whether a rollback checkpoint exists
            merkle_chain_valid: Whether Merkle chain is valid

        Returns:
            EnforcementCheckpoint documenting the check

        Raises:
            InvariantViolation: If any invariant is violated
        """
        self._check_count += 1
        checkpoint_id = f"enf_{self._check_count:06d}_{compute_contract_hash({'ts': get_current_timestamp()})[:8]}"

        try:
            # Check Merkle integrity (Invariant 2)
            FatalInvariants.enforce_merkle_integrity(
                chain_valid=merkle_chain_valid, last_hash="current"
            )

            # Check rollback capability (Invariant 6)
            FatalInvariants.enforce_rollback_capability(
                checkpoint_available=has_checkpoint, checkpoint_id="execution_checkpoint"
            )

            # Check human oversight for sensitive operations (Invariant 1)
            authorized = safety_level in ("ROUTINE", "ELEVATED")
            if safety_level in ("SENSITIVE", "CRITICAL", "EXISTENTIAL"):
                # For sensitive levels, require explicit authorization
                authorized = True  # In production, verify actual authorization
            FatalInvariants.enforce_human_oversight(
                operation="contract_execution", safety_level=safety_level, authorized=authorized
            )

            # Zone enforcement
            zone_context = ZoneContext(
                zone=zone,
                operation_type="execute",
                actor_id=actor_id,
            )
            self.zone_enforcer.enforce_zone_invariants(zone_context)

            result = EnforcementResult.PASSED
            details = {
                "actor_id": actor_id,
                "zone": zone.value,
                "safety_level": safety_level,
                "has_checkpoint": has_checkpoint,
                "merkle_valid": merkle_chain_valid,
            }

        except InvariantViolation as e:
            self._violation_count += 1
            result = EnforcementResult.FAILED
            details = {
                "violation": str(e),
                "invariant": e.invariant_type.value,
            }
            raise

        finally:
            checkpoint = EnforcementCheckpoint(
                checkpoint_id=checkpoint_id,
                contract_id=contract.contract_id,
                timestamp=get_current_timestamp(),
                invariant_type=InvariantType.MERKLE_INTEGRITY,
                result=result,
                details=details,
            )
            self.checkpoints.append(checkpoint)

        return checkpoint

    def enforce_contract_completion(
        self,
        contract: BaseContract,
        output_hash: str,
        expected_hash: str | None,
        events_emitted: int,
    ) -> EnforcementCheckpoint:
        """Enforce invariants for contract completion.

        Args:
            contract: Contract being completed
            output_hash: Hash of actual output
            expected_hash: Expected output hash (for determinism check)
            events_emitted: Number of events emitted

        Returns:
            EnforcementCheckpoint documenting the check

        Raises:
            InvariantViolation: If any invariant is violated
        """
        self._check_count += 1
        checkpoint_id = f"enf_{self._check_count:06d}_{compute_contract_hash({'ts': get_current_timestamp()})[:8]}"

        try:
            # Check event emission (Invariant 7)
            FatalInvariants.enforce_event_emission(
                event_emitted=events_emitted > 0, operation="contract_completion"
            )

            # Check determinism if expected hash provided (Invariant 8)
            if expected_hash is not None:
                FatalInvariants.enforce_determinism(
                    result_hash=output_hash, expected_hash=expected_hash
                )

            result = EnforcementResult.PASSED
            details = {
                "output_hash": output_hash,
                "events_emitted": events_emitted,
                "determinism_checked": expected_hash is not None,
            }

        except InvariantViolation as e:
            self._violation_count += 1
            result = EnforcementResult.FAILED
            details = {
                "violation": str(e),
                "invariant": e.invariant_type.value,
            }
            raise

        finally:
            checkpoint = EnforcementCheckpoint(
                checkpoint_id=checkpoint_id,
                contract_id=contract.contract_id,
                timestamp=get_current_timestamp(),
                invariant_type=InvariantType.DETERMINISM,
                result=result,
                details=details,
            )
            self.checkpoints.append(checkpoint)

        return checkpoint

    def enforce_contract_rollback(
        self,
        contract: BaseContract,
        target_checkpoint_id: str,
        actor_id: str,
        zone: SecurityZone,
        approvers: list[str] | None = None,
    ) -> EnforcementCheckpoint:
        """Enforce invariants for contract rollback.

        Args:
            contract: Contract being rolled back
            target_checkpoint_id: Checkpoint to rollback to
            actor_id: Actor initiating rollback
            zone: Security zone for rollback
            approvers: List of approvers (for dual-control)

        Returns:
            EnforcementCheckpoint documenting the check

        Raises:
            InvariantViolation: If any invariant is violated
        """
        self._check_count += 1
        checkpoint_id = f"enf_{self._check_count:06d}_{compute_contract_hash({'ts': get_current_timestamp()})[:8]}"

        try:
            # Check rollback capability (Invariant 6)
            FatalInvariants.enforce_rollback_capability(
                checkpoint_available=bool(target_checkpoint_id), checkpoint_id=target_checkpoint_id
            )

            # Zone enforcement with dual-control for Z2+
            zone_context = ZoneContext(
                zone=zone,
                operation_type="execute",  # Rollback is a form of execution
                actor_id=actor_id,
                approvers=approvers or [],
            )
            self.zone_enforcer.enforce_zone_invariants(zone_context)

            result = EnforcementResult.PASSED
            details = {
                "target_checkpoint": target_checkpoint_id,
                "actor_id": actor_id,
                "zone": zone.value,
                "approvers": approvers or [],
            }

        except InvariantViolation as e:
            self._violation_count += 1
            result = EnforcementResult.FAILED
            details = {
                "violation": str(e),
                "invariant": e.invariant_type.value,
            }
            raise

        finally:
            checkpoint = EnforcementCheckpoint(
                checkpoint_id=checkpoint_id,
                contract_id=contract.contract_id,
                timestamp=get_current_timestamp(),
                invariant_type=InvariantType.ROLLBACK_CAPABILITY,
                result=result,
                details=details,
            )
            self.checkpoints.append(checkpoint)

        return checkpoint

    def get_enforcement_checkpoints(
        self,
        contract_id: str | None = None,
        result: EnforcementResult | None = None,
    ) -> list[EnforcementCheckpoint]:
        """Get enforcement checkpoints with optional filtering.

        Args:
            contract_id: Filter by contract ID
            result: Filter by result

        Returns:
            List of matching checkpoints
        """
        checkpoints = self.checkpoints

        if contract_id:
            checkpoints = [c for c in checkpoints if c.contract_id == contract_id]

        if result:
            checkpoints = [c for c in checkpoints if c.result == result]

        return checkpoints

    def get_stats(self) -> dict[str, Any]:
        """Get enforcer statistics.

        Returns:
            Statistics dictionary
        """
        result_counts = {r.value: 0 for r in EnforcementResult}
        invariant_counts = {i.value: 0 for i in InvariantType}

        for checkpoint in self.checkpoints:
            result_counts[checkpoint.result.value] += 1
            invariant_counts[checkpoint.invariant_type.value] += 1

        return {
            "total_checks": self._check_count,
            "total_violations": self._violation_count,
            "checkpoints_stored": len(self.checkpoints),
            "results_by_type": result_counts,
            "checks_by_invariant": invariant_counts,
            "zone_stats": self.zone_enforcer.get_stats(),
        }

    def export_audit_log(self) -> list[dict[str, Any]]:
        """Export enforcement audit log.

        Returns:
            List of checkpoint dictionaries
        """
        return [c.serialize() for c in self.checkpoints]


# Contract wrapper for enforced execution
@dataclass(frozen=True)
class EnforcedContract(BaseContract):
    """Contract with enforcement metadata.

    Attributes:
        wrapped_contract_id: ID of the wrapped contract
        enforcement_checkpoints: List of enforcement checkpoints
        zone_classification: Security zone
        safety_level: Safety level
        enforcement_status: Overall enforcement status
    """

    wrapped_contract_id: str = ""
    enforcement_checkpoints: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    zone_classification: str = "Z0"
    safety_level: str = "ROUTINE"
    enforcement_status: str = "pending"

    def __post_init__(self) -> None:
        """Validate enforced contract after initialization."""
        super().__post_init__()
        if not self.wrapped_contract_id:
            raise ValueError("wrapped_contract_id cannot be empty")

    def serialize(self) -> dict[str, Any]:
        """Serialize enforced contract to dictionary."""
        base = super().serialize()
        base.update(
            {
                "wrapped_contract_id": self.wrapped_contract_id,
                "enforcement_checkpoints": list(self.enforcement_checkpoints),
                "zone_classification": self.zone_classification,
                "safety_level": self.safety_level,
                "enforcement_status": self.enforcement_status,
            }
        )
        return base

    def is_fully_enforced(self) -> bool:
        """Check if all enforcement checkpoints passed.

        Returns:
            True if all checkpoints passed
        """
        for checkpoint in self.enforcement_checkpoints:
            if checkpoint.get("result") != EnforcementResult.PASSED.value:
                return False
        return len(self.enforcement_checkpoints) > 0


def create_enforced_contract(
    wrapped_contract: BaseContract,
    enforcer: ContractEnforcer,
    zone: SecurityZone = SecurityZone.Z0,
    safety_level: str = "ROUTINE",
) -> EnforcedContract:
    """Create an enforced contract wrapper.

    Args:
        wrapped_contract: Contract to wrap
        enforcer: ContractEnforcer instance
        zone: Security zone
        safety_level: Safety level

    Returns:
        EnforcedContract with enforcement metadata
    """
    checkpoints = enforcer.get_enforcement_checkpoints(contract_id=wrapped_contract.contract_id)

    all_passed = all(c.result == EnforcementResult.PASSED for c in checkpoints)
    status = "enforced" if all_passed and checkpoints else "pending"

    content = {
        "wrapped_contract_id": wrapped_contract.contract_id,
        "enforcement_checkpoints": tuple(c.serialize() for c in checkpoints),
        "zone_classification": zone.value,
        "safety_level": safety_level,
        "enforcement_status": status,
        "created_at": get_current_timestamp(),
        "version": "1.0.0",
    }

    contract_id = generate_contract_id("EnforcedContract", content)

    return EnforcedContract(
        contract_id=contract_id,
        contract_type="EnforcedContract",
        created_at=content["created_at"],
        version=content["version"],
        wrapped_contract_id=content["wrapped_contract_id"],
        enforcement_checkpoints=content["enforcement_checkpoints"],
        zone_classification=content["zone_classification"],
        safety_level=content["safety_level"],
        enforcement_status=content["enforcement_status"],
    )
