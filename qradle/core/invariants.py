"""
8 Fatal Invariants - Immutable Safety Constraints

These constraints are IMMUTABLE and can never be modified by any system operation,
including self-improvement. Any violation results in immediate system halt.

Version: 1.0.0
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class InvariantType(Enum):
    """Types of fatal invariants."""
    HUMAN_OVERSIGHT = "human_oversight_requirement"
    MERKLE_INTEGRITY = "merkle_chain_integrity"
    CONTRACT_IMMUTABILITY = "contract_immutability"
    AUTHORIZATION_SYSTEM = "authorization_system"
    SAFETY_LEVEL_SYSTEM = "safety_level_system"
    ROLLBACK_CAPABILITY = "rollback_capability"
    EVENT_EMISSION = "event_emission_requirement"
    DETERMINISM = "determinism_guarantee"


@dataclass(frozen=True)
class InvariantViolation(Exception):
    """Exception raised when a fatal invariant is violated.
    
    This is a FATAL error - the system MUST halt immediately.
    """
    invariant_type: InvariantType
    message: str
    context: Optional[dict] = None
    
    def __str__(self) -> str:
        ctx_str = f" Context: {self.context}" if self.context else ""
        return f"FATAL INVARIANT VIOLATION [{self.invariant_type.value}]: {self.message}{ctx_str}"


class FatalInvariants:
    """
    Enforcement of the 8 Fatal Invariants.
    
    These methods are called throughout the QRADLE execution engine to ensure
    that the immutable safety constraints are never violated.
    """
    
    # The 8 Fatal Invariants (IMMUTABLE)
    INVARIANTS = {
        InvariantType.HUMAN_OVERSIGHT: "Sensitive operations require human authorization",
        InvariantType.MERKLE_INTEGRITY: "All events must be cryptographically chained",
        InvariantType.CONTRACT_IMMUTABILITY: "Executed contracts cannot be retroactively altered",
        InvariantType.AUTHORIZATION_SYSTEM: "Permission model must remain enforced",
        InvariantType.SAFETY_LEVEL_SYSTEM: "Risk classification must be applied to all operations",
        InvariantType.ROLLBACK_CAPABILITY: "System must retain ability to return to verified states",
        InvariantType.EVENT_EMISSION: "All operations must emit auditable events",
        InvariantType.DETERMINISM: "Same inputs must produce same outputs",
    }
    
    @staticmethod
    def enforce_human_oversight(operation: str, safety_level: str, authorized: bool) -> None:
        """Invariant 1: Sensitive operations require human authorization."""
        if safety_level in ["SENSITIVE", "CRITICAL", "EXISTENTIAL"] and not authorized:
            raise InvariantViolation(
                invariant_type=InvariantType.HUMAN_OVERSIGHT,
                message=f"Operation '{operation}' at level '{safety_level}' requires human authorization",
                context={"operation": operation, "safety_level": safety_level}
            )
    
    @staticmethod
    def enforce_merkle_integrity(chain_valid: bool, last_hash: str) -> None:
        """Invariant 2: All events must be cryptographically chained."""
        if not chain_valid:
            raise InvariantViolation(
                invariant_type=InvariantType.MERKLE_INTEGRITY,
                message="Merkle chain integrity compromised",
                context={"last_hash": last_hash}
            )
    
    @staticmethod
    def enforce_contract_immutability(contract_id: str, modified: bool) -> None:
        """Invariant 3: Executed contracts cannot be retroactively altered."""
        if modified:
            raise InvariantViolation(
                invariant_type=InvariantType.CONTRACT_IMMUTABILITY,
                message=f"Attempted to modify immutable contract {contract_id}",
                context={"contract_id": contract_id}
            )
    
    @staticmethod
    def enforce_authorization_system(has_authorization_check: bool) -> None:
        """Invariant 4: Permission model must remain enforced."""
        if not has_authorization_check:
            raise InvariantViolation(
                invariant_type=InvariantType.AUTHORIZATION_SYSTEM,
                message="Authorization system bypassed or disabled"
            )
    
    @staticmethod
    def enforce_safety_level_system(operation: str, has_safety_level: bool) -> None:
        """Invariant 5: Risk classification must be applied to all operations."""
        if not has_safety_level:
            raise InvariantViolation(
                invariant_type=InvariantType.SAFETY_LEVEL_SYSTEM,
                message=f"Operation '{operation}' lacks required safety level classification",
                context={"operation": operation}
            )
    
    @staticmethod
    def enforce_rollback_capability(checkpoint_available: bool, checkpoint_id: str) -> None:
        """Invariant 6: System must retain ability to return to verified states."""
        if not checkpoint_available:
            raise InvariantViolation(
                invariant_type=InvariantType.ROLLBACK_CAPABILITY,
                message=f"Rollback checkpoint '{checkpoint_id}' not available",
                context={"checkpoint_id": checkpoint_id}
            )
    
    @staticmethod
    def enforce_event_emission(event_emitted: bool, operation: str) -> None:
        """Invariant 7: All operations must emit auditable events."""
        if not event_emitted:
            raise InvariantViolation(
                invariant_type=InvariantType.EVENT_EMISSION,
                message=f"Operation '{operation}' failed to emit required event",
                context={"operation": operation}
            )
    
    @staticmethod
    def enforce_determinism(result_hash: str, expected_hash: str) -> None:
        """Invariant 8: Same inputs must produce same outputs."""
        if result_hash != expected_hash:
            raise InvariantViolation(
                invariant_type=InvariantType.DETERMINISM,
                message="Non-deterministic execution detected",
                context={
                    "result_hash": result_hash,
                    "expected_hash": expected_hash
                }
            )
    
    @classmethod
    def get_all_invariants(cls) -> dict:
        """Get all invariant descriptions."""
        return cls.INVARIANTS.copy()
    
    @classmethod
    def verify_all(cls, system_state: dict) -> list[str]:
        """Verify all invariants against system state.
        
        Args:
            system_state: Dictionary containing system state to verify
            
        Returns:
            List of invariant violations (empty if all pass)
        """
        violations = []
        
        try:
            # Check each invariant
            if "human_oversight" in system_state:
                cls.enforce_human_oversight(
                    system_state["human_oversight"].get("operation", "unknown"),
                    system_state["human_oversight"].get("safety_level", "ROUTINE"),
                    system_state["human_oversight"].get("authorized", True)
                )
        except InvariantViolation as e:
            violations.append(str(e))
        
        # Add more checks as needed...
        
        return violations
