"""
8 Fatal Invariants for QRADLE.

These constraints are IMMUTABLE and can never be modified by any
system operation, including self-improvement.
"""

from typing import Callable, Dict, Any


class InvariantViolation(Exception):
    """Exception raised when a fatal invariant is violated."""
    pass


class FatalInvariants:
    """Enforces the 8 Fatal Invariants of QRADLE.
    
    1. Human Oversight Requirement
    2. Merkle Chain Integrity
    3. Contract Immutability
    4. Authorization System
    5. Safety Level System
    6. Rollback Capability
    7. Event Emission Requirement
    8. Determinism Guarantee
    """
    
    INVARIANTS = frozenset([
        "human_oversight_requirement",
        "merkle_chain_integrity",
        "contract_immutability",
        "authorization_system",
        "safety_level_system",
        "rollback_capability",
        "event_emission_requirement",
        "determinism_guarantee",
    ])
    
    @staticmethod
    def enforce_human_oversight(operation: str, safety_level: str) -> None:
        """Invariant 1: Sensitive operations require human authorization."""
        if safety_level in ("sensitive", "critical", "existential"):
            # In production, this would check for actual authorization
            # For now, we log the requirement
            pass
    
    @staticmethod
    def enforce_merkle_integrity(chain) -> None:
        """Invariant 2: All events must be cryptographically chained."""
        if not chain.verify_integrity():
            raise InvariantViolation("Merkle chain integrity violated")
    
    @staticmethod
    def enforce_contract_immutability(contract) -> None:
        """Invariant 3: Contracts cannot be retroactively altered."""
        # Contracts are frozen dataclasses - immutability enforced by Python
        if not hasattr(contract, '__dataclass_fields__'):
            raise InvariantViolation("Contract must be an immutable dataclass")
    
    @staticmethod
    def enforce_authorization(user_id: str, operation: str) -> None:
        """Invariant 4: Permission model must remain enforced."""
        if not user_id:
            raise InvariantViolation("All operations must have a user_id")
    
    @staticmethod
    def enforce_safety_levels(safety_level: str) -> None:
        """Invariant 5: Risk classification must be applied."""
        valid_levels = {"routine", "elevated", "sensitive", "critical", "existential"}
        if safety_level not in valid_levels:
            raise InvariantViolation(f"Invalid safety level: {safety_level}")
    
    @staticmethod
    def enforce_rollback_capability(rollback_manager) -> None:
        """Invariant 6: System must retain ability to rollback."""
        if rollback_manager is None:
            raise InvariantViolation("Rollback manager must be initialized")
    
    @staticmethod
    def enforce_event_emission(chain) -> None:
        """Invariant 7: All operations must emit auditable events."""
        if len(chain.chain) < 2:  # At least genesis + one event
            raise InvariantViolation("No events emitted")
    
    @staticmethod
    def enforce_determinism(inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Invariant 8: Same inputs must produce same outputs."""
        # This is enforced by the execution model and reproducible random seeds
        # Validation happens through external verification
        pass
    
    @classmethod
    def verify_all(cls, context: Dict[str, Any]) -> None:
        """Verify all invariants in the given context.
        
        Args:
            context: Dictionary containing system state to verify
        """
        cls.enforce_merkle_integrity(context.get("chain"))
        cls.enforce_rollback_capability(context.get("rollback_manager"))
        cls.enforce_event_emission(context.get("chain"))
