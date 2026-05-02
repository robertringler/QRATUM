"""Zone Determinism - Security Zone Invariants for QRADLE.

This module implements zone determinism invariants across Z0-Z3 security zones
with runtime assertions ensuring proper isolation and deterministic execution.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable


class SecurityZone(Enum):
    """QRADLE Security Zones.

    Z0 - Public Zone: Non-sensitive operations, read-only data
    Z1 - Operational Zone: Standard business operations
    Z2 - Sensitive Zone: Sensitive data, dual-control required
    Z3 - Critical Zone: Air-gapped, maximum security
    """

    Z0 = "Z0"  # Public
    Z1 = "Z1"  # Operational
    Z2 = "Z2"  # Sensitive
    Z3 = "Z3"  # Critical


class ZoneViolation(Exception):
    """Exception raised when a zone invariant is violated."""

    def __init__(
        self,
        zone: SecurityZone,
        invariant: str,
        message: str,
        context: dict[str, Any] | None = None,
    ):
        self.zone = zone
        self.invariant = invariant
        self.context = context or {}
        super().__init__(f"ZONE VIOLATION [{zone.value}] - {invariant}: {message}")


@dataclass(frozen=True)
class ZonePolicy:
    """Immutable policy for a security zone.

    Attributes:
        zone: Security zone
        allow_external_network: Whether external network access is allowed
        require_dual_control: Whether dual-control approval is required
        require_audit_trail: Whether audit trail is required
        max_operation_time_seconds: Maximum operation duration
        allowed_operations: Set of allowed operation types
        require_human_oversight: Whether human oversight is required
        require_air_gap: Whether air-gap isolation is required
    """

    zone: SecurityZone
    allow_external_network: bool = True
    require_dual_control: bool = False
    require_audit_trail: bool = True
    max_operation_time_seconds: float = 3600.0  # 1 hour default
    allowed_operations: frozenset[str] = field(default_factory=frozenset)
    require_human_oversight: bool = False
    require_air_gap: bool = False

    def serialize(self) -> dict[str, Any]:
        """Serialize policy to dictionary."""
        return {
            "zone": self.zone.value,
            "allow_external_network": self.allow_external_network,
            "require_dual_control": self.require_dual_control,
            "require_audit_trail": self.require_audit_trail,
            "max_operation_time_seconds": self.max_operation_time_seconds,
            "allowed_operations": list(self.allowed_operations),
            "require_human_oversight": self.require_human_oversight,
            "require_air_gap": self.require_air_gap,
        }


# Default zone policies
DEFAULT_ZONE_POLICIES: dict[SecurityZone, ZonePolicy] = {
    SecurityZone.Z0: ZonePolicy(
        zone=SecurityZone.Z0,
        allow_external_network=True,
        require_dual_control=False,
        require_audit_trail=False,
        max_operation_time_seconds=3600.0,
        allowed_operations=frozenset({"read", "query", "list"}),
        require_human_oversight=False,
        require_air_gap=False,
    ),
    SecurityZone.Z1: ZonePolicy(
        zone=SecurityZone.Z1,
        allow_external_network=True,
        require_dual_control=False,
        require_audit_trail=True,
        max_operation_time_seconds=1800.0,
        allowed_operations=frozenset({"read", "query", "list", "create", "update", "execute"}),
        require_human_oversight=False,
        require_air_gap=False,
    ),
    SecurityZone.Z2: ZonePolicy(
        zone=SecurityZone.Z2,
        allow_external_network=False,
        require_dual_control=True,
        require_audit_trail=True,
        max_operation_time_seconds=900.0,
        allowed_operations=frozenset({"read", "query", "create", "update", "execute", "approve"}),
        require_human_oversight=True,
        require_air_gap=False,
    ),
    SecurityZone.Z3: ZonePolicy(
        zone=SecurityZone.Z3,
        allow_external_network=False,
        require_dual_control=True,
        require_audit_trail=True,
        max_operation_time_seconds=300.0,
        allowed_operations=frozenset({"read", "execute", "archive", "verify"}),
        require_human_oversight=True,
        require_air_gap=True,
    ),
}


@dataclass
class ZoneContext:
    """Execution context for zone operations.

    Attributes:
        zone: Current security zone
        operation_type: Type of operation being performed
        actor_id: ID of the actor performing the operation
        approvers: List of approver IDs (for dual-control)
        start_time: Operation start timestamp
        external_network_requested: Whether external network is requested
        metadata: Additional context metadata
    """

    zone: SecurityZone
    operation_type: str
    actor_id: str
    approvers: list[str] = field(default_factory=list)
    start_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    external_network_requested: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def has_dual_control(self) -> bool:
        """Check if dual-control is satisfied.

        Dual-control requires two distinct individuals to authorize an operation:
        1. The actor performing the operation
        2. At least one additional approver who is not the actor

        Returns:
            True if the actor plus at least one distinct approver exist
        """
        unique_approvers = set(self.approvers)
        unique_approvers.discard(self.actor_id)  # Actor cannot self-approve
        # Dual control: actor (1) + at least one distinct approver (1) = 2 people
        return len(unique_approvers) >= 1

    def get_elapsed_seconds(self) -> float:
        """Get elapsed time since operation start.

        Returns:
            Elapsed seconds
        """
        start = datetime.fromisoformat(self.start_time.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        return (now - start).total_seconds()


class ZoneDeterminismEnforcer:
    """Enforces zone determinism invariants at runtime.

    All operations must pass through this enforcer to ensure:
    - Zone policies are respected
    - Operations are deterministic within zones
    - State transitions are properly logged
    - Invariants are never violated
    """

    def __init__(
        self,
        policies: dict[SecurityZone, ZonePolicy] | None = None,
    ):
        """Initialize the enforcer.

        Args:
            policies: Optional custom zone policies
        """
        self.policies = policies or DEFAULT_ZONE_POLICIES.copy()
        self._transition_log: list[dict[str, Any]] = []
        self._violation_count = 0

    def get_policy(self, zone: SecurityZone) -> ZonePolicy:
        """Get policy for a zone.

        Args:
            zone: Security zone

        Returns:
            ZonePolicy for the zone
        """
        return self.policies.get(zone, DEFAULT_ZONE_POLICIES[zone])

    def enforce_zone_invariants(self, context: ZoneContext) -> None:
        """Enforce all zone invariants for the given context.

        Args:
            context: Zone execution context

        Raises:
            ZoneViolation: If any invariant is violated
        """
        policy = self.get_policy(context.zone)

        # Invariant 1: Operation allowed in zone
        self._enforce_operation_allowed(context, policy)

        # Invariant 2: External network policy
        self._enforce_network_policy(context, policy)

        # Invariant 3: Dual-control requirement
        self._enforce_dual_control(context, policy)

        # Invariant 4: Operation time limit
        self._enforce_time_limit(context, policy)

        # Invariant 5: Air-gap requirement
        self._enforce_air_gap(context, policy)

        # Invariant 6: Audit trail requirement
        self._enforce_audit_trail(context, policy)

        # Log successful enforcement
        self._log_transition(context, "invariants_enforced")

    def _enforce_operation_allowed(self, context: ZoneContext, policy: ZonePolicy) -> None:
        """Enforce that operation is allowed in zone."""
        if policy.allowed_operations and context.operation_type not in policy.allowed_operations:
            self._violation_count += 1
            raise ZoneViolation(
                zone=context.zone,
                invariant="operation_allowed",
                message=f"Operation '{context.operation_type}' not allowed in zone {context.zone.value}",
                context={
                    "operation": context.operation_type,
                    "allowed": list(policy.allowed_operations),
                },
            )

    def _enforce_network_policy(self, context: ZoneContext, policy: ZonePolicy) -> None:
        """Enforce network access policy."""
        if context.external_network_requested and not policy.allow_external_network:
            self._violation_count += 1
            raise ZoneViolation(
                zone=context.zone,
                invariant="network_policy",
                message=f"External network access not allowed in zone {context.zone.value}",
                context={"external_network_requested": True},
            )

    def _enforce_dual_control(self, context: ZoneContext, policy: ZonePolicy) -> None:
        """Enforce dual-control requirement."""
        if policy.require_dual_control and not context.has_dual_control():
            self._violation_count += 1
            raise ZoneViolation(
                zone=context.zone,
                invariant="dual_control",
                message=f"Dual-control required in zone {context.zone.value}",
                context={
                    "actor": context.actor_id,
                    "approvers": context.approvers,
                },
            )

    def _enforce_time_limit(self, context: ZoneContext, policy: ZonePolicy) -> None:
        """Enforce operation time limit."""
        elapsed = context.get_elapsed_seconds()
        if elapsed > policy.max_operation_time_seconds:
            self._violation_count += 1
            raise ZoneViolation(
                zone=context.zone,
                invariant="time_limit",
                message=f"Operation exceeded time limit in zone {context.zone.value}",
                context={
                    "elapsed_seconds": elapsed,
                    "max_seconds": policy.max_operation_time_seconds,
                },
            )

    def _enforce_air_gap(self, context: ZoneContext, policy: ZonePolicy) -> None:
        """Enforce air-gap requirement."""
        if policy.require_air_gap:
            # In air-gapped mode, no external network is allowed
            if context.external_network_requested:
                self._violation_count += 1
                raise ZoneViolation(
                    zone=context.zone,
                    invariant="air_gap",
                    message=f"Air-gap violation in zone {context.zone.value}",
                    context={"air_gap_required": True},
                )

    def _enforce_audit_trail(self, context: ZoneContext, policy: ZonePolicy) -> None:
        """Enforce audit trail requirement.

        This method logs the operation; actual audit storage is handled elsewhere.
        """
        if policy.require_audit_trail:
            # Audit entry will be logged in _log_transition
            pass

    def _log_transition(self, context: ZoneContext, event: str) -> None:
        """Log a zone transition.

        Args:
            context: Zone context
            event: Event description
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "zone": context.zone.value,
            "operation": context.operation_type,
            "actor": context.actor_id,
            "event": event,
            "metadata": context.metadata,
        }
        # Compute entry hash for integrity
        entry["entry_hash"] = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()
        self._transition_log.append(entry)

    def validate_zone_transition(
        self,
        from_zone: SecurityZone,
        to_zone: SecurityZone,
        context: ZoneContext,
    ) -> bool:
        """Validate a zone transition.

        Args:
            from_zone: Source zone
            to_zone: Target zone
            context: Zone context

        Returns:
            True if transition is valid

        Raises:
            ZoneViolation: If transition is invalid
        """
        # Valid transitions:
        # Z0 -> Z1 (escalate to operational)
        # Z1 -> Z2 (escalate to sensitive, requires approval)
        # Z2 -> Z3 (escalate to critical, requires dual-control)
        # Z3 -> Z2 (de-escalate from critical)
        # Z2 -> Z1 (de-escalate from sensitive)
        # Z1 -> Z0 (de-escalate to public)
        # Same zone -> Same zone (allowed)

        from_val = int(from_zone.value[1])
        to_val = int(to_zone.value[1])

        # Same zone is always allowed
        if from_zone == to_zone:
            return True

        # Can only transition one level at a time
        if abs(to_val - from_val) > 1:
            raise ZoneViolation(
                zone=from_zone,
                invariant="zone_transition",
                message=f"Cannot skip zones: {from_zone.value} -> {to_zone.value}",
                context={"from_zone": from_zone.value, "to_zone": to_zone.value},
            )

        # Escalation to Z2 or Z3 requires dual-control
        if to_val >= 2 and not context.has_dual_control():
            raise ZoneViolation(
                zone=to_zone,
                invariant="zone_escalation",
                message=f"Dual-control required for escalation to {to_zone.value}",
                context={
                    "from_zone": from_zone.value,
                    "to_zone": to_zone.value,
                    "has_dual_control": False,
                },
            )

        self._log_transition(context, f"zone_transition:{from_zone.value}->{to_zone.value}")
        return True

    def execute_in_zone(
        self,
        context: ZoneContext,
        operation: Callable[[], Any],
    ) -> Any:
        """Execute an operation within zone constraints.

        Args:
            context: Zone execution context
            operation: The operation to execute

        Returns:
            Operation result

        Raises:
            ZoneViolation: If zone invariants are violated
        """
        # Enforce all invariants before execution
        self.enforce_zone_invariants(context)

        # Execute operation
        try:
            result = operation()
            self._log_transition(context, "operation_completed")
            return result
        except Exception as e:
            self._log_transition(context, f"operation_failed:{type(e).__name__}")
            raise

    def get_transition_log(self) -> list[dict[str, Any]]:
        """Get the zone transition log.

        Returns:
            List of transition log entries
        """
        return self._transition_log.copy()

    def verify_log_integrity(self) -> bool:
        """Verify integrity of transition log.

        Returns:
            True if log is valid
        """
        for entry in self._transition_log:
            stored_hash = entry.get("entry_hash", "")
            # Recompute hash
            entry_copy = {k: v for k, v in entry.items() if k != "entry_hash"}
            computed_hash = hashlib.sha256(
                json.dumps(entry_copy, sort_keys=True).encode()
            ).hexdigest()
            if stored_hash != computed_hash:
                return False
        return True

    def get_stats(self) -> dict[str, Any]:
        """Get enforcer statistics.

        Returns:
            Statistics dictionary
        """
        zone_counts = {zone.value: 0 for zone in SecurityZone}
        for entry in self._transition_log:
            zone = entry.get("zone", "unknown")
            if zone in zone_counts:
                zone_counts[zone] += 1

        return {
            "total_transitions": len(self._transition_log),
            "violation_count": self._violation_count,
            "transitions_by_zone": zone_counts,
            "log_integrity_valid": self.verify_log_integrity(),
        }


# Global enforcer instance (singleton pattern)
_global_enforcer: ZoneDeterminismEnforcer | None = None


def get_zone_enforcer() -> ZoneDeterminismEnforcer:
    """Get the global zone enforcer instance.

    Returns:
        ZoneDeterminismEnforcer instance
    """
    global _global_enforcer
    if _global_enforcer is None:
        _global_enforcer = ZoneDeterminismEnforcer()
    return _global_enforcer


def enforce_zone(
    zone: SecurityZone,
    operation_type: str,
    actor_id: str,
    approvers: list[str] | None = None,
    external_network: bool = False,
) -> ZoneContext:
    """Convenience function to enforce zone invariants.

    Args:
        zone: Security zone
        operation_type: Type of operation
        actor_id: Actor ID
        approvers: Optional list of approvers
        external_network: Whether external network is needed

    Returns:
        ZoneContext if enforcement passes

    Raises:
        ZoneViolation: If enforcement fails
    """
    enforcer = get_zone_enforcer()
    context = ZoneContext(
        zone=zone,
        operation_type=operation_type,
        actor_id=actor_id,
        approvers=approvers or [],
        external_network_requested=external_network,
    )
    enforcer.enforce_zone_invariants(context)
    return context
