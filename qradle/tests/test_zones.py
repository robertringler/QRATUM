"""Tests for Zone Determinism Enforcement."""

import pytest

from qradle.core.zones import (
    DEFAULT_ZONE_POLICIES,
    SecurityZone,
    ZoneContext,
    ZoneDeterminismEnforcer,
    ZoneViolation,
    enforce_zone,
)


class TestSecurityZones:
    """Test suite for security zones."""

    def test_zone_values(self):
        """Test zone enum values."""
        assert SecurityZone.Z0.value == "Z0"
        assert SecurityZone.Z1.value == "Z1"
        assert SecurityZone.Z2.value == "Z2"
        assert SecurityZone.Z3.value == "Z3"

    def test_default_policies_exist(self):
        """Test that default policies exist for all zones."""
        for zone in SecurityZone:
            assert zone in DEFAULT_ZONE_POLICIES
            policy = DEFAULT_ZONE_POLICIES[zone]
            assert policy.zone == zone


class TestZoneContext:
    """Test suite for zone context."""

    def test_context_creation(self):
        """Test creating a zone context."""
        context = ZoneContext(
            zone=SecurityZone.Z0,
            operation_type="read",
            actor_id="user_001",
        )
        assert context.zone == SecurityZone.Z0
        assert context.operation_type == "read"
        assert context.actor_id == "user_001"

    def test_dual_control_without_approvers(self):
        """Test dual control check without approvers."""
        context = ZoneContext(
            zone=SecurityZone.Z2,
            operation_type="execute",
            actor_id="user_001",
            approvers=[],
        )
        assert context.has_dual_control() is False

    def test_dual_control_with_self_approval(self):
        """Test dual control doesn't count self-approval."""
        context = ZoneContext(
            zone=SecurityZone.Z2,
            operation_type="execute",
            actor_id="user_001",
            approvers=["user_001"],  # Self-approval
        )
        assert context.has_dual_control() is False

    def test_dual_control_with_proper_approver(self):
        """Test dual control with proper approver."""
        context = ZoneContext(
            zone=SecurityZone.Z2,
            operation_type="execute",
            actor_id="user_001",
            approvers=["user_002"],
        )
        assert context.has_dual_control() is True


class TestZoneDeterminismEnforcer:
    """Test suite for zone determinism enforcer."""

    def test_enforcer_initialization(self):
        """Test enforcer initializes correctly."""
        enforcer = ZoneDeterminismEnforcer()
        assert enforcer is not None
        assert len(enforcer.policies) == 4  # Z0-Z3

    def test_z0_read_allowed(self):
        """Test Z0 allows read operations."""
        enforcer = ZoneDeterminismEnforcer()
        context = ZoneContext(
            zone=SecurityZone.Z0,
            operation_type="read",
            actor_id="user_001",
        )
        # Should not raise
        enforcer.enforce_zone_invariants(context)

    def test_z0_create_not_allowed(self):
        """Test Z0 does not allow create operations."""
        enforcer = ZoneDeterminismEnforcer()
        context = ZoneContext(
            zone=SecurityZone.Z0,
            operation_type="create",
            actor_id="user_001",
        )
        with pytest.raises(ZoneViolation) as exc_info:
            enforcer.enforce_zone_invariants(context)
        assert "operation_allowed" in str(exc_info.value)

    def test_z1_create_allowed(self):
        """Test Z1 allows create operations."""
        enforcer = ZoneDeterminismEnforcer()
        context = ZoneContext(
            zone=SecurityZone.Z1,
            operation_type="create",
            actor_id="user_001",
        )
        # Should not raise
        enforcer.enforce_zone_invariants(context)

    def test_z2_requires_dual_control(self):
        """Test Z2 requires dual control."""
        enforcer = ZoneDeterminismEnforcer()
        context = ZoneContext(
            zone=SecurityZone.Z2,
            operation_type="read",
            actor_id="user_001",
            approvers=[],  # No approvers
        )
        with pytest.raises(ZoneViolation) as exc_info:
            enforcer.enforce_zone_invariants(context)
        assert "dual_control" in str(exc_info.value)

    def test_z2_with_dual_control(self):
        """Test Z2 passes with dual control."""
        enforcer = ZoneDeterminismEnforcer()
        context = ZoneContext(
            zone=SecurityZone.Z2,
            operation_type="read",
            actor_id="user_001",
            approvers=["user_002"],
        )
        # Should not raise
        enforcer.enforce_zone_invariants(context)

    def test_z3_rejects_external_network(self):
        """Test Z3 rejects external network requests."""
        enforcer = ZoneDeterminismEnforcer()
        context = ZoneContext(
            zone=SecurityZone.Z3,
            operation_type="read",
            actor_id="user_001",
            approvers=["user_002"],
            external_network_requested=True,
        )
        with pytest.raises(ZoneViolation) as exc_info:
            enforcer.enforce_zone_invariants(context)
        assert "air_gap" in str(exc_info.value) or "network" in str(exc_info.value)

    def test_zone_transition_same_zone(self):
        """Test same-zone transition is allowed."""
        enforcer = ZoneDeterminismEnforcer()
        context = ZoneContext(
            zone=SecurityZone.Z1,
            operation_type="read",
            actor_id="user_001",
        )
        result = enforcer.validate_zone_transition(SecurityZone.Z1, SecurityZone.Z1, context)
        assert result is True

    def test_zone_transition_skip_zones(self):
        """Test cannot skip zones in transition."""
        enforcer = ZoneDeterminismEnforcer()
        context = ZoneContext(
            zone=SecurityZone.Z0,
            operation_type="read",
            actor_id="user_001",
        )
        with pytest.raises(ZoneViolation) as exc_info:
            enforcer.validate_zone_transition(SecurityZone.Z0, SecurityZone.Z2, context)
        assert "skip zones" in str(exc_info.value)

    def test_zone_escalation_requires_dual_control(self):
        """Test escalation to Z2 requires dual control."""
        enforcer = ZoneDeterminismEnforcer()
        context = ZoneContext(
            zone=SecurityZone.Z1,
            operation_type="read",
            actor_id="user_001",
            approvers=[],  # No approvers
        )
        with pytest.raises(ZoneViolation) as exc_info:
            enforcer.validate_zone_transition(SecurityZone.Z1, SecurityZone.Z2, context)
        assert "escalation" in str(exc_info.value)

    def test_execute_in_zone(self):
        """Test executing operation in zone."""
        enforcer = ZoneDeterminismEnforcer()
        context = ZoneContext(
            zone=SecurityZone.Z0,
            operation_type="read",
            actor_id="user_001",
        )

        result = enforcer.execute_in_zone(context, lambda: 42)
        assert result == 42

    def test_transition_log(self):
        """Test transition log is recorded."""
        enforcer = ZoneDeterminismEnforcer()
        context = ZoneContext(
            zone=SecurityZone.Z0,
            operation_type="read",
            actor_id="user_001",
        )
        enforcer.enforce_zone_invariants(context)

        log = enforcer.get_transition_log()
        assert len(log) > 0
        assert log[-1]["zone"] == "Z0"
        assert log[-1]["operation"] == "read"

    def test_log_integrity_verification(self):
        """Test log integrity verification."""
        enforcer = ZoneDeterminismEnforcer()
        context = ZoneContext(
            zone=SecurityZone.Z0,
            operation_type="read",
            actor_id="user_001",
        )
        enforcer.enforce_zone_invariants(context)

        assert enforcer.verify_log_integrity() is True

    def test_enforcer_stats(self):
        """Test enforcer statistics."""
        enforcer = ZoneDeterminismEnforcer()
        context = ZoneContext(
            zone=SecurityZone.Z0,
            operation_type="read",
            actor_id="user_001",
        )
        enforcer.enforce_zone_invariants(context)

        stats = enforcer.get_stats()
        assert stats["total_transitions"] > 0
        assert stats["violation_count"] == 0
        assert stats["log_integrity_valid"] is True


class TestEnforceZoneConvenience:
    """Test suite for enforce_zone convenience function."""

    def test_enforce_zone_success(self):
        """Test enforce_zone convenience function succeeds."""
        context = enforce_zone(
            zone=SecurityZone.Z0,
            operation_type="read",
            actor_id="user_001",
        )
        assert context.zone == SecurityZone.Z0

    def test_enforce_zone_failure(self):
        """Test enforce_zone convenience function fails correctly."""
        with pytest.raises(ZoneViolation):
            enforce_zone(
                zone=SecurityZone.Z0,
                operation_type="delete",  # Not allowed in Z0
                actor_id="user_001",
            )
