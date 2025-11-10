"""Additional tests for HCAL methods to increase coverage."""

import pytest
import yaml

from quasim.hcal import HCAL
from quasim.hcal.policy import PolicyEngine, PolicyViolation


class TestHCALDiscoverPlanApply:
    """Test HCAL discover, plan, and apply methods."""

    def test_discover_without_policy(self):
        """Test discover without a policy file."""
        hcal = HCAL(dry_run=True)
        topology = hcal.discover(full=False)

        assert "devices" in topology
        assert "summary" in topology
        assert isinstance(topology["summary"]["total_devices"], int)

    def test_discover_full(self):
        """Test full discovery."""
        hcal = HCAL(dry_run=True)
        topology = hcal.discover(full=True)

        assert "devices" in topology
        assert "interconnects" in topology
        assert "summary" in topology

    def test_plan_without_devices(self):
        """Test plan creation without specifying devices."""
        hcal = HCAL(dry_run=True)
        plan = hcal.plan(profile="balanced")

        assert "plan_id" in plan
        assert "profile" in plan
        assert plan["profile"] == "balanced"
        assert "devices" in plan

    def test_plan_with_unknown_profile(self):
        """Test plan with an unknown profile."""
        hcal = HCAL(dry_run=True)
        plan = hcal.plan(profile="unknown-profile")

        # Should still create a plan, just with empty setpoints
        assert "plan_id" in plan
        assert "profile" in plan

    def test_apply_with_empty_plan(self):
        """Test applying an empty plan."""
        hcal = HCAL(dry_run=True)
        plan = {"plan_id": "test", "devices": {}}
        result = hcal.apply(plan)

        assert "plan_id" in result
        assert result["actuation_enabled"] is False

    def test_apply_with_actuation_override(self):
        """Test applying plan with actuation override."""
        hcal = HCAL(dry_run=True)
        plan = {"plan_id": "test", "devices": {"GPU0": {"power_limit_w": 200}}}

        # Override to enable actuation
        result = hcal.apply(plan, enable_actuation=True)

        assert result["actuation_enabled"] is True

    def test_discover_topology(self):
        """Test discover_topology method."""
        hcal = HCAL(dry_run=True)
        topology = hcal.discover_topology()

        assert topology is not None
        assert len(topology.devices) > 0


class TestPolicyEngineAdditional:
    """Additional PolicyEngine tests."""

    def test_policy_engine_default(self):
        """Test PolicyEngine with default settings."""
        engine = PolicyEngine()

        assert engine.environment is not None
        assert engine.requires_approval()
        assert engine.is_dry_run_default()

    def test_policy_engine_custom_config(self, tmp_path):
        """Test PolicyEngine with custom configuration."""
        policy_data = {
            "environment": "LAB",
            "allowed_backends": ["nvml"],
            "device_allowlist": ["GPU0"],
            "limits": {"power_watts_max": 250},
            "rate_limits": {"commands_per_minute": 30},
            "approvals": {"required": False},
            "dry_run_default": False,
        }

        policy_file = tmp_path / "policy.yaml"
        with open(policy_file, "w") as f:
            yaml.dump(policy_data, f)

        engine = PolicyEngine(policy_file)

        assert engine.environment.value == "lab"  # Environment enum uses lowercase
        assert not engine.requires_approval()
        assert not engine.is_dry_run_default()

    def test_is_backend_allowed(self):
        """Test backend validation."""
        engine = PolicyEngine()

        assert engine.is_backend_allowed("nvml")
        assert engine.is_backend_allowed("rocm_smi")

    def test_is_device_allowed_with_allowlist(self, tmp_path):
        """Test device allowlist enforcement."""
        policy_data = {
            "environment": "DEV",
            "allowed_backends": ["nvml"],
            "device_allowlist": ["GPU0", "GPU1"],
            "limits": {},
            "rate_limits": {},
            "approvals": {"required": False},
        }

        policy_file = tmp_path / "policy.yaml"
        with open(policy_file, "w") as f:
            yaml.dump(policy_data, f)

        engine = PolicyEngine(policy_file)

        assert engine.is_device_allowed("GPU0")
        assert engine.is_device_allowed("GPU1")
        assert not engine.is_device_allowed("GPU2")

    def test_rate_limiter_reset(self):
        """Test rate limiter over time."""
        import time

        engine = PolicyEngine()
        engine.rate_limiter.commands_per_minute = 2
        engine.rate_limiter.window_seconds = 1

        # First 2 should pass
        engine.rate_limiter.check_and_record()
        engine.rate_limiter.check_and_record()

        # 3rd should fail
        with pytest.raises(PolicyViolation):
            engine.rate_limiter.check_and_record()

        # Wait for window to expire
        time.sleep(1.1)

        # Should pass again
        engine.rate_limiter.check_and_record()

    def test_validate_operation_with_invalid_device(self, tmp_path):
        """Test operation validation with device not on allowlist."""
        policy_data = {
            "environment": "DEV",
            "allowed_backends": ["nvml"],
            "device_allowlist": ["GPU0"],
            "limits": {"power_watts_max": 300},
            "rate_limits": {},
            "approvals": {"required": False},
        }

        policy_file = tmp_path / "policy.yaml"
        with open(policy_file, "w") as f:
            yaml.dump(policy_data, f)

        engine = PolicyEngine(policy_file)

        # Should raise for device not in allowlist
        with pytest.raises(PolicyViolation, match="not in allowlist"):
            engine.validate_operation(
                device_id="GPU5",
                operation="reconfigure",
                setpoints={"power_limit_w": 200},
                enable_actuation=False,
            )


class TestHCALWithPolicyFile:
    """Test HCAL with policy files."""

    def test_hcal_from_policy_with_audit(self, tmp_path):
        """Test HCAL.from_policy with audit logging."""
        policy_data = {
            "environment": "DEV",
            "allowed_backends": ["nvml"],
            "device_allowlist": ["GPU0"],
            "limits": {"power_watts_max": 300},
            "rate_limits": {},
            "approvals": {"required": False},
        }

        policy_file = tmp_path / "policy.yaml"
        with open(policy_file, "w") as f:
            yaml.dump(policy_data, f)

        audit_dir = tmp_path / "audit"
        hcal = HCAL.from_policy(
            policy_path=policy_file,
            enable_actuation=False,
            audit_log_dir=audit_dir,
        )

        assert hcal.dry_run is True
        assert audit_dir.exists()

    def test_plan_with_device_filtering(self, tmp_path):
        """Test plan creation with device filtering."""
        policy_data = {
            "environment": "DEV",
            "allowed_backends": ["nvml"],
            "device_allowlist": ["GPU0"],
            "limits": {},
            "rate_limits": {},
            "approvals": {"required": False},
        }

        policy_file = tmp_path / "policy.yaml"
        with open(policy_file, "w") as f:
            yaml.dump(policy_data, f)

        hcal = HCAL.from_policy(policy_file)

        # Try to plan for both allowed and disallowed devices
        plan = hcal.plan(profile="balanced", devices=["GPU0", "GPU1"])

        # GPU1 should be filtered out
        assert "GPU0" in plan["devices"]
        if "warnings" in plan:
            assert "GPU1" in plan["warnings"]["filtered_devices"]

    def test_get_audit_log(self):
        """Test getting audit log."""
        hcal = HCAL(dry_run=True)
        log = hcal.get_audit_log()

        assert isinstance(log, list)

    def test_verify_audit_chain(self):
        """Test audit chain verification."""
        hcal = HCAL(dry_run=True)
        valid = hcal.verify_audit_chain()

        assert isinstance(valid, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
