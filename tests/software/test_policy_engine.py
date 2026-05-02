"""

Tests for policy engine and validation.
"""

import pytest
import yaml

from quasim.hcal.policy import (DeviceLimits, Environment, PolicyEngine,
                                PolicyViolation)


class TestPolicyEngine:
    """Test policy engine functionality."""

    def test_default_policy(self):
        """Test default policy initialization."""

        policy = PolicyEngine()

        assert policy.environment == Environment.DEV
        assert "nvml" in policy.config["allowed_backends"]
        assert policy.requires_approval()

    def test_load_policy_from_file(self, tmp_path):
        """Test loading policy from YAML."""

        policy_data = {
            "environment": "LAB",
            "allowed_backends": ["nvml", "rocm_smi"],
            "device_allowlist": ["GPU0", "GPU1"],
            "limits": {
                "power_watts_max": 250,
                "temp_c_max": 80,
            },
            "rate_limits": {
                "commands_per_minute": 20,
            },
            "approvals": {
                "required": True,
                "method": "file_token",
            },
        }

        policy_file = tmp_path / "policy.yaml"
        with open(policy_file, "w") as f:
            yaml.dump(policy_data, f)

        policy = PolicyEngine(policy_file)

        assert policy.environment == Environment.LAB
        assert policy.is_backend_allowed("nvml")
        assert policy.is_backend_allowed("rocm_smi")
        assert not policy.is_backend_allowed("unknown")
        assert policy.is_device_allowed("GPU0")
        assert not policy.is_device_allowed("GPU2")

    def test_device_limits_validation(self):
        """Test device limits validation."""

        limits = DeviceLimits(
            power_watts_max=300,
            temp_c_max=85,
            voltage_mv_range=(700, 1200),
        )

        # Valid setpoints
        limits.validate_setpoint("power_watts", 250)
        limits.validate_setpoint("temp_c", 80)
        limits.validate_setpoint("voltage_mv", 1000)

        # Invalid setpoints
        with pytest.raises(PolicyViolation, match="Power setpoint.*exceeds"):
            limits.validate_setpoint("power_watts", 350)

        with pytest.raises(PolicyViolation, match="Temperature.*exceeds"):
            limits.validate_setpoint("temp_c", 90)

        with pytest.raises(PolicyViolation, match="Voltage.*outside range"):
            limits.validate_setpoint("voltage_mv", 1500)

    def test_rate_limiting(self):
        """Test rate limit enforcement."""

        policy = PolicyEngine()
        policy.rate_limiter.commands_per_minute = 3
        policy.rate_limiter.window_seconds = 10

        # First 3 commands should pass
        for _ in range(3):
            policy.rate_limiter.check_and_record()

        # 4th should fail
        with pytest.raises(PolicyViolation, match="Rate limit exceeded"):
            policy.rate_limiter.check_and_record()

    def test_operation_validation(self):
        """Test operation validation."""

        policy = PolicyEngine()

        # Valid operation
        policy.validate_operation(
            device_id="GPU0",
            operation="reconfigure",
            setpoints={"power_limit_w": 200},
            enable_actuation=False,
        )

        # Invalid: exceeds power limit
        with pytest.raises(PolicyViolation):
            policy.validate_operation(
                device_id="GPU0",
                operation="reconfigure",
                setpoints={"power_limit_w": 500},
                enable_actuation=False,
            )

    def test_prod_environment_restrictions(self, tmp_path):
        """Test production environment restrictions."""

        policy_data = {
            "environment": "PROD",
            "allowed_backends": ["nvml"],
            "limits": {"power_watts_max": 300},
            "approvals": {"required": True},
        }

        policy_file = tmp_path / "policy.yaml"
        with open(policy_file, "w") as f:
            yaml.dump(policy_data, f)

        policy = PolicyEngine(policy_file)

        # Firmware updates should be blocked in PROD
        with pytest.raises(PolicyViolation, match="not allowed in PROD"):
            policy.validate_operation(
                device_id="GPU0", operation="firmware_update", setpoints={}, enable_actuation=True
            )

    def test_approval_validation(self, tmp_path):
        """Test approval token validation."""

        token_file = tmp_path / "token.txt"
        token_file.write_text("secret-token-123")

        policy_data = {
            "environment": "PROD",
            "allowed_backends": ["nvml"],
            "limits": {},
            "approvals": {
                "required": True,
                "method": "file_token",
                "token_path": str(token_file),
            },
        }

        policy_file = tmp_path / "policy.yaml"
        with open(policy_file, "w") as f:
            yaml.dump(policy_data, f)

        policy = PolicyEngine(policy_file)

        # Valid token
        policy.validate_approval("secret-token-123")

        # Invalid token
        with pytest.raises(PolicyViolation, match="Invalid approval token"):
            policy.validate_approval("wrong-token")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
