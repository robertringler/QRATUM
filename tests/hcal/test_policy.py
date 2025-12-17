"""Tests for policy validation."""

from __future__ import annotations

"""Tests for HCAL policy engine."""

import tempfile
from pathlib import Path

import pytest
import yaml

from quasim.hcal.policy import PolicyConfig, PolicyValidator


class TestPolicyConfig:
    """Tests for PolicyConfig class."""

    def test_policy_config_creation(self) -> None:
        """Test creating a policy config."""

        config = PolicyConfig(
            environment="DEV",
            allowed_backends=["cpu", "cuda"],
            limits={"max_qubits": 20},
        )

        assert config.environment == "DEV"
        assert config.allowed_backends == ["cpu", "cuda"]
        assert config.limits == {"max_qubits": 20}


class TestPolicyValidator:
    """Tests for PolicyValidator class."""

    def test_validator_init(self) -> None:
        """Test validator initialization."""

        validator = PolicyValidator()
        assert validator.policy is None

    def test_validator_with_config(self) -> None:
        """Test validator with configuration."""

        config = PolicyConfig(
            environment="LAB",
            allowed_backends=["cpu"],
            limits={"max_qubits": 10},
        )
        validator = PolicyValidator(config)
        assert validator.policy == config

    def test_from_file_valid(self, tmp_path: Path) -> None:
        """Test loading valid policy from file."""

        policy_data = {
            "environment": "DEV",
            "allowed_backends": ["cpu", "cuda"],
            "limits": {"max_qubits": 20, "max_circuits": 100},
        }

        policy_file = tmp_path / "policy.yaml"
        with open(policy_file, "w") as f:
            yaml.dump(policy_data, f)

        validator = PolicyValidator.from_file(policy_file)
        assert validator.policy.environment == "DEV"
        assert "cpu" in validator.policy.allowed_backends
        assert validator.policy.limits["max_qubits"] == 20

    def test_from_file_missing_file(self, tmp_path: Path) -> None:
        """Test loading from nonexistent file."""

        policy_file = tmp_path / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError):
            PolicyValidator.from_file(policy_file)

    def test_from_file_invalid_environment(self, tmp_path: Path) -> None:
        """Test loading policy with invalid environment."""

        policy_data = {
            "environment": "INVALID",
            "allowed_backends": ["cpu"],
            "limits": {"max_qubits": 10},
        }

        policy_file = tmp_path / "policy.yaml"
        with open(policy_file, "w") as f:
            yaml.dump(policy_data, f)

        with pytest.raises(ValueError, match="Invalid environment"):
            PolicyValidator.from_file(policy_file)

    def test_from_file_missing_keys(self, tmp_path: Path) -> None:
        """Test loading policy with missing required keys."""

        policy_data = {
            "environment": "DEV",
            # missing allowed_backends and limits
        }

        policy_file = tmp_path / "policy.yaml"
        with open(policy_file, "w") as f:
            yaml.dump(policy_data, f)

        with pytest.raises(ValueError, match="Missing required policy key"):
            PolicyValidator.from_file(policy_file)

    def test_validate_backend_allowed(self) -> None:
        """Test validating allowed backend."""

        config = PolicyConfig(
            environment="DEV",
            allowed_backends=["cpu", "cuda"],
            limits={},
        )
        validator = PolicyValidator(config)

        assert validator.validate_backend("cpu") is True
        assert validator.validate_backend("cuda") is True

    def test_validate_backend_not_allowed(self) -> None:
        """Test validating disallowed backend."""

        config = PolicyConfig(
            environment="PROD",
            allowed_backends=["cpu"],
            limits={},
        )
        validator = PolicyValidator(config)

        assert validator.validate_backend("cuda") is False

    def test_validate_backend_no_policy(self) -> None:
        """Test validating backend with no policy."""

        validator = PolicyValidator()
        assert validator.validate_backend("any") is True

    def test_check_limits_within(self) -> None:
        """Test checking limits within bounds."""

        config = PolicyConfig(
            environment="DEV",
            allowed_backends=["cpu"],
            limits={"max_qubits": 20},
        )
        validator = PolicyValidator(config)

        assert validator.check_limits("max_qubits", 10) is True
        assert validator.check_limits("max_qubits", 20) is True

    def test_check_limits_exceeded(self) -> None:
        """Test checking limits when exceeded."""

        config = PolicyConfig(
            environment="PROD",
            allowed_backends=["cpu"],
            limits={"max_qubits": 10},
        )
        validator = PolicyValidator(config)

        assert validator.check_limits("max_qubits", 20) is False

    def test_check_limits_unknown_resource(self) -> None:
        """Test checking limits for unknown resource."""

        config = PolicyConfig(
            environment="DEV",
            allowed_backends=["cpu"],
            limits={"max_qubits": 20},
        )
        validator = PolicyValidator(config)

        # Unknown resources are allowed by default
        assert validator.check_limits("unknown_resource", 1000) is True

    def test_check_limits_no_policy(self) -> None:
        """Test checking limits with no policy."""

        validator = PolicyValidator()
        assert validator.check_limits("any", 1000) is True


from quasim.hcal.policy import Environment, PolicyEngine


def test_default_policy():
    """Test default policy initialization."""

    engine = PolicyEngine()

    assert engine.policy is not None
    assert engine.policy.environment == Environment.DEV
    assert engine.policy.dry_run_default is True
    assert engine.check_device_allowed("gpu0")


def test_load_valid_policy():
    """Test loading valid policy from file."""

    policy_data = {
        "environment": "lab",
        "device_allowlist": ["gpu0", "gpu1"],
        "backend_restrictions": ["AmdRocmBackend"],
        "device_limits": {
            "gpu0": {
                "max_power_watts": 300.0,
                "max_temp_celsius": 85.0,
            }
        },
        "rate_limit": {
            "commands_per_minute": 60,
        },
        "approval_gate": {
            "required": True,
            "min_approvers": 2,
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(policy_data, f)
        policy_path = Path(f.name)

    try:
        engine = PolicyEngine(policy_path)

        assert engine.policy.environment == Environment.LAB
        assert "gpu0" in engine.policy.device_allowlist
        assert "AmdRocmBackend" in engine.policy.backend_restrictions
        assert engine.policy.device_limits["gpu0"].max_power_watts == 300.0
        assert engine.policy.rate_limit.commands_per_minute == 60
        assert engine.policy.approval_gate.required is True

    finally:
        policy_path.unlink()


def test_invalid_policy_schema():
    """Test that invalid policy schema raises error."""

    policy_data = {
        "environment": "lab",
        # Missing required field: device_allowlist
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(policy_data, f)
        policy_path = Path(f.name)

    try:
        with pytest.raises(ValueError, match="Missing required field"):
            PolicyEngine(policy_path)
    finally:
        policy_path.unlink()


def test_invalid_environment():
    """Test that invalid environment raises error."""

    policy_data = {
        "environment": "invalid",
        "device_allowlist": ["*"],
        "backend_restrictions": [],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(policy_data, f)
        policy_path = Path(f.name)

    try:
        with pytest.raises(ValueError, match="Invalid environment"):
            PolicyEngine(policy_path)
    finally:
        policy_path.unlink()


def test_device_allowlist():
    """Test device allowlist enforcement."""

    policy_data = {
        "environment": "lab",
        "device_allowlist": ["gpu0", "gpu1"],
        "backend_restrictions": [],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(policy_data, f)
        policy_path = Path(f.name)

    try:
        engine = PolicyEngine(policy_path)

        assert engine.check_device_allowed("gpu0") is True
        assert engine.check_device_allowed("gpu1") is True
        assert engine.check_device_allowed("gpu2") is False
        assert engine.check_device_allowed("cpu0") is False

    finally:
        policy_path.unlink()


def test_wildcard_device_allowlist():
    """Test wildcard device allowlist."""

    policy_data = {
        "environment": "dev",
        "device_allowlist": ["*"],
        "backend_restrictions": [],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(policy_data, f)
        policy_path = Path(f.name)

    try:
        engine = PolicyEngine(policy_path)

        assert engine.check_device_allowed("gpu0") is True
        assert engine.check_device_allowed("any_device") is True

    finally:
        policy_path.unlink()


def test_backend_restrictions():
    """Test backend restrictions enforcement."""

    policy_data = {
        "environment": "lab",
        "device_allowlist": ["*"],
        "backend_restrictions": ["AmdRocmBackend", "XilinxXrtBackend"],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(policy_data, f)
        policy_path = Path(f.name)

    try:
        engine = PolicyEngine(policy_path)

        assert engine.check_backend_allowed("NvidiaNvmlBackend") is True
        assert engine.check_backend_allowed("AmdRocmBackend") is False
        assert engine.check_backend_allowed("XilinxXrtBackend") is False

    finally:
        policy_path.unlink()


def test_limit_enforcement():
    """Test device limit enforcement."""

    policy_data = {
        "environment": "lab",
        "device_allowlist": ["gpu0"],
        "backend_restrictions": [],
        "device_limits": {
            "gpu0": {
                "max_power_watts": 250.0,
                "min_power_watts": 100.0,
                "max_temp_celsius": 80.0,
            }
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(policy_data, f)
        policy_path = Path(f.name)

    try:
        engine = PolicyEngine(policy_path)

        # Within limits
        assert engine.check_limits("gpu0", {"power_watts": 200.0}) is True

        # Exceeds max power
        assert engine.check_limits("gpu0", {"power_watts": 300.0}) is False

        # Below min power
        assert engine.check_limits("gpu0", {"power_watts": 50.0}) is False

        # Exceeds max temp
        assert engine.check_limits("gpu0", {"temp_celsius": 90.0}) is False

        # No limits defined for device
        assert engine.check_limits("gpu1", {"power_watts": 999.0}) is True

    finally:
        policy_path.unlink()


def test_rate_limiting():
    """Test rate limiting enforcement."""

    policy_data = {
        "environment": "dev",
        "device_allowlist": ["*"],
        "backend_restrictions": [],
        "rate_limit": {
            "commands_per_minute": 5,
            "window_seconds": 60,
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(policy_data, f)
        policy_path = Path(f.name)

    try:
        engine = PolicyEngine(policy_path)

        # First 5 checks should pass
        for _ in range(5):
            assert engine.check_rate_limit() is True

        # 6th check should fail
        assert engine.check_rate_limit() is False

    finally:
        policy_path.unlink()


def test_approval_requirements():
    """Test approval gate requirements."""

    # Approval not required
    policy_data = {
        "environment": "dev",
        "device_allowlist": ["*"],
        "backend_restrictions": [],
        "approval_gate": {
            "required": False,
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(policy_data, f)
        policy_path = Path(f.name)

    try:
        engine = PolicyEngine(policy_path)
        assert engine.check_approval("test_operation") is True
    finally:
        policy_path.unlink()

    # Approval required (should fail without actual approval)
    policy_data["approval_gate"]["required"] = True

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(policy_data, f)
        policy_path = Path(f.name)

    try:
        engine = PolicyEngine(policy_path)
        assert engine.check_approval("test_operation") is False
    finally:
        policy_path.unlink()


def test_environment_getters():
    """Test environment and dry-run getters."""

    policy_data = {
        "environment": "prod",
        "device_allowlist": ["*"],
        "backend_restrictions": [],
        "dry_run_default": True,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(policy_data, f)
        policy_path = Path(f.name)

    try:
        engine = PolicyEngine(policy_path)

        assert engine.get_environment() == Environment.PROD
        assert engine.is_dry_run_default() is True

    finally:
        policy_path.unlink()
