"""Tests for HCAL policy engine."""

import tempfile
from pathlib import Path

import pytest
import yaml

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
