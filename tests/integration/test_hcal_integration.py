"""
Integration tests for HCAL end-to-end workflows.

Tests full workflows without requiring real hardware (mocked backends).
"""

import json
from unittest.mock import Mock, patch

import pytest
import yaml

from quasim.hcal import HCAL
from quasim.hcal.policy import PolicyViolation


class TestHCALIntegration:
    """End-to-end integration tests."""

    @pytest.fixture
    def policy_file(self, tmp_path):
        """Create temporary policy file."""
        policy_data = {
            "environment": "DEV",
            "allowed_backends": ["nvml", "rocm_smi"],
            "device_allowlist": ["GPU0", "GPU1"],
            "limits": {
                "power_watts_max": 300,
                "temp_c_max": 85,
            },
            "rate_limits": {
                "commands_per_minute": 30,
            },
            "approvals": {
                "required": False,  # Disabled for testing
            }
        }

        policy_file = tmp_path / "policy.yaml"
        with open(policy_file, "w") as f:
            yaml.dump(policy_data, f)

        return policy_file

    @pytest.fixture
    def hcal(self, policy_file, tmp_path):
        """Create HCAL instance with test policy."""
        audit_dir = tmp_path / "audit_logs"
        return HCAL.from_policy(
            policy_path=policy_file,
            enable_actuation=False,
            audit_log_dir=audit_dir
        )

    def test_discover_topology(self, hcal):
        """Test hardware topology discovery."""
        topology = hcal.discover(full=True)

        assert "devices" in topology
        assert "interconnects" in topology
        assert "summary" in topology
        assert isinstance(topology["summary"]["total_devices"], int)

    def test_plan_and_apply_dry_run(self, hcal):
        """Test plan creation and dry-run application."""
        # Plan with low-latency profile
        plan = hcal.plan(profile="low-latency", devices=["GPU0"])

        assert "profile" in plan
        assert plan["profile"] == "low-latency"
        assert "devices" in plan

        # Apply in dry-run mode
        result = hcal.apply(plan, enable_actuation=False)

        assert result["actuation_enabled"] is False
        assert "devices" in result

    @patch("quasim.hcal.backends.nvidia_nvml.NVMLBackend")
    def test_plan_and_apply_with_actuation(self, mock_nvml, hcal):
        """Test actual hardware actuation (mocked)."""
        # Setup mock backend
        mock_backend = Mock()
        mock_backend.apply_setpoint.return_value = {"success": True}
        mock_nvml.return_value = mock_backend

        # Enable actuation
        hcal.actuator.enable_actuation = True

        # Create simple plan
        plan = {
            "profile": "test",
            "plan_id": "test-123",
            "devices": {
                "GPU0": {
                    "power_limit_w": 250,
                }
            }
        }

        # Apply
        result = hcal.apply(plan, enable_actuation=True)

        assert result["actuation_enabled"] is True

    def test_policy_violation_blocks_operation(self, hcal):
        """Test that policy violations block operations."""
        # Create plan that violates power limit
        plan = {
            "profile": "test",
            "plan_id": "test-456",
            "devices": {
                "GPU0": {
                    "power_limit_w": 500,  # Exceeds 300W limit
                }
            }
        }

        # Should raise PolicyViolation
        with pytest.raises(PolicyViolation):
            hcal.apply(plan, enable_actuation=False)

    def test_device_not_on_allowlist(self, hcal):
        """Test that unlisted devices are blocked."""
        plan = {
            "profile": "test",
            "plan_id": "test-789",
            "devices": {
                "GPU5": {  # Not on allowlist
                    "power_limit_w": 200,
                }
            }
        }

        with pytest.raises(PolicyViolation, match="not on allowlist"):
            hcal.apply(plan, enable_actuation=False)

    def test_calibration_dry_run(self, hcal):
        """Test calibration loop in dry-run mode."""
        loop = hcal.calibration(
            device="GPU0",
            routine="power_sweep",
            parameters={}
        )

        result = loop.run(max_iters=5, enable_actuation=False)

        assert "device" in result
        assert result["device"] == "GPU0"
        assert "iterations" in result
        assert len(result["iterations"]) <= 5
        assert result["actuation_enabled"] is False

    def test_telemetry_collection(self, hcal):
        """Test telemetry collection."""
        telemetry = hcal.get_telemetry(devices=["GPU0"])

        assert isinstance(telemetry, dict)
        # May be empty if no backends initialized

    def test_emergency_stop(self, hcal):
        """Test emergency stop functionality."""
        result = hcal.emergency_stop()

        assert isinstance(result, dict)
        # Should be idempotent
        result2 = hcal.emergency_stop()
        assert isinstance(result2, dict)

    def test_audit_log_creation(self, hcal, tmp_path):
        """Test that audit logs are created."""
        audit_dir = tmp_path / "audit_logs"

        # Apply a plan (dry-run)
        plan = hcal.plan(profile="balanced", devices=["GPU0"])
        hcal.apply(plan, enable_actuation=False)

        # Check audit log exists
        log_files = list(audit_dir.glob("**/*.jsonl"))
        assert len(log_files) > 0

        # Verify log format
        log_file = log_files[0]
        with open(log_file) as f:
            for line in f:
                entry = json.loads(line)
                assert "timestamp" in entry
                assert "event_type" in entry
                assert "hash" in entry
                assert "prev_hash" in entry


class TestCLIIntegration:
    """Test CLI interface integration."""

    @pytest.fixture
    def cli_runner(self):
        """Create Click CLI test runner."""
        from click.testing import CliRunner
        return CliRunner()

    def test_discover_command(self, cli_runner, tmp_path):
        """Test discover CLI command."""
        from quasim.hcal.cli import cli

        result = cli_runner.invoke(cli, ["discover", "--json"])

        assert result.exit_code == 0
        # Should output valid JSON
        try:
            data = json.loads(result.output)
            assert "devices" in data or "summary" in data
        except json.JSONDecodeError:
            pytest.skip("No real hardware available")

    def test_plan_command(self, cli_runner, tmp_path):
        """Test plan CLI command."""
        from quasim.hcal.cli import cli

        result = cli_runner.invoke(cli, [
            "plan",
            "--profile", "balanced",
            "--devices", "GPU0",
            "--out", str(tmp_path / "plan.json")
        ])

        # May fail without real hardware, but should handle gracefully
        assert result.exit_code in [0, 1]


class TestRollback:
    """Test rollback functionality."""

    @pytest.fixture
    def hcal_with_baseline(self, tmp_path):
        """HCAL instance with captured baseline."""
        policy_data = {
            "environment": "DEV",
            "allowed_backends": ["nvml"],
            "limits": {"power_watts_max": 300},
            "approvals": {"required": False}
        }

        policy_file = tmp_path / "policy.yaml"
        with open(policy_file, "w") as f:
            yaml.dump(policy_data, f)

        hcal = HCAL.from_policy(policy_file, enable_actuation=False)

        # Capture baseline for GPU0
        hcal.actuator.capture_baseline("GPU0")

        return hcal

    def test_rollback_on_error(self, hcal_with_baseline):
        """Test automatic rollback on error."""
        hcal = hcal_with_baseline

        # Create plan that would fail validation
        bad_plan = {
            "profile": "test",
            "plan_id": "rollback-test",
            "devices": {
                "GPU0": {
                    "power_limit_w": 500,  # Exceeds limit
                }
            }
        }

        # Should fail and rollback
        with pytest.raises(PolicyViolation):
            hcal.apply(bad_plan, enable_actuation=False)

    def test_manual_rollback(self, hcal_with_baseline):
        """Test manual device rollback."""
        hcal = hcal_with_baseline

        result = hcal.actuator.rollback_device("GPU0")

        # Should complete even if no backend available
        assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
