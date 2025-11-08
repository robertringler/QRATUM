"""Tests for QuASIM HCAL CLI commands."""

import json
import tempfile
from pathlib import Path

from click.testing import CliRunner

from quasim.cli.main import main


class TestHCALCLI:
    """Test suite for HCAL CLI commands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_cli_help(self):
        """Test that CLI help is available."""
        result = self.runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Hardware Configuration and Calibration" in result.output
        assert "discover" in result.output
        assert "plan" in result.output
        assert "apply" in result.output
        assert "calibrate" in result.output
        assert "stop" in result.output

    def test_discover_json(self):
        """Test hardware discovery with JSON output."""
        result = self.runner.invoke(main, ["discover", "--json"])
        assert result.exit_code == 0

        # Parse JSON output
        data = json.loads(result.output)
        assert "devices" in data
        assert len(data["devices"]) > 0

        # Check device structure
        device = data["devices"][0]
        assert "id" in device
        assert "type" in device
        assert "status" in device

    def test_discover_text(self):
        """Test hardware discovery with text output."""
        result = self.runner.invoke(main, ["discover"])
        assert result.exit_code == 0
        assert "Hardware Discovery Report" in result.output
        assert "Device:" in result.output

    def test_plan_low_latency(self):
        """Test plan command with low-latency profile."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_file = Path(tmpdir) / "test_plan.json"
            result = self.runner.invoke(
                main,
                [
                    "plan",
                    "--profile",
                    "low-latency",
                    "--devices",
                    "GPU0,GPU1",
                    "--output",
                    str(plan_file),
                ],
            )
            assert result.exit_code == 0
            assert "Reconfiguration plan created" in result.output
            assert plan_file.exists()

            # Validate plan structure
            with open(plan_file) as f:
                plan_data = json.load(f)

            assert plan_data["profile"] == "low-latency"
            assert plan_data["target_devices"] == ["GPU0", "GPU1"]
            assert len(plan_data["actions"]) == 2
            assert plan_data["profile_config"]["priority"] == "latency"

    def test_plan_unknown_profile(self):
        """Test plan command with unknown profile."""
        result = self.runner.invoke(
            main, ["plan", "--profile", "unknown-profile", "--devices", "GPU0"]
        )
        assert result.exit_code == 1
        assert "Unknown profile" in result.output

    def test_apply_dry_run(self):
        """Test apply command in dry-run mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test plan
            plan_file = Path(tmpdir) / "test_plan.json"
            plan_data = {
                "plan_version": "1.0",
                "profile": "low-latency",
                "target_devices": ["GPU0"],
                "actions": [
                    {
                        "device": "GPU0",
                        "action": "reconfigure",
                        "changes": {"power_mode": "max_performance"},
                    }
                ],
                "estimated_downtime_seconds": 10,
            }

            with open(plan_file, "w") as f:
                json.dump(plan_data, f)

            # Test dry-run
            result = self.runner.invoke(main, ["apply", "--plan", str(plan_file)])
            assert result.exit_code == 0
            assert "DRY-RUN MODE" in result.output
            assert "Dry-run validation passed" in result.output

    def test_apply_with_actuation(self):
        """Test apply command with actuation enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test plan
            plan_file = Path(tmpdir) / "test_plan.json"
            plan_data = {
                "plan_version": "1.0",
                "profile": "low-latency",
                "target_devices": ["GPU0"],
                "actions": [
                    {
                        "device": "GPU0",
                        "action": "reconfigure",
                        "changes": {"power_mode": "max_performance"},
                    }
                ],
                "estimated_downtime_seconds": 10,
            }

            with open(plan_file, "w") as f:
                json.dump(plan_data, f)

            # Test actuation
            result = self.runner.invoke(
                main,
                [
                    "apply",
                    "--plan",
                    str(plan_file),
                    "--enable-actuation",
                    "--require-approval",
                    "token",
                ],
            )
            assert result.exit_code == 0
            assert "APPLYING RECONFIGURATION PLAN" in result.output
            assert "ACTUATION ENABLED" in result.output
            assert "completed successfully" in result.output

    def test_apply_actuation_without_approval(self):
        """Test that actuation requires approval token."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test plan
            plan_file = Path(tmpdir) / "test_plan.json"
            plan_data = {
                "plan_version": "1.0",
                "profile": "low-latency",
                "target_devices": ["GPU0"],
                "actions": [],
            }

            with open(plan_file, "w") as f:
                json.dump(plan_data, f)

            # Test actuation without approval
            result = self.runner.invoke(
                main, ["apply", "--plan", str(plan_file), "--enable-actuation"]
            )
            assert result.exit_code == 1
            assert "require-approval token required" in result.output

    def test_apply_nonexistent_plan(self):
        """Test apply command with nonexistent plan file."""
        result = self.runner.invoke(main, ["apply", "--plan", "/nonexistent/plan.json"])
        assert result.exit_code == 1
        assert "Plan file not found" in result.output

    def test_calibrate_power_sweep(self):
        """Test calibrate command with power_sweep routine."""
        result = self.runner.invoke(
            main, ["calibrate", "--device", "GPU0", "--routine", "power_sweep", "--max-iters", "10"]
        )
        assert result.exit_code == 0
        assert "Starting Hardware Calibration" in result.output
        assert "GPU0" in result.output
        assert "power_sweep" in result.output
        assert "Calibration completed successfully" in result.output

    def test_calibrate_thermal_test(self):
        """Test calibrate command with thermal_test routine."""
        result = self.runner.invoke(
            main, ["calibrate", "--device", "GPU1", "--routine", "thermal_test", "--max-iters", "5"]
        )
        assert result.exit_code == 0
        assert "GPU1" in result.output
        assert "thermal_test" in result.output

    def test_calibrate_unknown_routine_without_force(self):
        """Test calibrate command with unknown routine fails without --force."""
        result = self.runner.invoke(
            main,
            ["calibrate", "--device", "GPU0", "--routine", "unknown_routine", "--max-iters", "5"],
        )
        assert result.exit_code == 1
        assert "Unknown routine" in result.output
        assert "Available routines:" in result.output
        assert "Use --force" in result.output

    def test_calibrate_unknown_routine_with_force(self):
        """Test calibrate command with unknown routine succeeds with --force."""
        result = self.runner.invoke(
            main,
            [
                "calibrate",
                "--device",
                "GPU0",
                "--routine",
                "unknown_routine",
                "--max-iters",
                "5",
                "--force",
            ],
        )
        assert result.exit_code == 0
        assert "unknown routine" in result.output
        assert "--force" in result.output

    def test_stop_all(self):
        """Test stop command with --all flag."""
        result = self.runner.invoke(main, ["stop", "--all"])
        assert result.exit_code == 0
        assert "EMERGENCY STOP INITIATED" in result.output
        assert "Stopping ALL devices" in result.output
        assert "Emergency stop completed" in result.output

    def test_stop_specific_device(self):
        """Test stop command for specific device."""
        result = self.runner.invoke(main, ["stop", "--device", "GPU0"])
        assert result.exit_code == 0
        assert "EMERGENCY STOP INITIATED" in result.output
        assert "Stopping device: GPU0" in result.output
        assert "Stopped: GPU0" in result.output

    def test_stop_no_arguments(self):
        """Test that stop command requires --all or --device."""
        result = self.runner.invoke(main, ["stop"])
        assert result.exit_code == 1
        assert "Must specify either --all or --device" in result.output

    def test_stop_conflicting_arguments(self):
        """Test that stop command doesn't accept both --all and --device."""
        result = self.runner.invoke(main, ["stop", "--all", "--device", "GPU0"])
        assert result.exit_code == 1
        assert "Cannot specify both --all and --device" in result.output
