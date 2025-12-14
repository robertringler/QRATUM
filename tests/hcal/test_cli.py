"""Tests for CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from click.testing import CliRunner

from quasim.hcal.cli import cli


class TestCLI:
    """Tests for HCAL CLI."""

    def test_cli_help(self) -> None:
        """Test CLI help command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "Hardware Calibration and Analysis Layer" in result.output

    def test_discover_command(self) -> None:
        """Test discover command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["discover"])

        assert result.exit_code == 0
        assert "Discovered" in result.output
        assert "device" in result.output

    def test_discover_command_json(self) -> None:
        """Test discover command with JSON output."""
        runner = CliRunner()
        result = runner.invoke(cli, ["discover", "--json"])

        assert result.exit_code == 0

        # Parse JSON output
        data = json.loads(result.output)
        assert isinstance(data, dict)
        assert "devices" in data or "summary" in data

    def test_validate_policy_command(self, tmp_path: Path) -> None:
        """Test validate-policy command with valid policy."""
        policy_data = {
            "environment": "DEV",
            "allowed_backends": ["cpu", "cuda"],
            "limits": {"max_qubits": 20},
        }

        policy_file = tmp_path / "policy.yaml"
        with open(policy_file, "w") as f:
            yaml.dump(policy_data, f)

        runner = CliRunner()
        result = runner.invoke(cli, ["validate-policy", str(policy_file)])

        assert result.exit_code == 0
        assert "✓ Policy validation passed" in result.output
        assert "Environment: DEV" in result.output

    def test_validate_policy_command_invalid(self, tmp_path: Path) -> None:
        """Test validate-policy command with invalid policy."""
        policy_data = {
            "environment": "INVALID",
            "allowed_backends": ["cpu"],
            "limits": {},
        }

        policy_file = tmp_path / "policy.yaml"
        with open(policy_file, "w") as f:
            yaml.dump(policy_data, f)

        runner = CliRunner()
        result = runner.invoke(cli, ["validate-policy", str(policy_file)])

        assert result.exit_code == 1
        assert "✗ Policy validation failed" in result.output

    def test_validate_policy_command_missing_file(self) -> None:
        """Test validate-policy command with missing file."""
        runner = CliRunner()
        result = runner.invoke(cli, ["validate-policy", "/nonexistent/policy.yaml"])

        # Click will catch the file not existing before our code runs
        assert result.exit_code != 0
