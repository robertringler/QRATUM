"""Tests for CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from click.testing import CliRunner

from quasim.hcal.cli import main


class TestCLI:
    """Tests for HCAL CLI."""

    def test_cli_help(self) -> None:
        """Test CLI help command."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "Hardware Control Abstraction Layer" in result.output

    def test_discover_command(self) -> None:
        """Test discover command."""
        runner = CliRunner()
        result = runner.invoke(main, ["discover"])

        assert result.exit_code == 0
        assert "Discovered" in result.output
        assert "device(s)" in result.output

    def test_discover_command_json(self) -> None:
        """Test discover command with JSON output."""
        runner = CliRunner()
        result = runner.invoke(main, ["discover", "--json"])

        assert result.exit_code == 0

        # Parse JSON output
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) >= 2

        # Verify device structure
        for device in data:
            assert "id" in device
            assert "name" in device
            assert "type" in device
            assert "status" in device

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
        result = runner.invoke(main, ["validate-policy", str(policy_file)])

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
        result = runner.invoke(main, ["validate-policy", str(policy_file)])

        assert result.exit_code == 1
        assert "✗ Policy validation failed" in result.output

    def test_validate_policy_command_missing_file(self) -> None:
        """Test validate-policy command with missing file."""
        runner = CliRunner()
        result = runner.invoke(main, ["validate-policy", "/nonexistent/policy.yaml"])

        # Click will catch the file not existing before our code runs
        assert result.exit_code != 0
