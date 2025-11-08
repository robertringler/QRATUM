"""
Tests for QuASIM HCAL CLI module.
"""

import pytest
from click.testing import CliRunner

from quasim.hcal.cli import cli


@pytest.fixture
def runner():
    """Create a Click CLI test runner."""
    return CliRunner()


def test_cli_help(runner):
    """Test that CLI help is displayed."""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "QuASIM Hardware Calibration and Analysis Layer" in result.output
    assert "status" in result.output
    assert "calibrate" in result.output
    assert "monitor" in result.output
    assert "info" in result.output


def test_cli_version(runner):
    """Test that CLI version is displayed."""
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_info_command(runner):
    """Test the info command."""
    result = runner.invoke(cli, ["info"])
    assert result.exit_code == 0
    assert "QuASIM HCAL Information" in result.output
    assert "Version: 0.1.0" in result.output
    assert "Python:" in result.output


def test_status_command(runner):
    """Test the status command."""
    result = runner.invoke(cli, ["status"])
    assert result.exit_code == 0
    assert "Hardware Status" in result.output


def test_calibrate_command(runner):
    """Test the calibrate command."""
    result = runner.invoke(cli, ["calibrate"])
    assert result.exit_code == 0
    assert "Running hardware calibration" in result.output
    assert "Calibration complete" in result.output


def test_calibrate_command_with_config(runner, tmp_path):
    """Test the calibrate command with config file."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("test: value\n")

    result = runner.invoke(cli, ["calibrate", "--config", str(config_file)])
    assert result.exit_code == 0
    assert "Using configuration:" in result.output
    assert str(config_file) in result.output


def test_calibrate_command_with_output(runner, tmp_path):
    """Test the calibrate command with output path."""
    output_file = tmp_path / "results.yaml"

    result = runner.invoke(cli, ["calibrate", "--output", str(output_file)])
    assert result.exit_code == 0
    assert "Results saved to:" in result.output
    assert str(output_file) in result.output


def test_monitor_command_short_duration(runner):
    """Test the monitor command with short duration."""
    result = runner.invoke(cli, ["monitor", "--duration", "1", "--interval", "1"])
    assert result.exit_code == 0
    assert "Monitoring hardware" in result.output
    assert "Monitoring complete" in result.output
