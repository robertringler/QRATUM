"""Tests for CLI commands."""

import json
import subprocess
import sys

import pytest


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory."""
    return tmp_path


class TestRevultraCLI:
    """Tests for quasim-revultra CLI."""

    def test_cli_help(self):
        """Test that CLI help works."""
        result = subprocess.run(
            [sys.executable, "-m", "quasim.cli.revultra_cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "REVULTRA" in result.stdout

    def test_analyze_command(self, temp_output_dir):
        """Test analyze command."""
        output_file = temp_output_dir / "results.json"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "quasim.cli.revultra_cli",
                "analyze",
                "--ciphertext",
                "ATTACKATDAWN",
                "--export",
                str(output_file),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert output_file.exists()

        # Verify JSON output
        with open(output_file) as f:
            data = json.load(f)

        assert "complexity" in data
        assert "frequency_analysis" in data

    def test_frequency_command(self):
        """Test frequency command."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "quasim.cli.revultra_cli",
                "frequency",
                "--ciphertext",
                "HELLO",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Character Frequencies" in result.stdout

    def test_demo_command(self, temp_output_dir):
        """Test demo command."""
        output_file = temp_output_dir / "demo.json"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "quasim.cli.revultra_cli",
                "demo",
                "--section",
                "kryptos",
                "--export",
                str(output_file),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Kryptos K4" in result.stdout


class TestQGHCLI:
    """Tests for quasim-qgh CLI."""

    def test_cli_help(self):
        """Test that CLI help works."""
        result = subprocess.run(
            [sys.executable, "-m", "quasim.cli.qgh_cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "QGH" in result.stdout

    def test_demo_command(self, temp_output_dir):
        """Test demo command."""
        output_file = temp_output_dir / "qgh_demo.json"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "quasim.cli.qgh_cli",
                "demo",
                "--section",
                "tensor",
                "--export",
                str(output_file),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert output_file.exists()

    def test_resolve_command(self):
        """Test resolve command."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "quasim.cli.qgh_cli",
                "resolve",
                "--state",
                "0.5,0.3,0.15,0.05",
                "--iterations",
                "50",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Converged" in result.stdout

    def test_monitor_command(self):
        """Test monitor command."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "quasim.cli.qgh_cli",
                "monitor",
                "--num-streams",
                "3",
                "--samples",
                "30",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Monitoring" in result.stdout or "Stream Statistics" in result.stdout


class TestTERCObsCLI:
    """Tests for quasim-terc-obs CLI."""

    def test_cli_help(self):
        """Test that CLI help works."""
        result = subprocess.run(
            [sys.executable, "-m", "quasim.cli.terc_obs_cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "TERC" in result.stdout

    def test_list_command(self):
        """Test list command."""
        result = subprocess.run(
            [sys.executable, "-m", "quasim.cli.terc_obs_cli", "list"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Registered" in result.stdout or "observables" in result.stdout.lower()

    def test_emit_command(self, temp_output_dir):
        """Test emit command."""
        output_file = temp_output_dir / "observables.json"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "quasim.cli.terc_obs_cli",
                "emit",
                "--text",
                "ATTACKATDAWN",
                "--out",
                str(output_file),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert output_file.exists()

        # Verify JSON structure
        with open(output_file) as f:
            data = json.load(f)

        assert "observables" in data
        assert "format_version" in data

    def test_consensus_command(self, temp_output_dir):
        """Test consensus command."""
        output_file = temp_output_dir / "consensus.json"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "quasim.cli.terc_obs_cli",
                "consensus",
                "--num-nodes",
                "5",
                "--state-dim",
                "3",
                "--out",
                str(output_file),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert output_file.exists()

    def test_validate_command(self, temp_output_dir):
        """Test validate command."""
        # Create a valid observable file
        obs_file = temp_output_dir / "test_obs.json"
        obs_data = {
            "observables": {"test": 1.0},
            "format_version": "1.0",
            "source": "test",
        }
        with open(obs_file, "w") as f:
            json.dump(obs_data, f)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "quasim.cli.terc_obs_cli",
                "validate",
                "--obs-file",
                str(obs_file),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "valid" in result.stdout.lower()


class TestCLIIntegration:
    """Integration tests for CLI workflows."""

    def test_revultra_to_terc_pipeline(self, temp_output_dir):
        """Test pipeline from REVULTRA analysis to TERC emission."""
        # Step 1: Run REVULTRA analysis
        revultra_output = temp_output_dir / "revultra_results.json"
        result1 = subprocess.run(
            [
                sys.executable,
                "-m",
                "quasim.cli.revultra_cli",
                "analyze",
                "--ciphertext",
                "ATTACKATDAWN" * 10,
                "--export",
                str(revultra_output),
            ],
            capture_output=True,
            text=True,
        )
        assert result1.returncode == 0
        assert revultra_output.exists()

        # Step 2: Emit TERC observables
        terc_output = temp_output_dir / "terc_obs.json"
        result2 = subprocess.run(
            [
                sys.executable,
                "-m",
                "quasim.cli.terc_obs_cli",
                "emit",
                "--text",
                "ATTACKATDAWN" * 10,
                "--out",
                str(terc_output),
            ],
            capture_output=True,
            text=True,
        )
        assert result2.returncode == 0
        assert terc_output.exists()

        # Step 3: Validate TERC observables
        result3 = subprocess.run(
            [
                sys.executable,
                "-m",
                "quasim.cli.terc_obs_cli",
                "validate",
                "--obs-file",
                str(terc_output),
            ],
            capture_output=True,
            text=True,
        )
        assert result3.returncode == 0

    def test_qgh_demo_to_json(self, temp_output_dir):
        """Test QGH demo with JSON export."""
        output_file = temp_output_dir / "qgh_all_demos.json"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "quasim.cli.qgh_cli",
                "demo",
                "--section",
                "all",
                "--export",
                str(output_file),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert output_file.exists()

        # Verify JSON structure
        with open(output_file) as f:
            data = json.load(f)

        # Should contain results from multiple demos
        assert len(data) > 0
