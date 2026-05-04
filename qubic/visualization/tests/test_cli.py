"""Tests for CLI."""

import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from qubic.visualization.cli import cli


class TestCLI:
    """Tests for visualization CLI."""

    def setup_method(self):
        """Set up test fixtures."""

        self.runner = CliRunner()

    def test_cli_help(self):
        """Test CLI help command."""

        result = self.runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "QUBIC Visualization Subsystem" in result.output

    def test_render_help(self):
        """Test render command help."""

        result = self.runner.invoke(cli, ["render", "--help"])
        assert result.exit_code == 0
        assert "Render a single-frame visualization" in result.output

    def test_render_tire_synthetic(self):
        """Test rendering synthetic tire data."""

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "tire_test.png"

            result = self.runner.invoke(
                cli,
                [
                    "render",
                    "--adapter",
                    "tire",
                    "--backend",
                    "headless",
                    "--output",
                    str(output_path),
                ],
            )

            assert result.exit_code == 0
            assert output_path.exists()

    def test_render_quantum_synthetic(self):
        """Test rendering synthetic quantum data."""

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "quantum_test.png"

            result = self.runner.invoke(
                cli,
                [
                    "render",
                    "--adapter",
                    "quantum",
                    "--backend",
                    "headless",
                    "--output",
                    str(output_path),
                ],
            )

            assert result.exit_code == 0
            assert output_path.exists()

    def test_render_mesh_synthetic(self):
        """Test rendering synthetic mesh data."""

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "mesh_test.png"

            result = self.runner.invoke(
                cli,
                [
                    "render",
                    "--adapter",
                    "mesh",
                    "--backend",
                    "headless",
                    "--output",
                    str(output_path),
                ],
            )

            assert result.exit_code == 0
            assert output_path.exists()

    def test_render_with_field(self):
        """Test rendering with scalar field specified."""

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "tire_field.png"

            result = self.runner.invoke(
                cli,
                [
                    "render",
                    "--adapter",
                    "tire",
                    "--backend",
                    "headless",
                    "--field",
                    "temperature",
                    "--output",
                    str(output_path),
                ],
            )

            assert result.exit_code == 0
            assert output_path.exists()

    def test_render_with_colormap(self):
        """Test rendering with custom colormap."""

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "custom_colormap.png"

            result = self.runner.invoke(
                cli,
                [
                    "render",
                    "--adapter",
                    "tire",
                    "--backend",
                    "headless",
                    "--colormap",
                    "plasma",
                    "--output",
                    str(output_path),
                ],
            )

            assert result.exit_code == 0
            assert output_path.exists()

    @pytest.mark.slow
    def test_animate_gif(self):
        """Test animation creation (GIF)."""

        try:
            import imageio  # noqa: F401
        except ImportError:
            pytest.skip("imageio not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "animation.gif"

            result = self.runner.invoke(
                cli,
                [
                    "animate",
                    "--output",
                    str(output_path),
                    "--fps",
                    "5",
                ],
            )

            assert result.exit_code == 0
            assert output_path.exists()

    def test_example_tire(self):
        """Test tire example."""

        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.runner.invoke(
                cli,
                [
                    "example",
                    "tire",
                    "--output-dir",
                    tmpdir,
                ],
            )

            assert result.exit_code == 0

            # Check that output files were created
            output_dir = Path(tmpdir)
            assert any(output_dir.glob("*.png"))

    def test_example_quantum(self):
        """Test quantum example."""

        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.runner.invoke(
                cli,
                [
                    "example",
                    "quantum",
                    "--output-dir",
                    tmpdir,
                ],
            )

            assert result.exit_code == 0

            # Check that output files were created
            output_dir = Path(tmpdir)
            assert any(output_dir.glob("*.png"))
