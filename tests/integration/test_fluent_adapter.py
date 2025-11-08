"""Integration tests for Fluent adapter."""

import json
import subprocess
import tempfile
from pathlib import Path

import pytest


class TestFluentAdapter:
    """Test suite for Fluent adapter integration."""

    def test_fluent_adapter_help(self):
        """Test that Fluent adapter help works."""
        result = subprocess.run(
            ["python3", "integrations/adapters/fluent/quasim_fluent_driver.py", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "QuASIM Fluent Driver" in result.stdout

    def test_fluent_adapter_minimal_run(self):
        """Test Fluent adapter with minimal inputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create test mesh file
            mesh_file = tmpdir_path / "test_mesh.msh"
            mesh_file.write_text("# Test mesh file\n")

            # Create job config
            job_config = tmpdir_path / "job_config.json"
            job_config.write_text(
                json.dumps(
                    {
                        "solver": "pressure_poisson",
                        "max_iterations": 10,
                        "convergence_tolerance": 1e-6,
                    }
                )
            )

            # Create output path
            output_file = tmpdir_path / "results.csv"

            # Run adapter
            result = subprocess.run(
                [
                    "python3",
                    "integrations/adapters/fluent/quasim_fluent_driver.py",
                    "--mesh",
                    str(mesh_file),
                    "--job",
                    str(job_config),
                    "--output",
                    str(output_file),
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0, f"Stderr: {result.stderr}"
            assert output_file.exists()
            assert "QuASIM Fluent Driver completed successfully" in result.stderr

    def test_fluent_adapter_with_boundary_conditions(self):
        """Test Fluent adapter with boundary conditions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create test files
            mesh_file = tmpdir_path / "test_mesh.msh"
            mesh_file.write_text("# Test mesh\n")

            bc_file = tmpdir_path / "bc.yaml"
            bc_file.write_text("inlet:\n  type: velocity\n  value: [1, 0, 0]\n")

            job_config = tmpdir_path / "job.json"
            job_config.write_text(
                json.dumps(
                    {
                        "solver": "pressure_poisson",
                        "max_iterations": 10,
                        "convergence_tolerance": 1e-6,
                    }
                )
            )

            output_file = tmpdir_path / "results.csv"

            # Run adapter with BC
            result = subprocess.run(
                [
                    "python3",
                    "integrations/adapters/fluent/quasim_fluent_driver.py",
                    "--mesh",
                    str(mesh_file),
                    "--bc",
                    str(bc_file),
                    "--job",
                    str(job_config),
                    "--output",
                    str(output_file),
                    "--format",
                    "csv",
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0
            assert output_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
