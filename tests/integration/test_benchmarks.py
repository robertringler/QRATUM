"""Integration tests for benchmarks."""

import subprocess
import tempfile
from pathlib import Path

import pytest


class TestBenchmarks:
    """Test suite for benchmark harness."""

    def test_benchmark_quick_mode(self):
        """Test quick benchmark mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    "python3",
                    "integrations/benchmarks/aero/run_benchmarks.py",
                    "--quick",
                    "--output-dir",
                    tmpdir,
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0, f"Stderr: {result.stderr}"

            # Check output files
            tmpdir_path = Path(tmpdir)
            assert (tmpdir_path / "perf.csv").exists()
            assert (tmpdir_path / "perf_substitution_table.md").exists()

            # Verify CSV content
            csv_content = (tmpdir_path / "perf.csv").read_text()
            assert "Scenario" in csv_content
            assert "CFD" in csv_content
            assert "FEA" in csv_content
            assert "Orbital_MC" in csv_content

            # Verify Markdown content
            md_content = (tmpdir_path / "perf_substitution_table.md").read_text()
            assert "Performance Substitution Table" in md_content
            assert "Speedup" in md_content

    def test_cfd_kernel_standalone(self):
        """Test CFD kernel can run standalone."""
        result = subprocess.run(
            ["python3", "integrations/kernels/cfd/pressure_poisson.py"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0, f"Stderr: {result.stderr}"
        assert "Pressure Poisson Solver - Summary" in result.stdout
        assert "Status: converged" in result.stdout or "Status: max_iterations" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
