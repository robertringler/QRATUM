"""Integration tests for QuASIM SDK."""

import sys
from pathlib import Path

import pytest

# Add SDK to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "integrations" / "sdk" / "python"))

from quasim_client import JobStatus, QuASIMClient


class TestSDK:
    """Test suite for QuASIM Python SDK."""

    def test_client_creation(self):
        """Test that client can be created."""
        client = QuASIMClient(api_url="http://localhost:8000")
        assert client is not None
        assert client.api_url == "http://localhost:8000"

    def test_health_check(self):
        """Test health check endpoint."""
        client = QuASIMClient(api_url="http://localhost:8000")
        # Mock health check returns True
        assert client.health_check() is True

    def test_job_submission(self):
        """Test job submission."""
        client = QuASIMClient(api_url="http://localhost:8000")

        job = client.submit_job(
            job_type="cfd",
            config={"solver": "pressure_poisson", "max_iterations": 100}
        )

        assert job is not None
        assert job.job_id is not None
        assert job.status == JobStatus.QUEUED
        assert job.job_type == "cfd"

    def test_cfd_job_submission(self):
        """Test CFD-specific job submission."""
        client = QuASIMClient(api_url="http://localhost:8000")

        job = client.submit_cfd(
            mesh_file="test_wing.msh",
            config={"solver": "pressure_poisson", "max_iterations": 1000},
            boundary_conditions={"inlet": {"type": "velocity", "value": [1, 0, 0]}}
        )

        assert job is not None
        assert job.status == JobStatus.QUEUED

    def test_fea_job_submission(self):
        """Test FEA job submission."""
        client = QuASIMClient(api_url="http://localhost:8000")

        job = client.submit_fea(
            mesh_file="composite_plate.msh",
            material_properties={"E": 70e9, "nu": 0.3},
            load_cases={"load1": {"type": "pressure", "value": 1e6}}
        )

        assert job is not None
        assert job.status == JobStatus.QUEUED

    def test_orbital_mc_submission(self):
        """Test orbital MC job submission."""
        client = QuASIMClient(api_url="http://localhost:8000")

        job = client.submit_orbital_mc(
            num_trajectories=1000,
            initial_conditions={"a": 7000e3, "e": 0.001, "i": 98.0}
        )

        assert job is not None
        assert job.status == JobStatus.QUEUED

    def test_job_status(self):
        """Test getting job status."""
        client = QuASIMClient(api_url="http://localhost:8000")

        # Submit job first
        job = client.submit_job("cfd", {"solver": "test"})

        # Get status
        status = client.get_status(job.job_id)
        assert status is not None
        assert status.job_id == job.job_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
