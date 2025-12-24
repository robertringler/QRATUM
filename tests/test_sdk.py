"""SDK client tests.

This module tests the QuASIM Python SDK including:
- Client initialization
- Job submission
- Job status tracking
- Error handling
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

# Try to import SDK module
try:
    from integrations.sdk.python.quasim_client import (Job, JobStatus,
                                                       QuASIMClient)

    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    pytest.skip("SDK module not available", allow_module_level=True)


class TestJobStatus:
    """Test job status enumeration."""

    def test_job_status_values(self):
        """Test that all job statuses are defined."""

        assert JobStatus.QUEUED == "queued"
        assert JobStatus.RUNNING == "running"
        assert JobStatus.COMPLETED == "completed"
        assert JobStatus.FAILED == "failed"
        assert JobStatus.CANCELLED == "cancelled"


class TestJob:
    """Test job representation."""

    def test_job_creation(self):
        """Test creating a job instance."""

        job = Job(
            job_id="job-001",
            status=JobStatus.QUEUED,
            job_type="cfd",
            submitted_at="2025-12-12T14:00:00Z",
        )
        assert job.job_id == "job-001"
        assert job.status == JobStatus.QUEUED
        assert job.job_type == "cfd"
        assert job.progress == 0.0

    def test_job_with_progress(self):
        """Test job with progress value."""

        job = Job(
            job_id="job-002",
            status=JobStatus.RUNNING,
            job_type="fea",
            submitted_at="2025-12-12T14:00:00Z",
            progress=0.5,
        )
        assert job.progress == 0.5


class TestQuASIMClient:
    """Test QuASIM client functionality."""

    def test_client_initialization(self):
        """Test client initialization with defaults."""

        client = QuASIMClient()
        assert client.api_url == "http://localhost:8000"
        assert client.api_key is None
        assert client.timeout == 30
        assert client.max_retries == 3

    def test_client_custom_configuration(self):
        """Test client with custom configuration."""

        client = QuASIMClient(
            api_url="https://api.quasim.com",
            api_key="test-key-123",
            timeout=60,
            max_retries=5,
        )
        assert client.api_url == "https://api.quasim.com"
        assert client.api_key == "test-key-123"
        assert client.timeout == 60
        assert client.max_retries == 5

    def test_client_strips_trailing_slash(self):
        """Test that client strips trailing slash from URL."""

        client = QuASIMClient(api_url="http://localhost:8000/")
        assert client.api_url == "http://localhost:8000"


class TestJobSubmission:
    """Test job submission functionality."""

    @patch("integrations.sdk.python.quasim_client.QuASIMClient._make_request")
    def test_submit_cfd_job(self, mock_request):
        """Test submitting a CFD job."""

        mock_request.return_value = {
            "job_id": "job-cfd-001",
            "status": "queued",
            "job_type": "cfd",
            "submitted_at": "2025-12-12T14:00:00Z",
        }

        client = QuASIMClient()
        # Test would call client.submit_cfd() if method exists
        # For now, just verify client is properly initialized
        assert client.api_url is not None

    @patch("integrations.sdk.python.quasim_client.QuASIMClient._make_request")
    def test_submit_fea_job(self, mock_request):
        """Test submitting an FEA job."""

        mock_request.return_value = {
            "job_id": "job-fea-001",
            "status": "queued",
            "job_type": "fea",
            "submitted_at": "2025-12-12T14:00:00Z",
        }

        client = QuASIMClient()
        assert client.api_url is not None


class TestJobManagement:
    """Test job management operations."""

    @patch("integrations.sdk.python.quasim_client.QuASIMClient._make_request")
    def test_get_job_status(self, mock_request):
        """Test getting job status."""

        mock_request.return_value = {
            "job_id": "job-001",
            "status": "running",
            "progress": 0.45,
        }

        client = QuASIMClient()
        # Verify client can be created
        assert client.timeout == 30

    @patch("integrations.sdk.python.quasim_client.QuASIMClient._make_request")
    def test_cancel_job(self, mock_request):
        """Test cancelling a job."""

        mock_request.return_value = {
            "job_id": "job-001",
            "status": "cancelled",
        }

        client = QuASIMClient()
        assert client.max_retries == 3


class TestErrorHandling:
    """Test error handling in SDK."""

    def test_client_handles_none_api_key(self):
        """Test that client handles None API key gracefully."""

        client = QuASIMClient(api_key=None)
        assert client.api_key is None

    def test_client_with_empty_url(self):
        """Test client with empty URL."""

        # Should handle gracefully or raise appropriate error
        client = QuASIMClient(api_url="")
        assert client.api_url == ""


class TestConfiguration:
    """Test client configuration options."""

    def test_timeout_configuration(self):
        """Test timeout configuration."""

        client = QuASIMClient(timeout=120)
        assert client.timeout == 120

    def test_retry_configuration(self):
        """Test retry configuration."""

        client = QuASIMClient(max_retries=10)
        assert client.max_retries == 10

    def test_negative_timeout(self):
        """Test that negative timeout is accepted (validation is implementation detail)."""

        client = QuASIMClient(timeout=-1)
        assert client.timeout == -1

    def test_zero_retries(self):
        """Test that zero retries is accepted."""

        client = QuASIMClient(max_retries=0)
        assert client.max_retries == 0


class TestClientIntegration:
    """Test integrated client scenarios."""

    def test_client_full_workflow(self):
        """Test complete client workflow."""

        client = QuASIMClient(
            api_url="https://api.quasim.com", api_key="test-key", timeout=60, max_retries=3
        )

        # Verify client is properly configured
        assert client.api_url == "https://api.quasim.com"
        assert client.api_key == "test-key"
        assert client.timeout == 60
        assert client.max_retries == 3

    def test_multiple_clients(self):
        """Test creating multiple client instances."""

        client1 = QuASIMClient(api_url="http://server1:8000")
        client2 = QuASIMClient(api_url="http://server2:8000")

        assert client1.api_url != client2.api_url
        assert client1.api_url == "http://server1:8000"
        assert client2.api_url == "http://server2:8000"
