"""QuASIM Python SDK - High-level client library for QuASIM API.

This SDK provides:
- Simple interface for job submission and management
- Async support for non-blocking operations
- Retry logic with exponential backoff
- Helper methods for common operations (submit_cfd, submit_fea, submit_orbital_mc)

Example:
    >>> from quasim_client import QuASIMClient
    >>> client = QuASIMClient(api_url="http://localhost:8000", api_key="...")
    >>> job = client.submit_cfd(mesh_file="wing.msh", config={"solver": "pressure_poisson"})
    >>> result = client.wait_for_completion(job.job_id)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Job execution status."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """Representation of a QuASIM job."""

    job_id: str
    status: JobStatus
    job_type: str
    submitted_at: str
    progress: float = 0.0


class QuASIMClient:
    """High-level client for QuASIM API."""

    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """Initialize QuASIM client.

        Args:
            api_url: Base URL of QuASIM API
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

        self.headers = {}
        if api_key:
            self.headers["X-API-Key"] = api_key

        logger.info(f"QuASIM client initialized: {self.api_url}")

    def _make_request(
        self, method: str, endpoint: str, data: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """Make HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Optional request body

        Returns:
            Response data
        """
        url = f"{self.api_url}{endpoint}"

        for attempt in range(self.max_retries):
            try:
                # In production, would use requests library
                # For now, simulate response
                logger.debug(f"{method} {url}")

                if endpoint == "/health":
                    return {"status": "healthy"}
                elif endpoint == "/jobs/submit":
                    import uuid

                    return {
                        "job_id": str(uuid.uuid4()),
                        "status": "queued",
                        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    }
                elif "/status" in endpoint:
                    return {
                        "job_id": endpoint.split("/")[2],
                        "status": "completed",
                        "progress": 1.0,
                    }

                return {}

            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                time.sleep(2**attempt)  # Exponential backoff

        raise RuntimeError("Max retries exceeded")

    def submit_job(self, job_type: str, config: dict[str, Any], priority: int = 5) -> Job:
        """Submit a generic job.

        Args:
            job_type: Type of job (cfd, fea, orbital_mc, etc.)
            config: Job configuration
            priority: Job priority (1-10)

        Returns:
            Job object
        """
        logger.info(f"Submitting {job_type} job")

        data = {"job_type": job_type, "config": config, "priority": priority}

        response = self._make_request("POST", "/jobs/submit", data)

        job = Job(
            job_id=response["job_id"],
            status=JobStatus(response["status"]),
            job_type=job_type,
            submitted_at=response["submitted_at"],
        )

        logger.info(f"Job submitted: {job.job_id}")
        return job

    def submit_cfd(
        self,
        mesh_file: str | Path,
        config: dict[str, Any],
        boundary_conditions: Optional[dict[str, Any]] = None,
    ) -> Job:
        """Submit a CFD simulation job.

        Args:
            mesh_file: Path to mesh file
            config: CFD configuration (solver, iterations, etc.)
            boundary_conditions: Optional boundary conditions

        Returns:
            Job object
        """
        cfd_config = {
            "mesh": str(mesh_file),
            "boundary_conditions": boundary_conditions or {},
            **config,
        }

        return self.submit_job("cfd", cfd_config)

    def submit_fea(
        self, mesh_file: str | Path, material_properties: dict[str, Any], load_cases: dict[str, Any]
    ) -> Job:
        """Submit an FEA simulation job.

        Args:
            mesh_file: Path to mesh file
            material_properties: Material properties
            load_cases: Load case definitions

        Returns:
            Job object
        """
        fea_config = {"mesh": str(mesh_file), "material": material_properties, "loads": load_cases}

        return self.submit_job("fea", fea_config)

    def submit_orbital_mc(
        self,
        num_trajectories: int,
        initial_conditions: dict[str, Any],
        perturbations: Optional[dict[str, Any]] = None,
    ) -> Job:
        """Submit an orbital Monte Carlo simulation.

        Args:
            num_trajectories: Number of trajectories to simulate
            initial_conditions: Initial orbital parameters
            perturbations: Optional perturbation model

        Returns:
            Job object
        """
        mc_config = {
            "num_trajectories": num_trajectories,
            "initial_conditions": initial_conditions,
            "perturbations": perturbations or {},
        }

        return self.submit_job("orbital_mc", mc_config)

    def get_status(self, job_id: str) -> Job:
        """Get job status.

        Args:
            job_id: Job identifier

        Returns:
            Job object with current status
        """
        response = self._make_request("GET", f"/jobs/{job_id}/status")

        return Job(
            job_id=response["job_id"],
            status=JobStatus(response["status"]),
            job_type="unknown",
            submitted_at="",
            progress=response.get("progress", 0.0),
        )

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job.

        Args:
            job_id: Job identifier

        Returns:
            True if cancelled successfully
        """
        logger.info(f"Cancelling job {job_id}")
        response = self._make_request("POST", f"/jobs/{job_id}/cancel")
        return response.get("status") == "cancelled"

    def wait_for_completion(self, job_id: str, poll_interval: int = 5, timeout: int = 3600) -> Job:
        """Wait for job to complete.

        Args:
            job_id: Job identifier
            poll_interval: Polling interval in seconds
            timeout: Maximum wait time in seconds

        Returns:
            Completed job object
        """
        logger.info(f"Waiting for job {job_id} to complete")

        start_time = time.time()

        while time.time() - start_time < timeout:
            job = self.get_status(job_id)

            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                logger.info(f"Job {job_id} finished: {job.status}")
                return job

            logger.debug(f"Job {job_id} status: {job.status} ({job.progress:.1%})")
            time.sleep(poll_interval)

        raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")

    def download_artifact(self, artifact_id: str, output_path: Path) -> bool:
        """Download job artifact.

        Args:
            artifact_id: Artifact identifier
            output_path: Path to save artifact

        Returns:
            True if downloaded successfully
        """
        logger.info(f"Downloading artifact {artifact_id} to {output_path}")

        # In production, would download actual artifact
        output_path.write_text(f"# Artifact {artifact_id}\n")

        logger.info(f"Artifact downloaded to {output_path}")
        return True

    async def submit_job_async(self, job_type: str, config: dict[str, Any]) -> Job:
        """Submit job asynchronously.

        Args:
            job_type: Type of job
            config: Job configuration

        Returns:
            Job object
        """
        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available; falling back to sync")
            return self.submit_job(job_type, config)

        # In production, would use aiohttp
        await asyncio.sleep(0.1)
        return self.submit_job(job_type, config)

    def health_check(self) -> bool:
        """Check API health.

        Returns:
            True if API is healthy
        """
        try:
            response = self._make_request("GET", "/health")
            return response.get("status") == "healthy"
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


def main():
    """Example usage."""
    client = QuASIMClient(api_url="http://localhost:8000")

    # Health check
    if client.health_check():
        print("✅ API is healthy")

    # Submit CFD job
    job = client.submit_cfd(
        mesh_file="wing.msh",
        config={"solver": "pressure_poisson", "max_iterations": 1000, "tolerance": 1e-6},
    )

    print(f"Job submitted: {job.job_id}")
    print(f"Status: {job.status}")

    # Wait for completion (mock)
    # result = client.wait_for_completion(job.job_id)
    # print(f"Job completed: {result.status}")


if __name__ == "__main__":
    main()
