"""Job submission and management endpoints.

This module provides endpoints for:
- Job submission (quantum circuits, tensor analysis, VQE, QAOA)
- Job listing with filtering
- Job cancellation
- Job validation

Integrates with existing QRATUM solvers without architectural changes.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any

try:
    from fastapi import APIRouter, Depends, HTTPException, Query, status
    from pydantic import BaseModel, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

if FASTAPI_AVAILABLE:
    from api.v1.auth import require_scope

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Job execution status."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(str, Enum):
    """Supported job types."""

    QUANTUM_CIRCUIT = "quantum_circuit"
    TENSOR_ANALYSIS = "tensor_analysis"
    VQE = "vqe"
    QAOA = "qaoa"
    DIGITAL_TWIN = "digital_twin"
    CFD = "cfd"
    FEA = "fea"
    ORBITAL_MC = "orbital_mc"


# In-memory job store (replace with database in production)
_job_store: dict[str, dict[str, Any]] = {}


if FASTAPI_AVAILABLE:

    class JobSubmitRequest(BaseModel):
        """Request to submit a new job."""

        job_type: JobType = Field(..., description="Type of simulation")
        name: str | None = Field(None, max_length=128, description="Job name")
        config: dict[str, Any] = Field(..., description="Job-type specific configuration")
        priority: int = Field(default=5, ge=1, le=10, description="Job priority (1-10)")
        timeout_seconds: int = Field(
            default=3600, ge=60, le=86400, description="Job timeout in seconds"
        )
        tags: list[str] | None = Field(None, max_length=10, description="Job tags")
        callback_url: str | None = Field(
            None, description="URL to POST results when job completes"
        )

    class JobSubmitResponse(BaseModel):
        """Response after job submission."""

        job_id: str = Field(..., description="Unique job identifier")
        status: JobStatus = Field(..., description="Initial job status")
        submitted_at: str = Field(..., description="Submission timestamp")
        estimated_duration_seconds: int | None = Field(
            None, description="Estimated duration"
        )

    class JobSummary(BaseModel):
        """Job summary for list responses."""

        job_id: str
        name: str | None = None
        job_type: JobType
        status: JobStatus
        submitted_at: str
        started_at: str | None = None
        completed_at: str | None = None
        progress: float = Field(default=0.0, ge=0.0, le=1.0)

    class JobDetail(JobSummary):
        """Detailed job information."""

        config: dict[str, Any]
        priority: int
        timeout_seconds: int
        tags: list[str] | None = None
        cluster: str | None = None
        worker_id: str | None = None
        error_message: str | None = None

    class JobListResponse(BaseModel):
        """Response for job listing."""

        jobs: list[JobSummary]
        total: int
        limit: int
        offset: int

    class JobCancelResponse(BaseModel):
        """Response for job cancellation."""

        job_id: str
        status: JobStatus
        cancelled_at: str

    class ValidationResponse(BaseModel):
        """Job validation response."""

        valid: bool
        errors: list[str] = Field(default_factory=list)
        warnings: list[str] = Field(default_factory=list)

    # Create router
    router = APIRouter(prefix="/jobs", tags=["Jobs"])

    @router.post("", response_model=JobSubmitResponse, status_code=status.HTTP_201_CREATED)
    async def submit_job(
        request: JobSubmitRequest,
        user: dict[str, Any] = Depends(require_scope("write:jobs")),
    ) -> JobSubmitResponse:
        """Submit a new simulation job.

        Args:
            request: Job submission request
            user: Authenticated user

        Returns:
            Job submission response with job ID
        """
        job_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat() + "Z"

        # Validate job configuration
        validation = _validate_job_config(request.job_type, request.config)
        if not validation["valid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "invalid_config", "errors": validation["errors"]},
            )

        # Estimate duration based on job type
        estimated_duration = _estimate_duration(request.job_type, request.config)

        # Store job
        job_data = {
            "job_id": job_id,
            "name": request.name,
            "job_type": request.job_type,
            "config": request.config,
            "priority": request.priority,
            "timeout_seconds": request.timeout_seconds,
            "tags": request.tags,
            "callback_url": request.callback_url,
            "status": JobStatus.QUEUED,
            "submitted_at": timestamp,
            "started_at": None,
            "completed_at": None,
            "progress": 0.0,
            "user_id": user["user_id"],
            "cluster": None,
            "worker_id": None,
            "error_message": None,
            "results": None,
        }
        _job_store[job_id] = job_data

        logger.info(
            "Job submitted: job_id=%s type=%s user=%s",
            job_id,
            request.job_type,
            user["user_id"],
        )

        return JobSubmitResponse(
            job_id=job_id,
            status=JobStatus.QUEUED,
            submitted_at=timestamp,
            estimated_duration_seconds=estimated_duration,
        )

    @router.get("", response_model=JobListResponse)
    async def list_jobs(
        status_filter: JobStatus | None = Query(None, alias="status"),
        job_type: JobType | None = Query(None),
        limit: int = Query(20, ge=1, le=100),
        offset: int = Query(0, ge=0),
        user: dict[str, Any] = Depends(require_scope("read:jobs")),
    ) -> JobListResponse:
        """List jobs with optional filtering.

        Args:
            status_filter: Filter by job status
            job_type: Filter by job type
            limit: Maximum results to return
            offset: Pagination offset
            user: Authenticated user

        Returns:
            List of jobs
        """
        # Filter jobs for current user
        user_jobs = [
            j for j in _job_store.values() if j["user_id"] == user["user_id"]
        ]

        # Apply filters
        if status_filter:
            user_jobs = [j for j in user_jobs if j["status"] == status_filter]
        if job_type:
            user_jobs = [j for j in user_jobs if j["job_type"] == job_type]

        # Sort by submission time (newest first)
        user_jobs.sort(key=lambda j: j["submitted_at"], reverse=True)

        total = len(user_jobs)
        paginated = user_jobs[offset : offset + limit]

        return JobListResponse(
            jobs=[
                JobSummary(
                    job_id=j["job_id"],
                    name=j.get("name"),
                    job_type=j["job_type"],
                    status=j["status"],
                    submitted_at=j["submitted_at"],
                    started_at=j.get("started_at"),
                    completed_at=j.get("completed_at"),
                    progress=j.get("progress", 0.0),
                )
                for j in paginated
            ],
            total=total,
            limit=limit,
            offset=offset,
        )

    @router.get("/{job_id}", response_model=JobDetail)
    async def get_job(
        job_id: str,
        user: dict[str, Any] = Depends(require_scope("read:jobs")),
    ) -> JobDetail:
        """Get detailed job information.

        Args:
            job_id: Job identifier
            user: Authenticated user

        Returns:
            Job details
        """
        job = _get_job_for_user(job_id, user["user_id"])

        return JobDetail(
            job_id=job["job_id"],
            name=job.get("name"),
            job_type=job["job_type"],
            status=job["status"],
            submitted_at=job["submitted_at"],
            started_at=job.get("started_at"),
            completed_at=job.get("completed_at"),
            progress=job.get("progress", 0.0),
            config=job["config"],
            priority=job["priority"],
            timeout_seconds=job["timeout_seconds"],
            tags=job.get("tags"),
            cluster=job.get("cluster"),
            worker_id=job.get("worker_id"),
            error_message=job.get("error_message"),
        )

    @router.delete("/{job_id}", response_model=JobCancelResponse)
    async def cancel_job(
        job_id: str,
        user: dict[str, Any] = Depends(require_scope("write:jobs")),
    ) -> JobCancelResponse:
        """Cancel a queued or running job.

        Args:
            job_id: Job identifier
            user: Authenticated user

        Returns:
            Cancellation confirmation
        """
        job = _get_job_for_user(job_id, user["user_id"])

        if job["status"] in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot cancel job in status: {job['status']}",
            )

        timestamp = datetime.utcnow().isoformat() + "Z"
        job["status"] = JobStatus.CANCELLED
        job["completed_at"] = timestamp

        logger.info("Job cancelled: job_id=%s user=%s", job_id, user["user_id"])

        return JobCancelResponse(
            job_id=job_id,
            status=JobStatus.CANCELLED,
            cancelled_at=timestamp,
        )

    # Validation router
    validation_router = APIRouter(tags=["Jobs"])

    @validation_router.post("/validate", response_model=ValidationResponse)
    async def validate_job(
        request: JobSubmitRequest,
        user: dict[str, Any] = Depends(require_scope("read:jobs")),
    ) -> ValidationResponse:
        """Validate job configuration without submitting.

        Args:
            request: Job configuration to validate
            user: Authenticated user

        Returns:
            Validation result
        """
        validation = _validate_job_config(request.job_type, request.config)
        return ValidationResponse(**validation)


def _get_job_for_user(job_id: str, user_id: str) -> dict[str, Any]:
    """Get job if it belongs to user.

    Args:
        job_id: Job identifier
        user_id: User identifier

    Returns:
        Job data

    Raises:
        HTTPException: If job not found or unauthorized
    """
    if job_id not in _job_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )

    job = _job_store[job_id]
    if job["user_id"] != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )

    return job


def _validate_job_config(job_type: JobType, config: dict[str, Any]) -> dict[str, Any]:
    """Validate job configuration.

    Args:
        job_type: Type of job
        config: Job configuration

    Returns:
        Validation result with errors and warnings
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not config:
        errors.append("Configuration cannot be empty")
        return {"valid": False, "errors": errors, "warnings": warnings}

    # Job-type specific validation
    if job_type == JobType.QUANTUM_CIRCUIT:
        if "circuit" not in config:
            errors.append("quantum_circuit jobs require 'circuit' in config")
        if "shots" not in config:
            warnings.append("No 'shots' specified; will use default (1024)")

    elif job_type == JobType.VQE:
        if "molecule" not in config:
            errors.append("vqe jobs require 'molecule' in config")
        if "optimizer" not in config:
            warnings.append("No 'optimizer' specified; will use default (COBYLA)")

    elif job_type == JobType.QAOA:
        if "problem" not in config:
            errors.append("qaoa jobs require 'problem' in config")
        if "p_layers" not in config:
            warnings.append("No 'p_layers' specified; will use default (3)")

    elif job_type == JobType.TENSOR_ANALYSIS:
        if "tensor" not in config and "tensor_network" not in config:
            errors.append("tensor_analysis jobs require 'tensor' or 'tensor_network' in config")

    elif job_type == JobType.DIGITAL_TWIN:
        if "system_type" not in config:
            errors.append("digital_twin jobs require 'system_type' in config")
        if "initial_state" not in config:
            errors.append("digital_twin jobs require 'initial_state' in config")

    elif job_type == JobType.CFD:
        if "mesh" not in config:
            errors.append("cfd jobs require 'mesh' in config")
        if "solver" not in config:
            warnings.append("No 'solver' specified; will use default")

    elif job_type == JobType.FEA:
        if "model" not in config:
            errors.append("fea jobs require 'model' in config")

    elif job_type == JobType.ORBITAL_MC:
        if "trajectories" not in config:
            errors.append("orbital_mc jobs require 'trajectories' in config")

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


def _estimate_duration(job_type: JobType, config: dict[str, Any]) -> int:
    """Estimate job duration based on type and configuration.

    Args:
        job_type: Type of job
        config: Job configuration

    Returns:
        Estimated duration in seconds
    """
    base_durations = {
        JobType.QUANTUM_CIRCUIT: 30,
        JobType.VQE: 120,
        JobType.QAOA: 90,
        JobType.TENSOR_ANALYSIS: 60,
        JobType.DIGITAL_TWIN: 300,
        JobType.CFD: 600,
        JobType.FEA: 450,
        JobType.ORBITAL_MC: 180,
    }

    base = base_durations.get(job_type, 60)

    # Adjust based on configuration
    if job_type == JobType.QUANTUM_CIRCUIT:
        shots = config.get("shots", 1024)
        base = base * (shots / 1024)

    elif job_type == JobType.VQE:
        max_iterations = config.get("max_iterations", 100)
        base = base * (max_iterations / 100)

    elif job_type == JobType.QAOA:
        p_layers = config.get("p_layers", 3)
        base = base * (p_layers / 3)

    return int(base)


def get_job_store() -> dict[str, dict[str, Any]]:
    """Get the job store (for testing).

    Returns:
        Job store dictionary
    """
    return _job_store


def clear_job_store() -> None:
    """Clear the job store (for testing)."""
    _job_store.clear()
