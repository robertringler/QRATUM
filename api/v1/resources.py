"""Resource allocation dashboard endpoints.

This module provides endpoints for:
- Resource utilization dashboard
- Cluster status (EKS, GKE, AKS)
- User quotas and usage
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

try:
    from fastapi import APIRouter, Depends
    from pydantic import BaseModel, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

if FASTAPI_AVAILABLE:
    from api.v1.auth import require_scope
    from api.v1.jobs import JobStatus, _job_store

logger = logging.getLogger(__name__)


if FASTAPI_AVAILABLE:

    class ClusterSummary(BaseModel):
        """Cluster summary."""

        cluster_id: str
        name: str | None = None
        provider: str
        region: str
        status: str
        nodes: int
        gpus_available: int
        gpus_total: int

    class ClusterListResponse(BaseModel):
        """List of clusters."""

        clusters: list[ClusterSummary]

    class Utilization(BaseModel):
        """Resource utilization."""

        cpu: float = Field(ge=0.0, le=1.0)
        memory: float = Field(ge=0.0, le=1.0)
        gpu: float = Field(ge=0.0, le=1.0)

    class ResourceDashboard(BaseModel):
        """Resource dashboard data."""

        timestamp: str
        clusters: list[ClusterSummary]
        queue_depth: int
        jobs_running: int
        jobs_queued: int
        utilization: Utilization

    class Quotas(BaseModel):
        """User quotas."""

        max_concurrent_jobs: int
        max_gpu_hours_per_month: int
        max_storage_gb: int

    class Usage(BaseModel):
        """Current usage."""

        concurrent_jobs: int
        gpu_hours_this_month: float
        storage_gb: float

    class QuotaResponse(BaseModel):
        """Quota and usage response."""

        quotas: Quotas
        usage: Usage

    # Create router
    router = APIRouter(prefix="/resources", tags=["Resources"])

    @router.get("", response_model=ResourceDashboard)
    async def get_resources(
        user: dict[str, Any] = Depends(require_scope("read:resources")),
    ) -> ResourceDashboard:
        """Get resource allocation dashboard.

        Args:
            user: Authenticated user

        Returns:
            Resource dashboard data
        """
        # Get current job counts
        jobs_queued = sum(1 for j in _job_store.values() if j["status"] == JobStatus.QUEUED)
        jobs_running = sum(1 for j in _job_store.values() if j["status"] == JobStatus.RUNNING)

        clusters = _get_clusters()

        # Calculate overall utilization
        total_gpus = sum(c.gpus_total for c in clusters)
        used_gpus = sum(c.gpus_total - c.gpus_available for c in clusters)
        gpu_util = used_gpus / total_gpus if total_gpus > 0 else 0.0

        return ResourceDashboard(
            timestamp=datetime.utcnow().isoformat() + "Z",
            clusters=clusters,
            queue_depth=jobs_queued,
            jobs_running=jobs_running,
            jobs_queued=jobs_queued,
            utilization=Utilization(
                cpu=0.45,  # Placeholder
                memory=0.62,  # Placeholder
                gpu=gpu_util,
            ),
        )

    @router.get("/clusters", response_model=ClusterListResponse)
    async def list_clusters(
        user: dict[str, Any] = Depends(require_scope("read:resources")),
    ) -> ClusterListResponse:
        """List available compute clusters.

        Args:
            user: Authenticated user

        Returns:
            List of clusters
        """
        return ClusterListResponse(clusters=_get_clusters())

    @router.get("/quotas", response_model=QuotaResponse)
    async def get_quotas(
        user: dict[str, Any] = Depends(require_scope("read:resources")),
    ) -> QuotaResponse:
        """Get user's resource quotas and current usage.

        Args:
            user: Authenticated user

        Returns:
            Quotas and usage
        """
        user_id = user["user_id"]

        # Count user's jobs
        user_jobs = [j for j in _job_store.values() if j["user_id"] == user_id]
        concurrent = sum(
            1 for j in user_jobs if j["status"] in [JobStatus.QUEUED, JobStatus.RUNNING]
        )

        # Calculate GPU hours (placeholder - would query from metrics in production)
        gpu_hours = len([j for j in user_jobs if j["status"] == JobStatus.COMPLETED]) * 0.5

        return QuotaResponse(
            quotas=Quotas(
                max_concurrent_jobs=10,
                max_gpu_hours_per_month=1000,
                max_storage_gb=100,
            ),
            usage=Usage(
                concurrent_jobs=concurrent,
                gpu_hours_this_month=gpu_hours,
                storage_gb=5.2,  # Placeholder
            ),
        )


def _get_clusters() -> list[ClusterSummary]:
    """Get list of available clusters.

    Returns:
        List of cluster summaries
    """
    # In production, fetch from cluster registry/database
    return [
        ClusterSummary(
            cluster_id="eks-us-east-1",
            name="QRATUM US East",
            provider="aws",
            region="us-east-1",
            status="online",
            nodes=8,
            gpus_available=24,
            gpus_total=32,
        ),
        ClusterSummary(
            cluster_id="gke-us-central1",
            name="QRATUM US Central",
            provider="gcp",
            region="us-central1",
            status="online",
            nodes=6,
            gpus_available=16,
            gpus_total=24,
        ),
        ClusterSummary(
            cluster_id="aks-westeurope",
            name="QRATUM Europe",
            provider="azure",
            region="westeurope",
            status="online",
            nodes=4,
            gpus_available=8,
            gpus_total=16,
        ),
    ]
