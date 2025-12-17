"""Real-time status monitoring with WebSocket support.

This module provides:
- HTTP endpoints for job status
- WebSocket endpoint for real-time status updates
- Status subscription management
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any

try:
    from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
    from pydantic import BaseModel, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

if FASTAPI_AVAILABLE:
    from api.v1.auth import TokenManager, require_scope
    from api.v1.jobs import JobStatus, _job_store

logger = logging.getLogger(__name__)


# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections and subscriptions."""

    def __init__(self):
        """Initialize connection manager."""
        # Map of job_id -> set of websocket connections
        self._subscriptions: dict[str, set[WebSocket]] = defaultdict(set)
        # Map of websocket -> set of job_ids
        self._connection_jobs: dict[WebSocket, set[str]] = defaultdict(set)
        # Map of websocket -> user_id
        self._connection_users: dict[WebSocket, str] = {}

    async def connect(self, websocket: WebSocket, user_id: str) -> None:
        """Accept a new WebSocket connection.

        Args:
            websocket: WebSocket connection
            user_id: Authenticated user ID
        """
        await websocket.accept()
        self._connection_users[websocket] = user_id
        logger.info("WebSocket connected: user=%s", user_id)

    def disconnect(self, websocket: WebSocket) -> None:
        """Handle WebSocket disconnection.

        Args:
            websocket: WebSocket connection
        """
        user_id = self._connection_users.pop(websocket, "unknown")

        # Remove from all subscriptions
        for job_id in self._connection_jobs.get(websocket, set()):
            self._subscriptions[job_id].discard(websocket)
            if not self._subscriptions[job_id]:
                del self._subscriptions[job_id]

        self._connection_jobs.pop(websocket, None)
        logger.info("WebSocket disconnected: user=%s", user_id)

    def subscribe(self, websocket: WebSocket, job_id: str) -> bool:
        """Subscribe to job status updates.

        Args:
            websocket: WebSocket connection
            job_id: Job ID to subscribe to

        Returns:
            True if subscription was successful
        """
        # Verify job exists and belongs to user
        user_id = self._connection_users.get(websocket)
        if not user_id:
            return False

        job = _job_store.get(job_id)
        if not job or job.get("user_id") != user_id:
            return False

        self._subscriptions[job_id].add(websocket)
        self._connection_jobs[websocket].add(job_id)
        logger.debug("Subscribed to job: job_id=%s user=%s", job_id, user_id)
        return True

    def unsubscribe(self, websocket: WebSocket, job_id: str) -> None:
        """Unsubscribe from job status updates.

        Args:
            websocket: WebSocket connection
            job_id: Job ID to unsubscribe from
        """
        self._subscriptions[job_id].discard(websocket)
        self._connection_jobs[websocket].discard(job_id)

        if not self._subscriptions[job_id]:
            del self._subscriptions[job_id]

    async def broadcast_status(self, job_id: str, status_data: dict[str, Any]) -> None:
        """Broadcast status update to all subscribers.

        Args:
            job_id: Job ID
            status_data: Status data to broadcast
        """
        message = json.dumps({"type": "status", "job_id": job_id, **status_data})

        disconnected = []
        for websocket in self._subscriptions.get(job_id, set()):
            try:
                await websocket.send_text(message)
            except Exception:
                disconnected.append(websocket)

        # Clean up disconnected clients
        for ws in disconnected:
            self.disconnect(ws)


# Global connection manager
_connection_manager = ConnectionManager()


def get_connection_manager() -> ConnectionManager:
    """Get the connection manager instance."""
    return _connection_manager


if FASTAPI_AVAILABLE:

    class JobStatusResponse(BaseModel):
        """Job status response."""

        job_id: str
        status: JobStatus
        progress: float = Field(ge=0.0, le=1.0)
        message: str | None = None
        metrics: dict[str, Any] | None = None

    class StatusMetrics(BaseModel):
        """Status metrics."""

        elapsed_seconds: float | None = None
        cpu_utilization: float | None = None
        memory_mb: int | None = None
        gpu_utilization: float | None = None

    # Create router
    router = APIRouter(tags=["Status"])

    @router.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
    async def get_job_status(
        job_id: str,
        user: dict[str, Any] = Depends(require_scope("read:jobs")),
    ) -> JobStatusResponse:
        """Get current job status.

        Args:
            job_id: Job identifier
            user: Authenticated user

        Returns:
            Job status
        """
        if job_id not in _job_store:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job not found: {job_id}",
            )

        job = _job_store[job_id]
        if job["user_id"] != user["user_id"]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job not found: {job_id}",
            )

        # Calculate metrics if job is running
        metrics = None
        if job["status"] == JobStatus.RUNNING and job.get("started_at"):
            started = datetime.fromisoformat(job["started_at"].replace("Z", "+00:00"))
            elapsed = (datetime.utcnow().replace(tzinfo=started.tzinfo) - started).total_seconds()
            metrics = {
                "elapsed_seconds": elapsed,
                "cpu_utilization": 0.45,  # Placeholder
                "memory_mb": 512,  # Placeholder
                "gpu_utilization": 0.65,  # Placeholder
            }

        return JobStatusResponse(
            job_id=job_id,
            status=job["status"],
            progress=job.get("progress", 0.0),
            message=job.get("error_message"),
            metrics=metrics,
        )

    @router.websocket("/ws/status")
    async def websocket_status(websocket: WebSocket):
        """WebSocket endpoint for real-time status updates.

        Protocol:
        - Connect: Include token in query param or first message
        - Subscribe: {"action": "subscribe", "job_id": "uuid"}
        - Unsubscribe: {"action": "unsubscribe", "job_id": "uuid"}
        - Status update: {"type": "status", "job_id": "uuid", "status": "running", ...}
        """
        manager = get_connection_manager()
        token_manager = TokenManager()

        # Get token from query parameter
        token = websocket.query_params.get("token")
        if not token:
            await websocket.close(code=4001, reason="Missing authentication token")
            return

        # Verify token
        is_valid, payload = token_manager.verify_token(token)
        if not is_valid:
            await websocket.close(code=4001, reason="Invalid authentication token")
            return

        user_id = payload.get("sub")
        if not user_id:
            await websocket.close(code=4001, reason="Invalid token payload")
            return

        # Accept connection
        await manager.connect(websocket, user_id)

        try:
            while True:
                # Receive message
                data = await websocket.receive_text()
                try:
                    message = json.loads(data)
                except json.JSONDecodeError:
                    await websocket.send_text(
                        json.dumps({"type": "error", "message": "Invalid JSON"})
                    )
                    continue

                action = message.get("action")
                job_id = message.get("job_id")

                if action == "subscribe" and job_id:
                    if manager.subscribe(websocket, job_id):
                        # Send current status
                        job = _job_store.get(job_id)
                        if job:
                            await websocket.send_text(
                                json.dumps(
                                    {
                                        "type": "status",
                                        "job_id": job_id,
                                        "status": job["status"],
                                        "progress": job.get("progress", 0.0),
                                    }
                                )
                            )
                    else:
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "error",
                                    "message": f"Cannot subscribe to job: {job_id}",
                                }
                            )
                        )

                elif action == "unsubscribe" and job_id:
                    manager.unsubscribe(websocket, job_id)
                    await websocket.send_text(
                        json.dumps(
                            {"type": "unsubscribed", "job_id": job_id}
                        )
                    )

                elif action == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))

                else:
                    await websocket.send_text(
                        json.dumps({"type": "error", "message": "Unknown action"})
                    )

        except WebSocketDisconnect:
            manager.disconnect(websocket)


# Utility function to update job status and notify subscribers
async def update_job_status(
    job_id: str, status: JobStatus, progress: float | None = None, message: str | None = None
) -> None:
    """Update job status and broadcast to subscribers.

    Args:
        job_id: Job identifier
        status: New status
        progress: Progress (0.0 to 1.0)
        message: Status message
    """
    if job_id not in _job_store:
        return

    job = _job_store[job_id]
    job["status"] = status

    if progress is not None:
        job["progress"] = progress

    if message is not None:
        job["error_message"] = message

    # Update timestamps
    now = datetime.utcnow().isoformat() + "Z"
    if status == JobStatus.RUNNING and not job.get("started_at"):
        job["started_at"] = now
    elif status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
        job["completed_at"] = now
        if status == JobStatus.COMPLETED:
            job["progress"] = 1.0

    # Broadcast to subscribers
    manager = get_connection_manager()
    await manager.broadcast_status(
        job_id,
        {
            "status": status,
            "progress": job.get("progress", 0.0),
            "message": message,
        },
    )
