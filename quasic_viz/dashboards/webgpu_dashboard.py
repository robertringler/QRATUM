"""Multi-user real-time WebGPU dashboard."""

from __future__ import annotations

import json
from typing import Any

# Dashboard configuration
TARGET_FPS = 30
FRAME_INTERVAL = 1.0 / TARGET_FPS  # ~0.033 seconds

try:
    import asyncio

    from fastapi import FastAPI, WebSocket, WebSocketDisconnect

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None
    WebSocket = None
    WebSocketDisconnect = None

# Global clients set for broadcasting
_clients: set[Any] = set()

# Global frame data storage
_latest_frame: dict[str, Any] = {"mesh": [], "fields": {}}


def get_latest_frame_json() -> str:
    """Get the latest frame data as JSON.

    Returns:
        JSON string of latest frame data
    """
    return json.dumps(_latest_frame)


def update_frame_data(mesh: Any, fields: dict[str, Any]) -> None:
    """Update the global frame data.

    Args:
        mesh: Mesh data to store
        fields: Field data to store
    """
    global _latest_frame
    # Convert numpy arrays to lists for JSON serialization
    try:
        import numpy as np

        mesh_data = mesh.tolist() if isinstance(mesh, np.ndarray) else mesh
        fields_data = {
            k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in fields.items()
        }
    except ImportError:
        mesh_data = mesh
        fields_data = fields

    _latest_frame = {"mesh": mesh_data, "fields": fields_data}


def create_dashboard_app() -> Any:
    """Create FastAPI application for WebGPU dashboard.

    Returns:
        FastAPI app instance or None if FastAPI not available
    """
    if not FASTAPI_AVAILABLE:
        return None

    app = FastAPI(
        title="QuASIC WebGPU Dashboard",
        description="Multi-user real-time visualization dashboard",
        version="0.1.0",
    )

    @app.get("/health")
    async def health_check() -> dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "clients_connected": len(_clients),
        }

    @app.websocket("/ws/dashboard")
    async def dashboard_ws(ws: WebSocket) -> None:
        """WebSocket endpoint for real-time dashboard updates.

        Args:
            ws: WebSocket connection
        """
        await ws.accept()
        _clients.add(ws)
        try:
            while True:
                await asyncio.sleep(FRAME_INTERVAL)
                frame_data = get_latest_frame_json()
                await asyncio.gather(
                    *[c.send_text(frame_data) for c in _clients],
                    return_exceptions=True,
                )
        except WebSocketDisconnect:
            pass
        finally:
            _clients.discard(ws)

    @app.get("/frame")
    async def get_current_frame() -> dict[str, Any]:
        """Get current frame data via REST.

        Returns:
            Current frame data
        """
        return _latest_frame

    return app


async def broadcast_frame(mesh: Any, fields: dict[str, Any]) -> None:
    """Broadcast frame to all connected clients.

    Args:
        mesh: Mesh data to broadcast
        fields: Field data to broadcast
    """
    update_frame_data(mesh, fields)
    frame_data = get_latest_frame_json()

    if _clients:
        import asyncio

        await asyncio.gather(
            *[c.send_text(frame_data) for c in _clients], return_exceptions=True
        )
