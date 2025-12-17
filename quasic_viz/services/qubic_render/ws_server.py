"""WebSocket server for real-time visualization streaming."""

from __future__ import annotations

from typing import Any

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None
    WebSocket = None
    WebSocketDisconnect = None

from .frame_cache import FrameCache


class WebSocketServer:
    """WebSocket server for real-time frame streaming.

    Args:
        frame_cache: Frame cache instance for historical replay
    """

    def __init__(self, frame_cache: FrameCache | None = None) -> None:
        """Initialize WebSocket server."""

        self._clients: set[Any] = set()
        self.frame_cache = frame_cache or FrameCache()
        self.app = self._create_app() if FASTAPI_AVAILABLE else None

    def _create_app(self) -> Any:
        """Create FastAPI application.

        Returns:
            FastAPI app instance
        """

        if not FASTAPI_AVAILABLE:
            return None

        app = FastAPI(
            title="QuASIC Render WebSocket Server",
            description="Real-time frame streaming server",
            version="0.1.0",
        )

        @app.get("/health")
        async def health_check() -> dict[str, Any]:
            """Health check endpoint."""

            return {
                "status": "healthy",
                "clients_connected": len(self._clients),
                "frames_cached": len(self.frame_cache),
            }

        @app.websocket("/ws/stream")
        async def stream_ws(ws: WebSocket) -> None:
            """WebSocket streaming endpoint."""

            await ws.accept()
            self._clients.add(ws)
            try:
                while True:
                    # Wait for commands or send frames
                    data = await ws.receive_text()
                    if data == "get_latest":
                        frame = self.frame_cache.get_latest()
                        if frame:
                            await ws.send_json(frame)
            except WebSocketDisconnect:
                pass
            finally:
                self._clients.discard(ws)

        @app.get("/frames/latest")
        async def get_latest_frame() -> dict[str, Any] | None:
            """Get latest cached frame."""

            return self.frame_cache.get_latest()

        @app.get("/frames/{timestamp}")
        async def get_frame_at(timestamp: float) -> dict[str, Any] | None:
            """Get frame at timestamp."""

            return self.frame_cache.get_frame_at(timestamp)

        return app

    async def broadcast(self, frame_data: dict[str, Any]) -> None:
        """Broadcast frame to all connected clients.

        Args:
            frame_data: Frame data to broadcast
        """

        if not self._clients:
            return

        import asyncio

        await asyncio.gather(
            *[c.send_json(frame_data) for c in self._clients], return_exceptions=True
        )

    @property
    def client_count(self) -> int:
        """Get number of connected clients."""

        return len(self._clients)


def create_ws_app(frame_cache: FrameCache | None = None) -> Any:
    """Create WebSocket server application.

    Args:
        frame_cache: Optional frame cache instance

    Returns:
        FastAPI app or None if FastAPI not available
    """

    server = WebSocketServer(frame_cache)
    return server.app
