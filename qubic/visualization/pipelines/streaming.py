"""Real-time streaming visualization pipeline."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from typing import Any, Callable

from qubic.visualization.backends.headless_backend import HeadlessBackend
from qubic.visualization.core.camera import Camera
from qubic.visualization.core.data_model import VisualizationData

logger = logging.getLogger(__name__)


class StreamingPipeline:
    """Pipeline for real-time WebSocket-based visualization streaming.

    Renders visualizations and streams them to connected clients
    via WebSocket protocol.
    """

    def __init__(
        self,
        figsize: tuple[int, int] = (10, 8),
        dpi: int = 100,
        max_fps: int = 10,
    ) -> None:
        """Initialize streaming pipeline.

        Args:
            figsize: Figure size in inches
            dpi: Resolution in dots per inch
            max_fps: Maximum frames per second for streaming
        """

        self.figsize = figsize
        self.dpi = dpi
        self.max_fps = max_fps
        self.frame_delay = 1.0 / max_fps

        # Use headless backend for server-side rendering
        self.backend = HeadlessBackend(figsize=figsize, dpi=dpi)

        self.websocket_connections = set()
        self.is_streaming = False

    async def stream_data(
        self,
        data_generator: Callable[[], VisualizationData],
        scalar_field: str | None = None,
        camera: Camera | None = None,
        colormap: str = "viridis",
    ) -> None:
        """Stream visualization updates to connected clients.

        Args:
            data_generator: Callable that returns updated VisualizationData
            scalar_field: Name of scalar field for color mapping
            camera: Camera settings
            colormap: Colormap name
        """

        self.is_streaming = True
        logger.info(f"Starting streaming at {self.max_fps} FPS")

        try:
            while self.is_streaming:
                # Get updated data
                data = data_generator()

                # Render frame
                self.backend.render(
                    data=data,
                    scalar_field=scalar_field,
                    camera=camera,
                    colormap=colormap,
                )

                # Convert to base64-encoded PNG
                frame_data = self._figure_to_base64()

                # Broadcast to all connected clients
                await self._broadcast({"type": "frame", "data": frame_data})

                # Rate limiting
                await asyncio.sleep(self.frame_delay)

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            self.is_streaming = False

    def _figure_to_base64(self) -> str:
        """Convert current figure to base64-encoded PNG.

        Returns:
            Base64-encoded PNG image data
        """

        buf = io.BytesIO()
        self.backend.fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        return img_base64

    async def _broadcast(self, message: dict[str, Any]) -> None:
        """Broadcast message to all connected clients.

        Args:
            message: Message dictionary to send
        """

        if not self.websocket_connections:
            return

        # Placeholder for WebSocket broadcasting
        # In production, this would use actual WebSocket connections
        logger.debug(f"Broadcasting to {len(self.websocket_connections)} clients")

    def add_connection(self, websocket: Any) -> None:
        """Add WebSocket connection.

        Args:
            websocket: WebSocket connection object
        """

        self.websocket_connections.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.websocket_connections)}")

    def remove_connection(self, websocket: Any) -> None:
        """Remove WebSocket connection.

        Args:
            websocket: WebSocket connection object
        """

        self.websocket_connections.discard(websocket)
        logger.info(f"Client disconnected. Total clients: {len(self.websocket_connections)}")

    def stop_streaming(self) -> None:
        """Stop the streaming loop."""

        logger.info("Stopping streaming")
        self.is_streaming = False

    def create_server(self, host: str = "0.0.0.0", port: int = 8765):
        """Create WebSocket server for streaming.

        Args:
            host: Server host address
            port: Server port

        Returns:
            Server object

        Raises:
            ImportError: If websockets library is not available
        """

        try:
            import websockets
        except ImportError:
            raise ImportError(
                "websockets is required for streaming. Install with: pip install websockets"
            ) from None

        async def handle_connection(websocket, path):
            """Handle WebSocket connection."""

            self.add_connection(websocket)
            try:
                await websocket.wait_closed()
            finally:
                self.remove_connection(websocket)

        logger.info(f"Creating WebSocket server on ws://{host}:{port}")

        return websockets.serve(handle_connection, host, port)
