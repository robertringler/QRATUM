"""AR/VR streaming adapter for real-time visualization delivery."""

from __future__ import annotations

import json
from typing import Any


def serialize_mesh_fields(mesh: Any, fields: dict[str, Any]) -> bytes:
    """Serialize mesh and fields for network transmission.

    Args:
        mesh: Mesh data (vertices, faces)
        fields: Dictionary of field data

    Returns:
        Serialized bytes payload
    """

    try:
        import numpy as np

        # Convert numpy arrays to lists for JSON serialization
        mesh_data = mesh.tolist() if hasattr(mesh, "tolist") else mesh
        fields_data = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in fields.items()}
    except ImportError:
        mesh_data = mesh
        fields_data = fields

    payload = {"mesh": mesh_data, "fields": fields_data}
    return json.dumps(payload).encode("utf-8")


async def send_ws_payload(ws_url: str, payload: bytes) -> None:
    """Send payload via WebSocket.

    Args:
        ws_url: WebSocket URL
        payload: Bytes payload to send
    """

    try:
        import websockets

        async with websockets.connect(ws_url) as websocket:
            await websocket.send(payload)
    except ImportError:
        # WebSocket library not available - silently skip
        pass


class ARAdapter:
    """Adapter for streaming visualization data to AR/VR clients.

    Provides real-time frame streaming via WebSocket or WebRTC
    to AR/VR headsets and devices.

    Args:
        ws_url: WebSocket URL for streaming endpoint
    """

    def __init__(self, ws_url: str) -> None:
        """Initialize AR adapter."""

        self.ws_url = ws_url
        self._connected = False

    async def send_frame(self, mesh: Any, fields: dict[str, Any]) -> None:
        """Send a frame to connected AR/VR clients.

        Args:
            mesh: Mesh data to stream
            fields: Field data to stream
        """

        payload = serialize_mesh_fields(mesh, fields)
        await send_ws_payload(self.ws_url, payload)

    async def connect(self) -> bool:
        """Establish connection to AR/VR streaming endpoint.

        Returns:
            True if connection successful
        """

        try:
            import websockets

            async with websockets.connect(self.ws_url) as _:
                self._connected = True
                return True
        except Exception:
            self._connected = False
            return False

    @property
    def is_connected(self) -> bool:
        """Check if adapter is connected.

        Returns:
            Connection status
        """

        return self._connected
