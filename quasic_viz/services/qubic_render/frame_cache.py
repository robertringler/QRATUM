"""Frame cache for historical replay and time-series analytics."""

from __future__ import annotations

import time
from collections import deque
from typing import Any


class FrameCache:
    """Cache for rendered frames with historical replay support.

    Args:
        max_frames: Maximum number of frames to retain
    """

    def __init__(self, max_frames: int = 10000) -> None:
        """Initialize frame cache."""

        self._history: deque[dict[str, Any]] = deque(maxlen=max_frames)
        self._max_frames = max_frames

    def cache_frame(self, mesh: Any, fields: dict[str, Any]) -> None:
        """Cache a rendered frame.

        Args:
            mesh: Mesh data
            fields: Field data dictionary
        """

        # Clone data to avoid mutation
        try:
            mesh_copy = mesh.copy() if hasattr(mesh, "copy") else mesh
            fields_copy = {k: v.copy() if hasattr(v, "copy") else v for k, v in fields.items()}
        except Exception:
            mesh_copy = mesh
            fields_copy = fields

        self._history.append(
            {
                "timestamp": time.time(),
                "mesh": mesh_copy,
                "fields": fields_copy,
            }
        )

    def get_frame_at(self, timestamp: float) -> dict[str, Any] | None:
        """Get frame at or after specified timestamp.

        Args:
            timestamp: Unix timestamp

        Returns:
            Frame data dictionary or None if not found
        """

        for frame in self._history:
            if frame["timestamp"] >= timestamp:
                return frame
        return None

    def get_latest(self) -> dict[str, Any] | None:
        """Get most recent frame.

        Returns:
            Most recent frame or None if cache is empty
        """

        if self._history:
            return self._history[-1]
        return None

    def get_range(self, start_time: float, end_time: float) -> list[dict[str, Any]]:
        """Get all frames within time range.

        Args:
            start_time: Start timestamp
            end_time: End timestamp

        Returns:
            List of frame data dictionaries
        """

        return [f for f in self._history if start_time <= f["timestamp"] <= end_time]

    def clear(self) -> None:
        """Clear all cached frames."""

        self._history.clear()

    def __len__(self) -> int:
        """Get number of cached frames."""

        return len(self._history)

    @property
    def oldest_timestamp(self) -> float | None:
        """Get timestamp of oldest frame."""

        if self._history:
            return self._history[0]["timestamp"]
        return None

    @property
    def newest_timestamp(self) -> float | None:
        """Get timestamp of newest frame."""

        if self._history:
            return self._history[-1]["timestamp"]
        return None


# Global frame cache instance for module-level access
FRAME_HISTORY: deque[dict[str, Any]] = deque(maxlen=10000)


def cache_frame(mesh: Any, fields: dict[str, Any]) -> None:
    """Cache a frame to the global history.

    Args:
        mesh: Mesh data
        fields: Field data dictionary
    """

    try:
        mesh_copy = mesh.copy() if hasattr(mesh, "copy") else mesh
        fields_copy = {k: v.copy() if hasattr(v, "copy") else v for k, v in fields.items()}
    except Exception:
        mesh_copy = mesh
        fields_copy = fields

    FRAME_HISTORY.append(
        {
            "timestamp": time.time(),
            "mesh": mesh_copy,
            "fields": fields_copy,
        }
    )


def get_frame_at(timestamp: float) -> dict[str, Any] | None:
    """Get frame from global history at or after timestamp.

    Args:
        timestamp: Unix timestamp

    Returns:
        Frame data or None
    """

    for frame in FRAME_HISTORY:
        if frame["timestamp"] >= timestamp:
            return frame
    return None
