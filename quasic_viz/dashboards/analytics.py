"""Time-series analytics for historical replay."""

from __future__ import annotations

import time
from collections import deque
from typing import Any


class TimeSeriesAnalytics:
    """Analytics engine for time-series visualization data.

    Provides historical replay and time-based queries.

    Args:
        max_history: Maximum number of frames to retain
    """

    def __init__(self, max_history: int = 10000) -> None:
        """Initialize analytics engine."""
        self._history: deque[dict[str, Any]] = deque(maxlen=max_history)
        self._start_time: float | None = None

    def record_frame(self, mesh: Any, fields: dict[str, Any]) -> None:
        """Record a frame to history.

        Args:
            mesh: Mesh data
            fields: Field data dictionary
        """
        if self._start_time is None:
            self._start_time = time.time()

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

    def get_frame_range(self, start_time: float, end_time: float) -> list[dict[str, Any]]:
        """Get all frames within time range.

        Args:
            start_time: Start timestamp
            end_time: End timestamp

        Returns:
            List of frame data dictionaries
        """
        return [f for f in self._history if start_time <= f["timestamp"] <= end_time]

    def get_latest_frames(self, count: int = 100) -> list[dict[str, Any]]:
        """Get most recent frames.

        Args:
            count: Number of frames to retrieve

        Returns:
            List of frame data dictionaries
        """
        return list(self._history)[-count:]

    def compute_field_statistics(
        self, field_name: str, frames: list[dict[str, Any]] | None = None
    ) -> dict[str, float]:
        """Compute statistics for a field over time.

        Args:
            field_name: Name of field to analyze
            frames: Frames to analyze (defaults to all history)

        Returns:
            Statistics dictionary with min, max, mean, std
        """
        import numpy as np

        if frames is None:
            frames = list(self._history)

        values = []
        for frame in frames:
            if field_name in frame.get("fields", {}):
                field_data = frame["fields"][field_name]
                if hasattr(field_data, "flatten"):
                    values.extend(field_data.flatten().tolist())
                elif hasattr(field_data, "__iter__"):
                    values.extend(list(field_data))

        if not values:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}

        arr = np.array(values)
        return {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
        }

    @property
    def frame_count(self) -> int:
        """Get total number of recorded frames."""
        return len(self._history)

    @property
    def duration_seconds(self) -> float:
        """Get total duration of recorded history."""
        if not self._history:
            return 0.0
        return self._history[-1]["timestamp"] - self._history[0]["timestamp"]

    def clear(self) -> None:
        """Clear all recorded history."""
        self._history.clear()
        self._start_time = None
