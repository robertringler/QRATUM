"""Sensor manager for HCAL - real-time telemetry collection."""

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class TelemetryReading:
    """Single telemetry reading."""

    device_id: str
    timestamp: datetime
    metrics: Dict[str, Any]


class SensorManager:
    """Manage sensor readings and telemetry."""

    def __init__(self, cache_ttl: float = 1.0):
        """Initialize sensor manager.

        Args:
            cache_ttl: Cache time-to-live in seconds.
        """

        self.cache_ttl = cache_ttl
        self._cache: Dict[str, tuple[TelemetryReading, float]] = {}

    def read_telemetry(
        self, device_id: str, backend: Any, use_cache: bool = True
    ) -> Optional[TelemetryReading]:
        """Read telemetry from device.

        Args:
            device_id: Device identifier.
            backend: Backend driver instance.
            use_cache: Use cached reading if available.

        Returns:
            TelemetryReading instance or None if failed.
        """

        # Check cache
        if use_cache and device_id in self._cache:
            reading, cache_time = self._cache[device_id]
            if time.time() - cache_time < self.cache_ttl:
                return reading

        # Read from backend
        try:
            metrics = backend.read_telemetry(device_id)
            reading = TelemetryReading(
                device_id=device_id, timestamp=datetime.now(), metrics=metrics
            )

            # Update cache
            self._cache[device_id] = (reading, time.time())

            return reading

        except Exception as e:
            print(f"Error reading telemetry from {device_id}: {e}")
            return None

    def read_multi_device(
        self, device_ids: List[str], backend: Any, use_cache: bool = True
    ) -> List[TelemetryReading]:
        """Read telemetry from multiple devices.

        Args:
            device_ids: List of device identifiers.
            backend: Backend driver instance.
            use_cache: Use cached readings if available.

        Returns:
            List of TelemetryReading instances.
        """

        readings = []

        for device_id in device_ids:
            reading = self.read_telemetry(device_id, backend, use_cache)
            if reading:
                readings.append(reading)

        return readings

    def aggregate_metrics(
        self, readings: List[TelemetryReading], metric_name: str
    ) -> Dict[str, Any]:
        """Aggregate metric across multiple readings.

        Args:
            readings: List of telemetry readings.
            metric_name: Metric name to aggregate.

        Returns:
            Dictionary with aggregated statistics.
        """

        values = []

        for reading in readings:
            if metric_name in reading.metrics:
                values.append(reading.metrics[metric_name])

        if not values:
            return {}

        return {
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "count": len(values),
        }

    def clear_cache(self, device_id: Optional[str] = None):
        """Clear telemetry cache.

        Args:
            device_id: Device identifier to clear. If None, clears all.
        """

        if device_id:
            if device_id in self._cache:
                del self._cache[device_id]
        else:
            self._cache.clear()

    def get_cached_reading(self, device_id: str) -> Optional[TelemetryReading]:
        """Get cached reading without updating.

        Args:
            device_id: Device identifier.

        Returns:
            Cached TelemetryReading or None.
        """

        if device_id in self._cache:
            reading, _ = self._cache[device_id]
            return reading
        return None
