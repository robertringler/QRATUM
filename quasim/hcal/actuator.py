"""HCAL actuator for hardware control."""

import time
from typing import Any, Dict, List, Optional


class Actuator:
    """Hardware actuator with baseline capture and rollback."""

    def __init__(self, enable_actuation: bool = False):
        """Initialize actuator.

        Args:
            enable_actuation: Whether to enable actual hardware changes
        """
        self.enable_actuation = enable_actuation
        self._baselines: Dict[str, Dict[str, Any]] = {}
        self._backends: Dict[str, Any] = {}

    def capture_baseline(self, device_id: str) -> Dict[str, Any]:
        """Capture baseline state for a device.

        Args:
            device_id: Device identifier

        Returns:
            Captured baseline state
        """
        baseline = {
            "device_id": device_id,
            "timestamp": time.time(),
            "state": {},
        }

        # Get backend if available
        backend = self._backends.get(device_id)
        if backend:
            baseline["state"] = backend.capture_state()

        self._baselines[device_id] = baseline
        return baseline

    def rollback_device(self, device_id: str) -> Dict[str, Any]:
        """Rollback device to baseline state.

        Args:
            device_id: Device identifier

        Returns:
            Rollback result
        """
        baseline = self._baselines.get(device_id)
        if not baseline:
            return {
                "success": False,
                "error": f"No baseline captured for {device_id}",
            }

        # Restore state if backend available
        backend = self._backends.get(device_id)
        if backend and self.enable_actuation:
            return backend.restore_state(baseline["state"])

        return {
            "success": True,
            "device_id": device_id,
            "dry_run": not self.enable_actuation,
        }

    def apply_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a plan to hardware.

        Args:
            plan: Plan to apply

        Returns:
            Application result
        """
        result = {
            "plan_id": plan.get("plan_id", "unknown"),
            "actuation_enabled": self.enable_actuation,
            "devices": {},
        }

        devices = plan.get("devices", {})
        for device_id, setpoints in devices.items():
            backend = self._backends.get(device_id)

            if backend and self.enable_actuation:
                device_result = backend.apply_setpoint(setpoints)
            else:
                device_result = {
                    "success": True,
                    "dry_run": True,
                    "setpoints": setpoints,
                }

            result["devices"][device_id] = device_result

        return result

    def register_backend(self, device_id: str, backend: Any) -> None:
        """Register a backend for a device.

        Args:
            device_id: Device identifier
            backend: Backend instance
        """
        self._backends[device_id] = backend

    def get_telemetry(self, device_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get telemetry from devices.

        Args:
            device_ids: List of device IDs, or None for all

        Returns:
            Telemetry data
        """
        telemetry = {}

        if device_ids is None:
            device_ids = list(self._backends.keys())

        for device_id in device_ids:
            backend = self._backends.get(device_id)
            if backend:
                telemetry[device_id] = backend.get_telemetry()

        return telemetry

    def emergency_stop(self) -> Dict[str, Any]:
        """Emergency stop all devices.

        Returns:
            Stop result
        """
        result = {
            "timestamp": time.time(),
            "devices_stopped": list(self._backends.keys()),
        }

        # In a real implementation, this would stop all operations
        # and potentially restore safe defaults

        return result


__all__ = ["Actuator"]
