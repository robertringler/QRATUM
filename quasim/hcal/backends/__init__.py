"""HCAL backend implementations."""

from typing import Any, Dict, Optional


class Backend:
    """Base backend interface."""

    def __init__(self, device_id: str):
        """Initialize backend.

        Args:
            device_id: Device identifier
        """

        self.device_id = device_id

    def get_telemetry(self) -> Dict[str, Any]:
        """Get current device telemetry.

        Returns:
            Dictionary of telemetry data
        """

        return {}

    def apply_setpoint(self, setpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Apply hardware setpoint.

        Args:
            setpoint: Setpoint configuration

        Returns:
            Result dictionary
        """

        return {"success": True}

    def capture_state(self) -> Dict[str, Any]:
        """Capture current hardware state.

        Returns:
            State dictionary
        """

        return {}

    def restore_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Restore hardware state.

        Args:
            state: State to restore

        Returns:
            Result dictionary
        """

        return {"success": True}


__all__ = ["Backend"]
"""HCAL backend drivers."""

from typing import Any, Dict


class BaseBackend:
    """Base class for hardware backends."""

    def __init__(self, dry_run: bool = True):
        """Initialize backend.

        Args:
            dry_run: Enable dry-run mode.
        """

        self.dry_run = dry_run

    def device_exists(self, device_id: str) -> bool:
        """Check if device exists.

        Args:
            device_id: Device identifier.

        Returns:
            True if device exists.
        """

        raise NotImplementedError

    def apply_setpoint(self, device_id: str, setpoint: Dict[str, Any]) -> bool:
        """Apply setpoint to device.

        Args:
            device_id: Device identifier.
            setpoint: Setpoint dictionary.

        Returns:
            True if successful.
        """

        raise NotImplementedError

    def read_configuration(self, device_id: str) -> Dict[str, Any]:
        """Read current device configuration.

        Args:
            device_id: Device identifier.

        Returns:
            Configuration dictionary.
        """

        raise NotImplementedError

    def read_telemetry(self, device_id: str) -> Dict[str, Any]:
        """Read device telemetry.

        Args:
            device_id: Device identifier.

        Returns:
            Telemetry dictionary.
        """

        raise NotImplementedError
