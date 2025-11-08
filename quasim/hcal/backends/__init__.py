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
