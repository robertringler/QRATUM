"""AMD ROCm backend stub for HCAL."""

from typing import Any, Dict

from quasim.hcal.backends import BaseBackend


class AmdRocmBackend(BaseBackend):
    """AMD ROCm backend driver (stub for future implementation)."""

    def __init__(self, dry_run: bool = True):
        """Initialize AMD ROCm backend.

        Args:
            dry_run: Enable dry-run mode.
        """

        super().__init__(dry_run)

    def device_exists(self, device_id: str) -> bool:
        """Check if device exists."""

        return False

    def apply_setpoint(self, device_id: str, setpoint: Dict[str, Any]) -> bool:
        """Apply setpoint to device."""

        print("AMD ROCm backend not yet implemented")
        return False

    def read_configuration(self, device_id: str) -> Dict[str, Any]:
        """Read current device configuration."""

        return {}

    def read_telemetry(self, device_id: str) -> Dict[str, Any]:
        """Read device telemetry."""

        return {}
