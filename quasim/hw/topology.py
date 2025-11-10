"""Hardware topology discovery and device information."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class DeviceInfo:
    """Information about a hardware device."""

    device_id: str
    type: str  # gpu, cpu, cryo, etc.
    name: Optional[str] = None
    capabilities: Optional[Dict[str, Any]] = None


class TopologyDiscovery:
    """
    Hardware topology discovery service.

    Discovers and tracks available hardware devices and their capabilities.
    """

    def __init__(self):
        self.devices: Dict[str, DeviceInfo] = {}

    def add_device(
        self,
        device_id: str,
        device_type: str,
        name: Optional[str] = None,
        capabilities: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a device to the topology."""
        self.devices[device_id] = DeviceInfo(
            device_id=device_id, type=device_type, name=name, capabilities=capabilities
        )

    def get_device(self, device_id: str) -> Optional[DeviceInfo]:
        """Get device information by ID."""
        return self.devices.get(device_id)

    def get_devices_by_type(self, device_type: str) -> Dict[str, DeviceInfo]:
        """Get all devices of a specific type."""
        return {
            device_id: info for device_id, info in self.devices.items() if info.type == device_type
        }
