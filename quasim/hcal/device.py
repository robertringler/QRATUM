"""Device discovery and management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Device:
    """Hardware device representation.

    Attributes:
        id: Unique device identifier
        name: Human-readable device name
        type: Device type (e.g., 'quantum_processor', 'controller')
        status: Current device status
        properties: Additional device properties
    """

    id: str
    name: str
    type: str
    status: str = "unknown"
    properties: dict[str, Any] = field(default_factory=dict)


class DeviceManager:
    """Manages hardware device discovery and lifecycle."""

    def __init__(self) -> None:
        """Initialize device manager."""

        self._devices: dict[str, Device] = {}

    def discover(self) -> list[Device]:
        """Discover available hardware devices.

        Returns:
            List of discovered devices
        """

        # In a real implementation, this would query hardware
        # For now, return mock devices for testing
        mock_devices = [
            Device(
                id="dev001",
                name="Quantum Processor 1",
                type="quantum_processor",
                status="online",
                properties={"qubits": 20, "backend": "superconducting"},
            ),
            Device(
                id="dev002",
                name="Control Unit 1",
                type="controller",
                status="online",
                properties={"channels": 8},
            ),
        ]

        for device in mock_devices:
            self._devices[device.id] = device

        return mock_devices

    def get_device(self, device_id: str) -> Device | None:
        """Get device by ID.

        Args:
            device_id: Device identifier

        Returns:
            Device if found, None otherwise
        """

        return self._devices.get(device_id)

    def list_devices(self) -> list[Device]:
        """List all managed devices.

        Returns:
            List of all devices
        """

        return list(self._devices.values())
