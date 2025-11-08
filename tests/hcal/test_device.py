"""Tests for device management."""

from __future__ import annotations

import pytest

from quasim.hcal.device import Device, DeviceManager


class TestDevice:
    """Tests for Device class."""

    def test_device_creation(self) -> None:
        """Test creating a device."""
        device = Device(
            id="test001",
            name="Test Device",
            type="test_type",
            status="online",
            properties={"key": "value"},
        )

        assert device.id == "test001"
        assert device.name == "Test Device"
        assert device.type == "test_type"
        assert device.status == "online"
        assert device.properties == {"key": "value"}

    def test_device_default_status(self) -> None:
        """Test device with default status."""
        device = Device(
            id="test002",
            name="Test Device 2",
            type="test_type",
        )

        assert device.status == "unknown"
        assert device.properties == {}


class TestDeviceManager:
    """Tests for DeviceManager class."""

    def test_device_manager_init(self) -> None:
        """Test device manager initialization."""
        manager = DeviceManager()
        assert manager.list_devices() == []

    def test_discover_devices(self) -> None:
        """Test device discovery."""
        manager = DeviceManager()
        devices = manager.discover()

        assert len(devices) >= 2
        assert all(isinstance(d, Device) for d in devices)
        assert all(d.status == "online" for d in devices)

    def test_get_device(self) -> None:
        """Test getting a device by ID."""
        manager = DeviceManager()
        manager.discover()

        device = manager.get_device("dev001")
        assert device is not None
        assert device.id == "dev001"
        assert device.name == "Quantum Processor 1"

    def test_get_nonexistent_device(self) -> None:
        """Test getting a nonexistent device."""
        manager = DeviceManager()
        device = manager.get_device("nonexistent")
        assert device is None

    def test_list_devices(self) -> None:
        """Test listing all devices."""
        manager = DeviceManager()
        manager.discover()

        devices = manager.list_devices()
        assert len(devices) >= 2
        device_ids = [d.id for d in devices]
        assert "dev001" in device_ids
        assert "dev002" in device_ids


@pytest.mark.requires_hardware
class TestDeviceHardware:
    """Tests that require actual hardware (skipped in mock mode)."""

    def test_hardware_connection(self) -> None:
        """Test connecting to real hardware."""
        # This test would only run with actual hardware
        pytest.skip("Hardware not available")
