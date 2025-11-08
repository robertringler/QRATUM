"""Topology discovery for HCAL - automatic hardware detection."""

import platform
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class DeviceType(Enum):
    """Hardware device types."""

    CPU = "cpu"
    GPU = "gpu"
    FPGA = "fpga"
    MEMORY = "memory"
    NETWORK = "network"
    STORAGE = "storage"


class InterconnectType(Enum):
    """Interconnect types."""

    PCIE = "pcie"
    NVLINK = "nvlink"
    INFINIBAND = "infiniband"
    NUMA = "numa"


@dataclass
class Device:
    """Hardware device information."""

    device_id: str
    device_type: DeviceType
    name: str
    vendor: str
    capabilities: Dict[str, any] = field(default_factory=dict)
    numa_node: Optional[int] = None
    pcie_address: Optional[str] = None


@dataclass
class Interconnect:
    """Interconnect between devices."""

    source: str
    destination: str
    interconnect_type: InterconnectType
    bandwidth_gbps: Optional[float] = None


@dataclass
class Topology:
    """Hardware topology information."""

    devices: List[Device]
    interconnects: List[Interconnect]
    numa_nodes: Dict[int, List[str]] = field(default_factory=dict)


class TopologyDiscovery:
    """Discover hardware topology."""

    def __init__(self):
        """Initialize topology discovery."""
        self.topology: Optional[Topology] = None

    def discover(self) -> Topology:
        """Discover hardware topology.

        Returns:
            Topology instance with discovered hardware.
        """
        devices = []
        interconnects = []
        numa_nodes = {}

        # Discover CPUs
        cpu_devices = self._discover_cpus()
        devices.extend(cpu_devices)

        # Discover GPUs
        gpu_devices = self._discover_gpus()
        devices.extend(gpu_devices)

        # Discover FPGAs
        fpga_devices = self._discover_fpgas()
        devices.extend(fpga_devices)

        # Discover interconnects
        interconnects = self._discover_interconnects(devices)

        # Build NUMA topology
        numa_nodes = self._build_numa_topology(devices)

        self.topology = Topology(
            devices=devices, interconnects=interconnects, numa_nodes=numa_nodes
        )
        return self.topology

    def _discover_cpus(self) -> List[Device]:
        """Discover CPU devices.

        Returns:
            List of CPU devices.
        """
        devices = []

        try:
            # Get basic CPU info
            cpu_count = subprocess.check_output(
                ["nproc"], text=True, stderr=subprocess.DEVNULL
            ).strip()

            cpu_info = platform.processor()
            if not cpu_info:
                cpu_info = platform.machine()

            device = Device(
                device_id="cpu0",
                device_type=DeviceType.CPU,
                name=cpu_info,
                vendor="unknown",
                capabilities={
                    "cores": int(cpu_count),
                    "architecture": platform.machine(),
                },
            )
            devices.append(device)

        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to platform module
            device = Device(
                device_id="cpu0",
                device_type=DeviceType.CPU,
                name=platform.processor() or platform.machine(),
                vendor="unknown",
                capabilities={"architecture": platform.machine()},
            )
            devices.append(device)

        return devices

    def _discover_gpus(self) -> List[Device]:
        """Discover GPU devices.

        Returns:
            List of GPU devices.
        """
        devices = []

        try:
            # Try nvidia-smi
            result = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,name,pci.bus_id", "--format=csv,noheader"],
                text=True,
                stderr=subprocess.DEVNULL,
            )

            for line in result.strip().split("\n"):
                if not line:
                    continue

                parts = line.split(", ")
                if len(parts) >= 3:
                    index, name, pci = parts[0], parts[1], parts[2]
                    device = Device(
                        device_id=f"gpu{index}",
                        device_type=DeviceType.GPU,
                        name=name,
                        vendor="NVIDIA",
                        pcie_address=pci,
                        capabilities={"index": int(index)},
                    )
                    devices.append(device)

        except (subprocess.CalledProcessError, FileNotFoundError):
            # nvidia-smi not available
            pass

        return devices

    def _discover_fpgas(self) -> List[Device]:
        """Discover FPGA devices.

        Returns:
            List of FPGA devices.
        """
        devices = []

        try:
            # Try Xilinx xbutil
            subprocess.check_output(
                ["xbutil", "examine"], text=True, stderr=subprocess.DEVNULL
            )

            # Parse FPGA devices (simplified)
            # In a real implementation, this would parse the output more thoroughly
            device = Device(
                device_id="fpga0",
                device_type=DeviceType.FPGA,
                name="Xilinx FPGA",
                vendor="Xilinx",
                capabilities={},
            )
            devices.append(device)

        except (subprocess.CalledProcessError, FileNotFoundError):
            # xbutil not available
            pass

        return devices

    def _discover_interconnects(self, devices: List[Device]) -> List[Interconnect]:
        """Discover interconnects between devices.

        Args:
            devices: List of discovered devices.

        Returns:
            List of interconnects.
        """
        interconnects = []

        # Check for NVLink between GPUs
        gpu_devices = [d for d in devices if d.device_type == DeviceType.GPU]

        if len(gpu_devices) > 1:
            try:
                # Try nvidia-smi to check NVLink
                subprocess.check_output(
                    ["nvidia-smi", "nvlink", "--status"], text=True, stderr=subprocess.DEVNULL
                )

                # If command succeeds, assume NVLink is present
                for i, gpu1 in enumerate(gpu_devices):
                    for gpu2 in gpu_devices[i + 1 :]:
                        interconnect = Interconnect(
                            source=gpu1.device_id,
                            destination=gpu2.device_id,
                            interconnect_type=InterconnectType.NVLINK,
                            bandwidth_gbps=600.0,  # NVLink 3.0 typical
                        )
                        interconnects.append(interconnect)

            except (subprocess.CalledProcessError, FileNotFoundError):
                # No NVLink
                pass

        # Add PCIe interconnects for all PCIe devices
        pcie_devices = [d for d in devices if d.pcie_address]
        for device in pcie_devices:
            interconnect = Interconnect(
                source="cpu0",
                destination=device.device_id,
                interconnect_type=InterconnectType.PCIE,
                bandwidth_gbps=16.0,  # PCIe 4.0 x16 typical
            )
            interconnects.append(interconnect)

        return interconnects

    def _build_numa_topology(self, devices: List[Device]) -> Dict[int, List[str]]:
        """Build NUMA topology map.

        Args:
            devices: List of discovered devices.

        Returns:
            Dictionary mapping NUMA node to device IDs.
        """
        numa_nodes = {}

        # Assign devices to NUMA nodes (simplified)
        for device in devices:
            numa_node = device.numa_node if device.numa_node is not None else 0
            if numa_node not in numa_nodes:
                numa_nodes[numa_node] = []
            numa_nodes[numa_node].append(device.device_id)

        return numa_nodes

    def get_device(self, device_id: str) -> Optional[Device]:
        """Get device by ID.

        Args:
            device_id: Device identifier.

        Returns:
            Device instance or None if not found.
        """
        if not self.topology:
            return None

        for device in self.topology.devices:
            if device.device_id == device_id:
                return device

        return None

    def get_devices_by_type(self, device_type: DeviceType) -> List[Device]:
        """Get devices by type.

        Args:
            device_type: Device type to filter by.

        Returns:
            List of devices of the specified type.
        """
        if not self.topology:
            return []

        return [d for d in self.topology.devices if d.device_type == device_type]
