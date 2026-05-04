"""
NVLink-Q Interface Specification

Defines the quantum-classical interconnect protocol that bridges PTAQ-1000
quantum processors with GB200 GPU clusters over NVLink 5.0 fabric.

NVLink-Q extends standard NVLink with:
- Quantum state transfer protocols
- Measurement result streaming
- Classical parameter injection
- Error correction data exchange
- Timing synchronization (quantum coherence critical)

Bandwidth: 1.8 TB/s per link (NVLink 5.0 baseline)
Latency: <100 nanoseconds (quantum coherence preservation)
Error rate: <1e-15 (quantum-grade reliability)

Architecture:
┌──────────────┐   NVLink-Q   ┌──────────────┐
│   GB200 GPU  │◀───────────▶│  PTAQ Module │
│              │   1.8 TB/s   │              │
│  Classical   │              │   Quantum    │
│  Processing  │              │   Processing │
└──────────────┘              └──────────────┘
"""

import hashlib
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class NVLinkQProtocol(Enum):
    """NVLink-Q protocol types"""

    STATE_TRANSFER = "state_transfer"  # Quantum state transfer
    MEASUREMENT_STREAM = "measurement_stream"  # Measurement result streaming
    PARAMETER_INJECTION = "parameter_injection"  # Classical parameter injection
    ERROR_CORRECTION = "error_correction"  # Error correction data
    SYNC_PULSE = "sync_pulse"  # Timing synchronization


class TransferDirection(Enum):
    """Data transfer direction"""

    GPU_TO_QUANTUM = "gpu_to_quantum"  # Classical → Quantum
    QUANTUM_TO_GPU = "quantum_to_gpu"  # Quantum → Classical
    BIDIRECTIONAL = "bidirectional"  # Both directions


class DataEncoding(Enum):
    """Data encoding schemes"""

    RAW_BINARY = "raw_binary"  # Raw binary data
    COMPRESSED = "compressed"  # Compressed data
    ERROR_PROTECTED = "error_protected"  # Error-protected with ECC
    QUANTUM_SAFE = "quantum_safe"  # Quantum-safe encoding


class TransferPriority(Enum):
    """Transfer priority levels"""

    CRITICAL = 0  # Critical (timing-sensitive quantum operations)
    HIGH = 1  # High priority
    NORMAL = 2  # Normal priority
    LOW = 3  # Low priority (logging, diagnostics)


@dataclass(frozen=True)
class NVLinkQCapabilities:
    """
    NVLink-Q hardware capabilities

    Attributes:
        bandwidth_tbs: Bandwidth in TB/s
        latency_ns: Latency in nanoseconds
        error_rate: Bit error rate
        max_packet_size_kb: Maximum packet size in KB
        supports_rdma: Remote Direct Memory Access support
        supports_coherence: Cache coherence support
        quantum_timing_sync: Quantum timing synchronization
    """

    bandwidth_tbs: float = 1.8  # 1.8 TB/s (NVLink 5.0)
    latency_ns: float = 100.0  # <100 ns
    error_rate: float = 1e-15  # <1e-15 BER
    max_packet_size_kb: int = 64  # 64 KB packets
    supports_rdma: bool = True
    supports_coherence: bool = True
    quantum_timing_sync: bool = True

    def get_max_throughput_gbs(self) -> float:
        """Get maximum throughput in GB/s"""
        return self.bandwidth_tbs * 1024.0  # TB/s to GB/s

    def estimate_transfer_time_us(self, data_size_mb: float) -> float:
        """
        Estimate transfer time for given data size

        Args:
            data_size_mb: Data size in MB

        Returns:
            Transfer time in microseconds
        """
        data_gb = data_size_mb / 1024.0
        transfer_time_s = data_gb / self.bandwidth_tbs
        transfer_time_us = transfer_time_s * 1_000_000.0

        # Add latency overhead
        return transfer_time_us + (self.latency_ns / 1000.0)


@dataclass
class NVLinkQPacket:
    """
    NVLink-Q packet

    Attributes:
        packet_id: Unique packet identifier
        protocol: Protocol type
        direction: Transfer direction
        source_id: Source device ID (GPU or quantum module)
        dest_id: Destination device ID
        data_size_bytes: Payload size in bytes
        encoding: Data encoding scheme
        priority: Transfer priority
        timestamp_ns: Packet timestamp in nanoseconds
        checksum: Packet checksum (for error detection)
        payload: Packet payload (in real system: pointer to data)
    """

    packet_id: int
    protocol: NVLinkQProtocol
    direction: TransferDirection
    source_id: int
    dest_id: int
    data_size_bytes: int
    encoding: DataEncoding = DataEncoding.ERROR_PROTECTED
    priority: TransferPriority = TransferPriority.NORMAL
    timestamp_ns: int = 0
    checksum: str = ""
    payload: Optional[bytes] = None

    def __post_init__(self):
        """Initialize packet"""
        if self.timestamp_ns == 0:
            object.__setattr__(self, "timestamp_ns", int(time.time() * 1e9))

        if not self.checksum and self.payload:
            object.__setattr__(self, "checksum", self.compute_checksum())

    def compute_checksum(self) -> str:
        """
        Compute packet checksum

        Returns:
            SHA-256 checksum as hex string
        """
        if not self.payload:
            return ""

        # Include metadata in checksum
        metadata = struct.pack(
            "QQQQ",
            self.packet_id,
            self.source_id,
            self.dest_id,
            self.data_size_bytes,
        )

        data = metadata + self.payload
        return hashlib.sha256(data).hexdigest()

    def verify_checksum(self) -> bool:
        """
        Verify packet integrity

        Returns:
            True if checksum matches
        """
        if not self.checksum or not self.payload:
            return False

        computed = self.compute_checksum()
        return computed == self.checksum

    def get_packet_overhead_bytes(self) -> int:
        """Get packet overhead (header + checksum)"""
        # Header: 8 fields × 8 bytes + checksum (32 bytes)
        return 64 + 32


@dataclass
class NVLinkQTransfer:
    """
    NVLink-Q transfer operation

    Represents a complete data transfer between GPU and quantum processor,
    potentially spanning multiple packets.

    Attributes:
        transfer_id: Unique transfer identifier
        protocol: Protocol type
        direction: Transfer direction
        source_id: Source device ID
        dest_id: Destination device ID
        total_size_bytes: Total transfer size in bytes
        packets: List of packets comprising this transfer
        start_time_ns: Transfer start time
        completion_time_ns: Transfer completion time (when complete)
        status: Transfer status
    """

    transfer_id: str
    protocol: NVLinkQProtocol
    direction: TransferDirection
    source_id: int
    dest_id: int
    total_size_bytes: int
    packets: List[NVLinkQPacket] = field(default_factory=list)
    start_time_ns: int = 0
    completion_time_ns: Optional[int] = None
    status: str = "pending"

    def __post_init__(self):
        """Initialize transfer"""
        if self.start_time_ns == 0:
            object.__setattr__(self, "start_time_ns", int(time.time() * 1e9))

    def get_progress(self) -> float:
        """
        Get transfer progress

        Returns:
            Progress as fraction (0.0 to 1.0)
        """
        if not self.packets:
            return 0.0

        transferred_bytes = sum(p.data_size_bytes for p in self.packets)
        return min(1.0, transferred_bytes / self.total_size_bytes)

    def is_complete(self) -> bool:
        """Check if transfer is complete"""
        return self.status == "completed"

    def get_elapsed_time_us(self) -> float:
        """
        Get elapsed transfer time

        Returns:
            Elapsed time in microseconds
        """
        if self.completion_time_ns:
            elapsed_ns = self.completion_time_ns - self.start_time_ns
        else:
            elapsed_ns = int(time.time() * 1e9) - self.start_time_ns

        return elapsed_ns / 1000.0  # Convert to microseconds

    def get_effective_bandwidth_gbs(self) -> float:
        """
        Calculate effective bandwidth

        Returns:
            Bandwidth in GB/s
        """
        if not self.is_complete():
            return 0.0

        elapsed_s = self.get_elapsed_time_us() / 1_000_000.0
        if elapsed_s == 0:
            return 0.0

        size_gb = self.total_size_bytes / (1024.0**3)
        return size_gb / elapsed_s


@dataclass
class NVLinkQInterface:
    """
    NVLink-Q Interface

    Manages data transfer between GPU and quantum processor over NVLink-Q.
    Provides high-level API for quantum-classical communication.

    Attributes:
        interface_id: Unique interface identifier
        gpu_id: Connected GPU device ID
        quantum_module_id: Connected quantum module ID
        capabilities: Interface capabilities
        active_transfers: Currently active transfers
        completed_transfers: Completed transfers
        packet_counter: Global packet counter
        total_bytes_sent: Total bytes sent
        total_bytes_received: Total bytes received
    """

    interface_id: str
    gpu_id: int
    quantum_module_id: int
    capabilities: NVLinkQCapabilities = field(default_factory=NVLinkQCapabilities)
    active_transfers: Dict[str, NVLinkQTransfer] = field(default_factory=dict)
    completed_transfers: Set[str] = field(default_factory=set)
    packet_counter: int = 0
    total_bytes_sent: int = 0
    total_bytes_received: int = 0

    def create_transfer(
        self, protocol: NVLinkQProtocol, direction: TransferDirection, data_size_bytes: int
    ) -> NVLinkQTransfer:
        """
        Create new transfer

        Args:
            protocol: Protocol type
            direction: Transfer direction
            data_size_bytes: Total data size

        Returns:
            NVLinkQTransfer instance
        """
        # Determine source and destination
        if direction == TransferDirection.GPU_TO_QUANTUM:
            source_id = self.gpu_id
            dest_id = self.quantum_module_id
        elif direction == TransferDirection.QUANTUM_TO_GPU:
            source_id = self.quantum_module_id
            dest_id = self.gpu_id
        else:
            source_id = self.gpu_id
            dest_id = self.quantum_module_id

        transfer = NVLinkQTransfer(
            transfer_id=f"transfer_{len(self.completed_transfers)}",
            protocol=protocol,
            direction=direction,
            source_id=source_id,
            dest_id=dest_id,
            total_size_bytes=data_size_bytes,
        )

        return transfer

    def send_data(
        self,
        protocol: NVLinkQProtocol,
        data_size_bytes: int,
        priority: TransferPriority = TransferPriority.NORMAL,
    ) -> str:
        """
        Send data from GPU to quantum processor

        Args:
            protocol: Protocol type
            data_size_bytes: Data size in bytes
            priority: Transfer priority

        Returns:
            Transfer ID
        """
        # Create transfer
        transfer = self.create_transfer(
            protocol=protocol,
            direction=TransferDirection.GPU_TO_QUANTUM,
            data_size_bytes=data_size_bytes,
        )

        # Packetize data
        max_packet_size = self.capabilities.max_packet_size_kb * 1024
        remaining_bytes = data_size_bytes

        while remaining_bytes > 0:
            packet_size = min(remaining_bytes, max_packet_size)

            packet = NVLinkQPacket(
                packet_id=self.packet_counter,
                protocol=protocol,
                direction=TransferDirection.GPU_TO_QUANTUM,
                source_id=self.gpu_id,
                dest_id=self.quantum_module_id,
                data_size_bytes=packet_size,
                priority=priority,
            )

            transfer.packets.append(packet)
            self.packet_counter += 1
            remaining_bytes -= packet_size

        # Register transfer
        transfer.status = "transmitting"
        self.active_transfers[transfer.transfer_id] = transfer
        self.total_bytes_sent += data_size_bytes

        return transfer.transfer_id

    def receive_data(self, protocol: NVLinkQProtocol, expected_size_bytes: int) -> str:
        """
        Receive data from quantum processor to GPU

        Args:
            protocol: Protocol type
            expected_size_bytes: Expected data size

        Returns:
            Transfer ID
        """
        # Create transfer
        transfer = self.create_transfer(
            protocol=protocol,
            direction=TransferDirection.QUANTUM_TO_GPU,
            data_size_bytes=expected_size_bytes,
        )

        # Register transfer
        transfer.status = "receiving"
        self.active_transfers[transfer.transfer_id] = transfer
        self.total_bytes_received += expected_size_bytes

        return transfer.transfer_id

    def complete_transfer(self, transfer_id: str) -> bool:
        """
        Mark transfer as completed

        Args:
            transfer_id: Transfer ID to complete

        Returns:
            True if transfer was active
        """
        if transfer_id not in self.active_transfers:
            return False

        transfer = self.active_transfers[transfer_id]
        transfer.status = "completed"
        transfer.completion_time_ns = int(time.time() * 1e9)

        # Move to completed
        del self.active_transfers[transfer_id]
        self.completed_transfers.add(transfer_id)

        return True

    def send_quantum_parameters(self, num_parameters: int, parameter_size_bytes: int = 8) -> str:
        """
        Send classical parameters for variational quantum algorithm

        Args:
            num_parameters: Number of parameters
            parameter_size_bytes: Size per parameter (default: 8 bytes = double)

        Returns:
            Transfer ID
        """
        total_size = num_parameters * parameter_size_bytes
        return self.send_data(
            protocol=NVLinkQProtocol.PARAMETER_INJECTION,
            data_size_bytes=total_size,
            priority=TransferPriority.HIGH,
        )

    def receive_measurement_results(self, num_measurements: int, result_size_bytes: int = 8) -> str:
        """
        Receive quantum measurement results

        Args:
            num_measurements: Number of measurements
            result_size_bytes: Size per result (default: 8 bytes)

        Returns:
            Transfer ID
        """
        total_size = num_measurements * result_size_bytes
        return self.receive_data(
            protocol=NVLinkQProtocol.MEASUREMENT_STREAM,
            expected_size_bytes=total_size,
        )

    def synchronize(self) -> bool:
        """
        Synchronize quantum-classical timing

        Returns:
            True if synchronization successful
        """
        # Send sync pulse
        transfer_id = self.send_data(
            protocol=NVLinkQProtocol.SYNC_PULSE,
            data_size_bytes=64,  # Minimal sync packet
            priority=TransferPriority.CRITICAL,
        )

        # Simulate sync completion
        self.complete_transfer(transfer_id)

        return True

    def get_interface_statistics(self) -> Dict[str, Any]:
        """
        Get interface statistics

        Returns:
            Dictionary of statistics
        """
        active_count = len(self.active_transfers)
        completed_count = len(self.completed_transfers)

        # Calculate average bandwidth
        avg_bandwidth_gbs = 0.0
        if completed_count > 0:
            bandwidths = []
            for transfer_id in self.completed_transfers:
                # Would need to store completed transfers for accurate stats
                # For now, estimate based on capabilities
                bandwidths.append(self.capabilities.get_max_throughput_gbs())
            avg_bandwidth_gbs = sum(bandwidths) / len(bandwidths) if bandwidths else 0.0

        return {
            "interface_id": self.interface_id,
            "gpu_id": self.gpu_id,
            "quantum_module_id": self.quantum_module_id,
            "bandwidth_tbs": self.capabilities.bandwidth_tbs,
            "latency_ns": self.capabilities.latency_ns,
            "active_transfers": active_count,
            "completed_transfers": completed_count,
            "total_bytes_sent": self.total_bytes_sent,
            "total_bytes_received": self.total_bytes_received,
            "total_packets": self.packet_counter,
            "avg_bandwidth_gbs": avg_bandwidth_gbs,
        }


@dataclass
class NVLinkQFabric:
    """
    NVLink-Q Fabric

    Manages multiple NVLink-Q interfaces connecting GPU cluster to
    quantum processor array.

    Attributes:
        fabric_id: Unique fabric identifier
        interfaces: Dictionary of interface_id -> NVLinkQInterface
        topology: Fabric topology description
    """

    fabric_id: str
    interfaces: Dict[str, NVLinkQInterface] = field(default_factory=dict)
    topology: str = "all-to-all"

    def create_interface(self, gpu_id: int, quantum_module_id: int) -> str:
        """
        Create NVLink-Q interface

        Args:
            gpu_id: GPU device ID
            quantum_module_id: Quantum module ID

        Returns:
            Interface ID
        """
        interface_id = f"nvlink_q_{gpu_id}_{quantum_module_id}"

        interface = NVLinkQInterface(
            interface_id=interface_id,
            gpu_id=gpu_id,
            quantum_module_id=quantum_module_id,
        )

        self.interfaces[interface_id] = interface
        return interface_id

    def get_interface(self, gpu_id: int, quantum_module_id: int) -> Optional[NVLinkQInterface]:
        """
        Get interface for GPU-quantum pair

        Args:
            gpu_id: GPU device ID
            quantum_module_id: Quantum module ID

        Returns:
            NVLinkQInterface if exists, None otherwise
        """
        interface_id = f"nvlink_q_{gpu_id}_{quantum_module_id}"
        return self.interfaces.get(interface_id)

    def synchronize_all(self) -> Dict[str, bool]:
        """
        Synchronize all interfaces

        Returns:
            Dictionary of interface_id -> sync_success
        """
        results = {}
        for interface_id, interface in self.interfaces.items():
            results[interface_id] = interface.synchronize()
        return results

    def get_fabric_statistics(self) -> Dict[str, Any]:
        """
        Get fabric-wide statistics

        Returns:
            Dictionary of statistics
        """
        total_interfaces = len(self.interfaces)
        total_active = sum(len(i.active_transfers) for i in self.interfaces.values())
        total_completed = sum(len(i.completed_transfers) for i in self.interfaces.values())
        total_bytes_sent = sum(i.total_bytes_sent for i in self.interfaces.values())
        total_bytes_received = sum(i.total_bytes_received for i in self.interfaces.values())

        return {
            "fabric_id": self.fabric_id,
            "topology": self.topology,
            "total_interfaces": total_interfaces,
            "active_transfers": total_active,
            "completed_transfers": total_completed,
            "total_bytes_sent": total_bytes_sent,
            "total_bytes_received": total_bytes_received,
            "aggregate_bandwidth_tbs": total_interfaces * 1.8,  # 1.8 TB/s per interface
        }


class NVLinkQInterfaceBuilder:
    """
    Builder for NVLink-Q fabric configurations
    """

    @staticmethod
    def create_standard_fabric(
        num_gpus: int, num_quantum_modules: int, fabric_id: str = "nvlink_q_fabric_0"
    ) -> NVLinkQFabric:
        """
        Create standard NVLink-Q fabric

        Args:
            num_gpus: Number of GPUs
            num_quantum_modules: Number of quantum modules
            fabric_id: Fabric identifier

        Returns:
            NVLinkQFabric instance
        """
        fabric = NVLinkQFabric(fabric_id=fabric_id)

        # Create interfaces (one-to-one mapping for simplicity)
        for i in range(min(num_gpus, num_quantum_modules)):
            fabric.create_interface(gpu_id=i, quantum_module_id=i)

        return fabric

    @staticmethod
    def create_full_mesh_fabric(
        num_gpus: int, num_quantum_modules: int, fabric_id: str = "nvlink_q_fabric_mesh"
    ) -> NVLinkQFabric:
        """
        Create full mesh NVLink-Q fabric (all-to-all connectivity)

        Args:
            num_gpus: Number of GPUs
            num_quantum_modules: Number of quantum modules
            fabric_id: Fabric identifier

        Returns:
            NVLinkQFabric instance
        """
        fabric = NVLinkQFabric(fabric_id=fabric_id, topology="full_mesh")

        # Create all possible interfaces
        for gpu_id in range(num_gpus):
            for module_id in range(num_quantum_modules):
                fabric.create_interface(gpu_id=gpu_id, quantum_module_id=module_id)

        return fabric
