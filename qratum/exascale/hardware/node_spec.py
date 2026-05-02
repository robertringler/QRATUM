"""
CN-QES Compute Node Specification

Defines the hardware architecture of individual compute nodes in the QRATUM
ExaScale system. Each node contains 16× NVIDIA GB200 NVL72 GPUs and
2× AMD EPYC 9754 CPUs, achieving exceptional compute density.

Node Composition:
- 16× NVIDIA GB200 NVL72 (Blackwell architecture)
- 2× AMD EPYC 9754 (128 cores each, 256 cores total per node)
- 1.5 TB system memory
- NVLink 5.0 @ 1.8 TB/s per GPU
- PCIe Gen 5 @ 128 GB/s
"""

from dataclasses import dataclass
from typing import List, Dict, Any
from enum import Enum


class MemoryType(Enum):
    """Memory hierarchy types"""
    HBM3 = "HBM3"  # GPU memory
    DDR5 = "DDR5"  # System memory
    NVRAM = "NVRAM"  # Non-volatile RAM


@dataclass(frozen=True)
class GPU_GB200_NVL72:
    """
    NVIDIA GB200 NVL72 GPU Specification
    
    Based on Blackwell architecture with unprecedented compute density
    and deterministic execution capabilities.
    
    Performance:
    - Peak FP64: 40 TFLOPS (Tensor Core off)
    - Peak FP32: 80 TFLOPS
    - Peak FP16: 160 TFLOPS
    - Peak FP8: 320 TFLOPS (AI workloads)
    - Memory: 192 GB HBM3e @ 8 TB/s bandwidth
    - TDP: 1000W per GPU
    """
    model: str = "GB200 NVL72"
    architecture: str = "Blackwell"
    fp64_tflops: float = 40.0
    fp32_tflops: float = 80.0
    fp16_tflops: float = 160.0
    fp8_tflops: float = 320.0
    memory_gb: int = 192
    memory_type: MemoryType = MemoryType.HBM3
    memory_bandwidth_tbs: float = 8.0  # TB/s
    nvlink_bandwidth_tbs: float = 1.8  # TB/s per GPU
    tdp_watts: int = 1000
    cuda_cores: int = 18432
    tensor_cores: int = 576
    
    def peak_power_kw(self) -> float:
        """Return peak power in kilowatts"""
        return self.tdp_watts / 1000.0


@dataclass(frozen=True)
class CPU_EPYC_9754:
    """
    AMD EPYC 9754 CPU Specification
    
    High-performance server CPU with exceptional memory bandwidth
    and deterministic execution support.
    
    Performance:
    - Cores: 128 cores (256 threads with SMT)
    - Base frequency: 2.25 GHz
    - Boost frequency: 3.7 GHz
    - L3 Cache: 256 MB
    - Memory: 12 channels DDR5-4800 (up to 6 TB)
    - TDP: 360W
    """
    model: str = "EPYC 9754"
    architecture: str = "Zen 4"
    cores: int = 128
    threads: int = 256
    base_freq_ghz: float = 2.25
    boost_freq_ghz: float = 3.7
    l3_cache_mb: int = 256
    memory_channels: int = 12
    memory_type: MemoryType = MemoryType.DDR5
    memory_bandwidth_gbs: float = 460.8  # GB/s
    tdp_watts: int = 360
    pcie_gen: int = 5
    pcie_lanes: int = 128
    
    def peak_power_kw(self) -> float:
        """Return peak power in kilowatts"""
        return self.tdp_watts / 1000.0


@dataclass
class ComputeNode:
    """
    CN-QES (Compute Node - Quantum ExaScale) Specification
    
    Complete compute node with GPUs, CPUs, memory, and networking.
    Designed for deterministic execution with hardware-level verification.
    
    Attributes:
        node_id: Unique identifier for this node
        gpus: List of GPU instances (16× GB200 NVL72)
        cpus: List of CPU instances (2× EPYC 9754)
        system_memory_gb: Total system memory in GB
        nvlink_topology: NVLink interconnect topology
        aetherfabric_nics: Number of AetherFabric-X NICs
        merkle_verification: Hardware Merkle engine enabled
    """
    node_id: str
    gpus: List[GPU_GB200_NVL72]
    cpus: List[CPU_EPYC_9754]
    system_memory_gb: int
    nvlink_topology: str
    aetherfabric_nics: int
    merkle_verification: bool = True
    
    def __post_init__(self):
        """Validate node configuration"""
        if len(self.gpus) != 16:
            raise ValueError(f"Node must have exactly 16 GPUs, got {len(self.gpus)}")
        if len(self.cpus) != 2:
            raise ValueError(f"Node must have exactly 2 CPUs, got {len(self.cpus)}")
        if self.system_memory_gb < 1500:
            raise ValueError(f"System memory must be ≥ 1500 GB, got {self.system_memory_gb}")
    
    def total_fp64_tflops(self) -> float:
        """Calculate total FP64 performance"""
        return sum(gpu.fp64_tflops for gpu in self.gpus)
    
    def total_fp8_tflops(self) -> float:
        """Calculate total FP8 (AI) performance"""
        return sum(gpu.fp8_tflops for gpu in self.gpus)
    
    def total_memory_gb(self) -> int:
        """Calculate total memory (GPU + system)"""
        gpu_memory = sum(gpu.memory_gb for gpu in self.gpus)
        return gpu_memory + self.system_memory_gb
    
    def peak_power_kw(self) -> float:
        """Calculate peak power consumption in kilowatts"""
        gpu_power = sum(gpu.peak_power_kw() for gpu in self.gpus)
        cpu_power = sum(cpu.peak_power_kw() for cpu in self.cpus)
        # Add 10% for other components (NICs, memory, cooling)
        return (gpu_power + cpu_power) * 1.1
    
    def get_specifications(self) -> Dict[str, Any]:
        """Return comprehensive node specifications"""
        return {
            "node_id": self.node_id,
            "gpu_count": len(self.gpus),
            "gpu_model": self.gpus[0].model if self.gpus else None,
            "cpu_count": len(self.cpus),
            "cpu_model": self.cpus[0].model if self.cpus else None,
            "cpu_cores_total": sum(cpu.cores for cpu in self.cpus),
            "fp64_tflops": self.total_fp64_tflops(),
            "fp8_tflops": self.total_fp8_tflops(),
            "total_memory_gb": self.total_memory_gb(),
            "system_memory_gb": self.system_memory_gb,
            "gpu_memory_gb": sum(gpu.memory_gb for gpu in self.gpus),
            "peak_power_kw": self.peak_power_kw(),
            "nvlink_topology": self.nvlink_topology,
            "aetherfabric_nics": self.aetherfabric_nics,
            "merkle_verification": self.merkle_verification,
        }


class NodeSpecification:
    """
    Factory for creating standard CN-QES node configurations
    """
    
    @staticmethod
    def create_standard_node(node_id: str) -> ComputeNode:
        """
        Create a standard CN-QES compute node
        
        Args:
            node_id: Unique identifier for the node
            
        Returns:
            ComputeNode with standard configuration
        """
        gpus = [GPU_GB200_NVL72() for _ in range(16)]
        cpus = [CPU_EPYC_9754() for _ in range(2)]
        
        return ComputeNode(
            node_id=node_id,
            gpus=gpus,
            cpus=cpus,
            system_memory_gb=1536,  # 1.5 TB
            nvlink_topology="NVSwitch Gen 4",
            aetherfabric_nics=8,  # 8× 800 Gbps NICs
            merkle_verification=True,
        )
    
    @staticmethod
    def calculate_system_performance(num_nodes: int) -> Dict[str, float]:
        """
        Calculate system-wide performance metrics
        
        Args:
            num_nodes: Number of compute nodes
            
        Returns:
            Dictionary with performance metrics
        """
        node = NodeSpecification.create_standard_node("sample")
        
        return {
            "nodes": num_nodes,
            "gpus": num_nodes * 16,
            "cpu_cores": num_nodes * 256,
            "fp64_tflops": num_nodes * node.total_fp64_tflops(),
            "fp64_eflops": (num_nodes * node.total_fp64_tflops()) / 1000.0,
            "fp8_tflops": num_nodes * node.total_fp8_tflops(),
            "fp8_eflops": (num_nodes * node.total_fp8_tflops()) / 1000.0,
            "total_memory_tb": (num_nodes * node.total_memory_gb()) / 1024.0,
            "peak_power_mw": (num_nodes * node.peak_power_kw()) / 1000.0,
        }


# Standard configuration for QRATUM ExaScale deployment
EXASCALE_NODE_COUNT = 3125
EXASCALE_SPECS = NodeSpecification.calculate_system_performance(EXASCALE_NODE_COUNT)
