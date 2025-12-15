"""Cross-hardware testing utilities for XENON Bioinformatics.

Provides:
- Hardware detection (CPU, GPU, QPU)
- Conditional test execution
- Cross-hardware validation
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np


class HardwareType(Enum):
    """Types of hardware targets."""
    
    CPU = "cpu"
    GPU_NVIDIA = "gpu_nvidia"
    GPU_AMD = "gpu_amd"
    QPU_SIMULATOR = "qpu_simulator"
    QPU_HARDWARE = "qpu_hardware"


@dataclass
class HardwareInfo:
    """Hardware capability information.
    
    Attributes:
        hardware_type: Type of hardware
        available: Whether hardware is available
        name: Hardware name/model
        compute_capability: Compute capability version
        memory_gb: Available memory in GB
    """
    
    hardware_type: HardwareType
    available: bool
    name: str = ""
    compute_capability: str = ""
    memory_gb: float = 0.0


class HardwareDetector:
    """Detect available hardware for testing.
    
    Provides automatic detection of CPU, GPU, and QPU resources.
    """
    
    def __init__(self):
        """Initialize hardware detector."""
        self.hardware_info: List[HardwareInfo] = []
        self._detect_hardware()
    
    def _detect_hardware(self) -> None:
        """Detect all available hardware."""
        # CPU always available
        self.hardware_info.append(HardwareInfo(
            hardware_type=HardwareType.CPU,
            available=True,
            name="CPU",
        ))
        
        # Detect NVIDIA GPU
        try:
            import torch
            if torch.cuda.is_available():
                self.hardware_info.append(HardwareInfo(
                    hardware_type=HardwareType.GPU_NVIDIA,
                    available=True,
                    name=torch.cuda.get_device_name(0),
                    compute_capability=".".join(map(str, torch.cuda.get_device_capability(0))),
                    memory_gb=torch.cuda.get_device_properties(0).total_memory / 1e9,
                ))
        except (ImportError, Exception):
            self.hardware_info.append(HardwareInfo(
                hardware_type=HardwareType.GPU_NVIDIA,
                available=False,
            ))
        
        # Detect AMD GPU
        try:
            import pyrsmi
            pyrsmi.rocm_smi.initRsmi()
            num_devices = pyrsmi.rocm_smi.listDevices()
            if num_devices:
                self.hardware_info.append(HardwareInfo(
                    hardware_type=HardwareType.GPU_AMD,
                    available=True,
                    name="AMD GPU",
                ))
        except (ImportError, Exception):
            self.hardware_info.append(HardwareInfo(
                hardware_type=HardwareType.GPU_AMD,
                available=False,
            ))
        
        # Detect QPU simulator
        try:
            import qiskit
            self.hardware_info.append(HardwareInfo(
                hardware_type=HardwareType.QPU_SIMULATOR,
                available=True,
                name="Qiskit Aer",
            ))
        except ImportError:
            self.hardware_info.append(HardwareInfo(
                hardware_type=HardwareType.QPU_SIMULATOR,
                available=False,
            ))
    
    def is_available(self, hardware_type: HardwareType) -> bool:
        """Check if hardware type is available.
        
        Args:
            hardware_type: Hardware type to check
            
        Returns:
            True if available
        """
        for info in self.hardware_info:
            if info.hardware_type == hardware_type:
                return info.available
        return False
    
    def get_available_hardware(self) -> List[HardwareType]:
        """Get list of available hardware types.
        
        Returns:
            List of available hardware types
        """
        return [info.hardware_type for info in self.hardware_info if info.available]
    
    def get_info(self, hardware_type: HardwareType) -> Optional[HardwareInfo]:
        """Get hardware information.
        
        Args:
            hardware_type: Hardware type
            
        Returns:
            Hardware info or None
        """
        for info in self.hardware_info:
            if info.hardware_type == hardware_type:
                return info
        return None


def requires_hardware(hardware_type: HardwareType):
    """Decorator to skip tests if hardware not available.
    
    Args:
        hardware_type: Required hardware type
        
    Returns:
        Test decorator
    """
    import pytest
    
    detector = HardwareDetector()
    condition = detector.is_available(hardware_type)
    reason = f"{hardware_type.value} not available"
    
    return pytest.mark.skipif(not condition, reason=reason)
