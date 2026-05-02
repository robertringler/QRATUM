"""
QRATUM ExaScale Hardware Specifications

Defines the CN-QES (Compute Node - Quantum ExaScale) node architecture,
system topology, and thermal management for the exascale deployment.

Components:
- node_spec: Individual compute node specifications
- topology: System-wide topology and interconnect layout
- cooling: Thermal management and power density control
"""

from .cooling import CoolingStrategy, ThermalManagement
from .node_spec import (CPU_EPYC_9754, GPU_GB200_NVL72, ComputeNode,
                        NodeSpecification)
from .topology import FailureDomain, RackLayout, SystemTopology

__all__ = [
    "ComputeNode",
    "NodeSpecification",
    "GPU_GB200_NVL72",
    "CPU_EPYC_9754",
    "SystemTopology",
    "RackLayout",
    "FailureDomain",
    "ThermalManagement",
    "CoolingStrategy",
]
