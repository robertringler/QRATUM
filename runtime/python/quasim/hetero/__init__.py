"""Heterogeneous execution layer for CPU, GPU, NPU, and TPU backends.

Supports SYCL, Kokkos, and OpenMP offload for dynamic workload scheduling.
"""
from __future__ import annotations

from .scheduler import HeteroScheduler, Device, DeviceType
from .workload import Workload, WorkloadType

__all__ = [
    "HeteroScheduler",
    "Device",
    "DeviceType",
    "Workload",
    "WorkloadType",
]
