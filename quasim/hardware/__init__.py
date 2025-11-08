"""QuASIM Hardware Control Module.

Provides backends for GPU reconfiguration and hardware management.
Supports NVIDIA (NVML), AMD (ROCm), and other hardware interfaces.
"""

from __future__ import annotations

__all__ = ["backends"]
"""Hardware backends for GPU and accelerator management."""

from quasim.hardware.nvml_backend import NVML_AVAILABLE, NVMLBackend

__all__ = ["NVMLBackend", "NVML_AVAILABLE"]
