"""Hardware backend implementations for GPU control.

Available backends:
    - nvml_backend: NVIDIA GPU control via NVML
    - (future) rocm_backend: AMD GPU control via ROCm
"""

from __future__ import annotations

__all__ = ["nvml_backend"]
