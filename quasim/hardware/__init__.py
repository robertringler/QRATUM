"""Hardware backends for GPU and accelerator management."""

from quasim.hardware.nvml_backend import NVML_AVAILABLE, NVMLBackend

__all__ = ["NVMLBackend", "NVML_AVAILABLE"]
