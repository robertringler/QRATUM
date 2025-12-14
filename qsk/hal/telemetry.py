"""Telemetry for HAL components."""
from __future__ import annotations


from .cpu import CPU
from .gpu import GPU
from .fpga import FPGA


def collect(cpu: CPU, gpu: GPU, fpga: FPGA) -> dict[str, dict]:
    return {"cpu": cpu.profile(), "gpu": gpu.profile(), "fpga": fpga.profile()}
