"""Telemetry for HAL components."""

from __future__ import annotations

from typing import Dict

from .cpu import CPU
from .fpga import FPGA
from .gpu import GPU


def collect(cpu: CPU, gpu: GPU, fpga: FPGA) -> Dict[str, dict]:
    return {"cpu": cpu.profile(), "gpu": gpu.profile(), "fpga": fpga.profile()}
