"""FPGA abstraction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FPGA:
    logic_cells: int = 100000
    clock_mhz: float = 250.0

    def profile(self) -> dict:
        return {"logic_cells": self.logic_cells, "clock_mhz": self.clock_mhz}
