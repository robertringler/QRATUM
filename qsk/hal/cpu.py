"""CPU abstraction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CPU:
    cores: int = 4
    frequency_ghz: float = 2.5

    def profile(self) -> dict:
        return {"cores": self.cores, "frequency_ghz": self.frequency_ghz}
