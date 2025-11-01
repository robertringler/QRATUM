"""High-level runtime facade bridging to libquasim."""
from __future__ import annotations

import contextlib
import ctypes
import pathlib
from dataclasses import dataclass
from typing import Iterable, List

LIB_PATH = pathlib.Path(__file__).resolve().parents[2] / "build" / "libquasim" / "libquasim.a"


@dataclass
class Config:
    simulation_precision: str = "fp8"
    max_workspace_mb: int = 16384


class _RuntimeHandle:
    """Tiny wrapper using pure Python to emulate tensor contraction."""

    def __init__(self, config: Config):
        self._config = config
        self._latencies: List[float] = []

    def simulate(self, tensors: Iterable[Iterable[complex]]) -> list[complex]:
        aggregates: list[complex] = []
        for tensor in tensors:
            total = 0 + 0j
            for value in tensor:
                total += complex(value)
            aggregates.append(total)
        self._latencies.append(float(len(aggregates)))
        return aggregates

    @property
    def average_latency(self) -> float:
        if not self._latencies:
            return 0.0
        return float(sum(self._latencies) / len(self._latencies))


@contextlib.contextmanager
def runtime(config: Config):
    handle = _RuntimeHandle(config)
    try:
        yield handle
    finally:
        if LIB_PATH.exists():
            ctypes.cdll.LoadLibrary(str(LIB_PATH))
