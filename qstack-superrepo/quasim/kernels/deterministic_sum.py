"""Deterministic summation kernel for demonstration purposes."""

from __future__ import annotations

from typing import Iterable, Sequence

from qstack_core.contracts import Kernel
from qstack_core.registry import GLOBAL_REGISTRY


class DeterministicSumKernel(Kernel):
    name = "deterministic_sum"

    def describe(self) -> str:
        return "Deterministically sums numeric inputs after sorting for stability."

    def run(self, values: Iterable[float] | Sequence[float]) -> float:
        ordered = sorted(values)
        return float(sum(ordered))


GLOBAL_REGISTRY.register_kernel(DeterministicSumKernel())

__all__ = ["DeterministicSumKernel"]
