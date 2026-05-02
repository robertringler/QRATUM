"""Adapter layer for the QuASIM runtime."""

from __future__ import annotations

from dataclasses import dataclass

from qstack.config import QuASIMConfig
from quasim import Config as QuasimRuntimeConfig
from quasim import Runtime as QuasimRuntime


@dataclass
class QuASIMAdapter:
    """Deterministic wrapper around the QuASIM runtime."""

    config: QuASIMConfig

    def _build_runtime_config(self) -> QuasimRuntimeConfig:
        return QuasimRuntimeConfig(
            simulation_precision=self.config.simulation_precision,
            max_workspace_mb=self.config.max_workspace_mb,
            backend=self.config.backend,
            seed=self.config.seed,
        )

    def simulate_circuit(self, circuit: list[list[complex]]) -> list[complex]:
        runtime_config = self._build_runtime_config()
        with QuasimRuntime(runtime_config) as runtime:
            return runtime.simulate(circuit)

    def simulate_batch(self, circuits: list[list[list[complex]]]) -> list[list[complex]]:
        runtime_config = self._build_runtime_config()
        results: list[list[complex]] = []
        with QuasimRuntime(runtime_config) as runtime:
            for circuit in circuits:
                results.append(runtime.simulate(circuit))
        return results
