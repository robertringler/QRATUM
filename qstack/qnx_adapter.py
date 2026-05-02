"""Adapter layer wrapping the QNX substrate for deterministic orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from qnx.core import QNXSubstrate
from qnx.types import SecurityLevel, SimulationConfig, SubstrateResult
from qstack.config import QNXConfig


def _resolve_security_level(level: str | SecurityLevel) -> SecurityLevel:
    """Map string-based configuration to the QNX SecurityLevel enum."""

    if isinstance(level, SecurityLevel):
        return level

    normalized = str(level).lower()
    for candidate in SecurityLevel:
        if candidate.value == normalized:
            return candidate

    # Default deterministically to lowest level if an unknown string is supplied.
    return SecurityLevel.LOW


@dataclass
class QNXAdapter:
    """Thin wrapper over :class:`qnx.core.QNXSubstrate`."""

    config: QNXConfig

    def _build_sim_config(self) -> SimulationConfig:
        """Create a QNX :class:`SimulationConfig` from the adapter config."""

        return SimulationConfig(
            scenario_id=self.config.scenario_id,
            timesteps=self.config.timesteps,
            seed=self.config.seed,
            backend=self.config.backend,
            parameters=None,
            security_level=_resolve_security_level(self.config.security_level),
        )

    def run_lifecycle(self, rtos: Any | None = None) -> Mapping[str, Any]:
        """Run the full lifecycle using the QNX substrate."""

        substrate = QNXSubstrate()
        sim_config = self._build_sim_config()
        return substrate.lifecycle_run(sim_config, rtos=rtos)

    def run_simulation(self) -> SubstrateResult:
        """Run a single simulation using the QNX substrate."""

        substrate = QNXSubstrate()
        sim_config = self._build_sim_config()
        return substrate.run_simulation(sim_config)
