from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, MutableMapping, Protocol


class SecurityLevel(str, Enum):
    """Security level for running simulations."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class SimulationConfig:
    """Configuration for running a simulation on the QNX substrate."""

    scenario_id: str
    timesteps: int
    seed: int | None = None
    backend: str = "quasim_modern"
    parameters: Mapping[str, Any] | None = None
    security_level: SecurityLevel = SecurityLevel.LOW


@dataclass
class SubstrateResult:
    """Structured result returned by the QNX substrate."""

    backend: str
    scenario_id: str
    timesteps: int
    seed: int | None
    raw_results: MutableMapping[str, Any]
    simulation_hash: str
    execution_time_ms: float
    carbon_emissions_kg: float
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class SimulationBackend(Protocol):
    """Protocol all simulation backends must implement."""

    name: str

    def run(self, config: SimulationConfig) -> MutableMapping[str, Any]:
        """Execute the simulation with the provided configuration."""

    def validate(self, config: SimulationConfig) -> bool:  # pragma: no cover - interface
        """Return True if the backend is operational for the given config."""
