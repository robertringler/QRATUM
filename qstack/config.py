"""Configuration objects for the Q-Stack core facade.

These dataclasses provide deterministic defaults for orchestrating QNX,
QuASIM, and QuNimbus subsystems via the QStackSystem facade.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class QNXConfig:
    """Configuration for the QNX substrate adapter."""

    backend: str = "quasim_modern"
    timesteps: int = 1
    scenario_id: str = "default_scenario"
    seed: int = 0
    security_level: str = "standard"


@dataclass
class QuASIMConfig:
    """Configuration for the QuASIM runtime adapter."""

    simulation_precision: str = "fp32"
    max_workspace_mb: int = 1024
    backend: str = "cpu"
    seed: int | None = None


@dataclass
class QuNimbusConfig:
    """Configuration for the QuNimbus adapter."""

    enable_synthetic_economy: bool = True
    enable_real_policies: bool = True
    enable_node_governance: bool = True
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class QStackConfig:
    """Aggregated configuration for the full Q-Stack."""

    qnx: QNXConfig = field(default_factory=QNXConfig)
    quasim: QuASIMConfig = field(default_factory=QuASIMConfig)
    qunimbus: QuNimbusConfig = field(default_factory=QuNimbusConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the full configuration."""

        return asdict(self)
