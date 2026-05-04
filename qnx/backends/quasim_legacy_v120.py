from __future__ import annotations

from typing import MutableMapping

from ..types import SimulationConfig


class QuasimLegacyV120Backend:
    """Stub adapter for the legacy QuASIM v1.2.0 engine."""

    name = "quasim_legacy_v1_2_0"

    def run(self, config: SimulationConfig) -> MutableMapping[str, object]:
        seed = config.seed if config.seed is not None else 0
        return {
            "engine": self.name,
            "scenario_id": config.scenario_id,
            "timesteps": config.timesteps,
            "seed": seed,
            "status": "not_implemented",
            "note": "Legacy v1.2.0 wiring TODO.",
        }

    def validate(self, config: SimulationConfig) -> bool:
        return False


__all__ = ["QuasimLegacyV120Backend"]
