from __future__ import annotations

from typing import MutableMapping

from quasim.api.scenarios import ScenarioSpec
from quasim.api.simulate import run_scenario

from ..types import SimulationConfig


class QuasimModernBackend:
    """Backend targeting the current QuASIM engine via the scenario API."""

    name = "quasim_modern"

    def run(self, config: SimulationConfig) -> MutableMapping[str, object]:
        seed = config.seed if config.seed is not None else 0
        spec = ScenarioSpec(
            scenario_id=config.scenario_id,
            timesteps=config.timesteps,
            seed=seed,
            extra=dict(config.parameters or {}),
        )
        result = run_scenario(
            scenario_id=spec.scenario_id,
            timesteps=spec.timesteps,
            seed=spec.seed,
            engine=self.name,
            extra=spec.extra,
        )
        payload = result.to_dict()
        payload.setdefault("raw_output", {}).setdefault("metadata", {}).update(
            {"security_level": config.security_level.value}
        )
        return payload

    def validate(self, config: SimulationConfig) -> bool:
        try:
            smoke_config = SimulationConfig(
                scenario_id=config.scenario_id,
                timesteps=1,
                seed=config.seed if config.seed is not None else 0,
                backend=config.backend,
                parameters=config.parameters,
                security_level=config.security_level,
            )
            self.run(smoke_config)
            return True
        except Exception:  # pragma: no cover - defensive
            return False


__all__ = ["QuasimModernBackend"]
