from __future__ import annotations

import json
import time
from typing import Any, MutableMapping

from .backends import get_backend_registry
from .logging import get_logger
from .security import compute_integrity_hash, validate_security_context
from .sustainability import estimate_carbon
from .types import SimulationConfig, SubstrateResult

logger = get_logger(__name__)


class QNXSubstrate:
    """Coordinator for running simulations across multiple backends."""

    def __init__(self) -> None:
        self.backends = get_backend_registry()
        logger.info("QNXSubstrate initialised with backends", extra={"backends": list(self.backends)})

    def run_simulation(self, config: SimulationConfig) -> SubstrateResult:
        """Run a simulation on the configured backend and return a structured result."""

        backend = self.backends.get(config.backend)
        if backend is None:
            raise ValueError(f"Backend '{config.backend}' is not available")

        validate_security_context(config.security_level)

        logger.info(
            "Running simulation",
            extra={"backend": config.backend, "scenario_id": config.scenario_id, "timesteps": config.timesteps},
        )

        start = time.perf_counter()
        raw_results = backend.run(config)
        execution_time_ms = (time.perf_counter() - start) * 1000

        hash_payload: MutableMapping[str, Any] = {
            "backend": config.backend,
            "scenario_id": config.scenario_id,
            "timesteps": config.timesteps,
            "seed": config.seed,
            "raw_results": raw_results,
        }
        simulation_hash = compute_integrity_hash(json.dumps(hash_payload, sort_keys=True, default=str))
        carbon_emissions = estimate_carbon(raw_results)

        errors = []
        warnings = []
        if isinstance(raw_results, dict):
            if raw_results.get("status") == "error":
                errors.append("backend_reported_error")
            if "warnings" in raw_results and isinstance(raw_results["warnings"], list):
                warnings.extend(raw_results["warnings"])

        return SubstrateResult(
            backend=config.backend,
            scenario_id=config.scenario_id,
            timesteps=config.timesteps,
            seed=config.seed,
            raw_results=raw_results,
            simulation_hash=simulation_hash,
            execution_time_ms=execution_time_ms,
            carbon_emissions_kg=carbon_emissions,
            errors=errors,
            warnings=warnings,
        )
