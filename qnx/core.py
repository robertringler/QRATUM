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
        logger.info(
            "QNXSubstrate initialised with backends", extra={"backends": list(self.backends)}
        )

    def initialise_runtime(self) -> MutableMapping[str, Any]:
        """Return runtime metadata for lifecycle coordination."""

        return {
            "status": "initialised",
            "available_backends": list(self.backends),
        }

    def boot_backend(
        self, backend_name: str, config: SimulationConfig | None = None
    ) -> MutableMapping[str, Any]:
        """Perform a lightweight backend probe to confirm readiness."""

        backend = self._resolve_backend(backend_name)
        probe_config = config or SimulationConfig(
            scenario_id="__boot_probe__",
            timesteps=1,
            seed=0,
            backend=backend_name,
        )

        ready = False
        try:
            ready = backend.validate(probe_config)
        except Exception as exc:  # pragma: no cover - defensive
            logger.info(
                "Backend validation failed", extra={"backend": backend_name, "error": str(exc)}
            )

        return {"backend": backend_name, "ready": bool(ready)}

    def run_simulation(self, config: SimulationConfig) -> SubstrateResult:
        """Run a simulation on the configured backend and return a structured result."""

        backend = self._resolve_backend(config.backend)

        validate_security_context(config.security_level)

        logger.info(
            "Running simulation",
            extra={
                "backend": config.backend,
                "scenario_id": config.scenario_id,
                "timesteps": config.timesteps,
            },
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
        simulation_hash = compute_integrity_hash(
            json.dumps(hash_payload, sort_keys=True, default=str)
        )
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

    def dispatch(self, config: SimulationConfig, *, rtos: Any | None = None) -> SubstrateResult:
        """Dispatch the simulation with optional RTOS instrumentation."""

        if rtos is not None:
            rtos.boot()
            rtos.dispatch_ticks(iterations=max(1, config.timesteps))

        result = self.run_simulation(config)

        if rtos is not None:
            rtos.send_ipc(
                "simulation_complete", {"hash": result.simulation_hash, "backend": result.backend}
            )
            rtos.teardown()

        return result

    def teardown_backend(self, backend_name: str) -> MutableMapping[str, str]:
        """Mark a backend as stopped for lifecycle tracking."""

        self._resolve_backend(backend_name)
        return {"backend": backend_name, "status": "stopped"}

    def lifecycle_run(
        self, config: SimulationConfig, *, rtos: Any | None = None
    ) -> MutableMapping[str, Any]:
        """Execute the full lifecycle: init → boot → dispatch → teardown."""

        init_state = self.initialise_runtime()
        boot_state = self.boot_backend(config.backend, config)
        result = self.dispatch(config, rtos=rtos)
        teardown_state = self.teardown_backend(config.backend)

        return {
            "init": init_state,
            "boot": boot_state,
            "dispatch_result": result,
            "teardown": teardown_state,
        }

    def _resolve_backend(self, backend_name: str):
        backend = self.backends.get(backend_name)
        if backend is None:
            raise ValueError(f"Backend '{backend_name}' is not available")
        return backend
