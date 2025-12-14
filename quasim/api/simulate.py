"""Universal simulation entrypoints and engine registry for QuASIM.

This module exposes a stable ``run_scenario`` API used by the QNX substrate to
route requests to different engines (modern, legacy, QVR, or future adapters).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, MutableMapping, Optional

from quantum.python.quasim_sim import simulate as quasim_simulate

from .scenarios import ScenarioSpec, build_circuit


@dataclass
class ScenarioResult:
    """Structured result returned by :func:`run_scenario`."""

    scenario_id: str
    timesteps: int
    seed: int
    engine: str
    raw_output: Dict[str, Any]

    @property
    def simulation_hash(self) -> str:
        """Deterministic SHA-256 hash of ``raw_output``."""

        data = json.dumps(self.raw_output, sort_keys=True, default=str).encode()
        return hashlib.sha256(data).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "timesteps": self.timesteps,
            "seed": self.seed,
            "engine": self.engine,
            "simulation_hash": self.simulation_hash,
            "raw_output": self.raw_output,
        }


_ENGINE_REGISTRY: Dict[str, Callable[..., MutableMapping[str, Any]]] = {}


def register_engine(name: str, fn: Callable[..., MutableMapping[str, Any]]) -> None:
    """Register an engine function for :func:`run_scenario`."""

    _ENGINE_REGISTRY[name] = fn


def _run_modern_engine(
    *, scenario_id: str, timesteps: int, seed: int, extra: Optional[Dict[str, Any]]
) -> MutableMapping[str, Any]:
    spec = ScenarioSpec(scenario_id=scenario_id, timesteps=timesteps, seed=seed, extra=extra or {})
    circuit = build_circuit(spec)
    result = quasim_simulate(circuit, precision="fp8")
    return {
        "engine": "quasim_modern",
        "scenario_id": scenario_id,
        "timesteps": timesteps,
        "seed": seed,
        "precision": "fp8",
        "circuit_summary": {
            "num_gates": len(circuit),
            "gate_size": len(circuit[0]) if circuit else 0,
        },
        "result": [complex(value) for value in result],
    }


def run_scenario(
    scenario_id: str,
    timesteps: int,
    seed: int,
    engine: str = "quasim_modern",
    extra: Optional[Dict[str, Any]] = None,
) -> ScenarioResult:
    """Execute a simulation scenario via a registered engine."""

    if engine not in _ENGINE_REGISTRY:
        raise ValueError(
            f"Engine '{engine}' is not registered. Available engines: {list(_ENGINE_REGISTRY.keys())}"
        )

    backend_fn = _ENGINE_REGISTRY[engine]
    raw_output = backend_fn(
        scenario_id=scenario_id,
        timesteps=timesteps,
        seed=seed,
        extra=extra or {},
    )

    return ScenarioResult(
        scenario_id=scenario_id,
        timesteps=timesteps,
        seed=seed,
        engine=engine,
        raw_output=raw_output,
    )


register_engine("quasim_modern", _run_modern_engine)

__all__ = ["ScenarioResult", "register_engine", "run_scenario", "build_circuit", "ScenarioSpec"]
