"""Universal QuASIM Simulation API Layer.

This module defines a single stable, forward-compatible entrypoint
`run_scenario()` that QNX can call to execute a simulation across any
internal QuASIM engine (modern, legacy, experimental, or external wrappers).

The contract is intentionally minimal and stable:
    - Inputs: scenario_id, timesteps, seed
    - Output: structured dict with deterministic fields
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, MutableMapping

from runtime.python.quasim.runtime import Config as RuntimeConfig
from runtime.python.quasim.runtime import runtime as runtime_context

# ---------------------------------------------------------------------------
# Dataclass for internal standardized return payload
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    scenario_id: str
    timesteps: int
    seed: int
    engine: str
    raw_output: dict[str, Any]

    @property
    def simulation_hash(self) -> str:
        """Deterministic SHA-256 hash of raw_output."""
        data = json.dumps(self.raw_output, sort_keys=True, default=str).encode()
        return hashlib.sha256(data).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "timesteps": self.timesteps,
            "seed": self.seed,
            "engine": self.engine,
            "simulation_hash": self.simulation_hash,
            "raw_output": self.raw_output,
        }


# ---------------------------------------------------------------------------
# Loader registry — backend adapters drop into this dictionary
# QNX backends call this same API.
# ---------------------------------------------------------------------------

_ENGINE_REGISTRY: dict[str, Callable[..., MutableMapping[str, Any]]] = {}


def register_engine(name: str, fn: Callable[..., MutableMapping[str, Any]]):
    """Register a backend engine function."""
    _ENGINE_REGISTRY[name] = fn


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _complex_to_json(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def _generate_tensors(seed: int, timesteps: int) -> list[list[complex]]:
    rng = random.Random(seed)
    tensors: list[list[complex]] = []
    for step in range(max(1, timesteps)):
        base = float(step + 1)
        tensors.append(
            [
                complex(base + rng.random(), rng.random()),
                complex(base / 2 + rng.random(), rng.random() / 2),
            ]
        )
    return tensors


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------


def _run_modern_backend(
    *, scenario_id: str, timesteps: int, seed: int, extra: dict[str, Any] | None
) -> MutableMapping[str, Any]:
    tensors = _generate_tensors(seed, timesteps)
    with runtime_context(RuntimeConfig()) as handle:
        outputs = handle.simulate(tensors)
        latency = handle.average_latency

    return {
        "status": "ok",
        "scenario_id": scenario_id,
        "timesteps": timesteps,
        "seed": seed,
        "engine": "quasim_modern",
        "average_latency": latency,
        "outputs": [_complex_to_json(value) for value in outputs],
        "metadata": extra or {},
    }


def _run_legacy_backend(
    *, scenario_id: str, timesteps: int, seed: int, extra: dict[str, Any] | None
) -> MutableMapping[str, Any]:
    # Legacy mode reuses the same runtime but applies a deterministic postprocess
    tensors = _generate_tensors(seed, timesteps)
    with runtime_context(RuntimeConfig(simulation_precision="fp16")) as handle:
        outputs = handle.simulate(tensors)
        latency = handle.average_latency

    adjusted = [complex(val.real * 0.95, val.imag * 0.95) for val in outputs]

    return {
        "status": "ok",
        "scenario_id": scenario_id,
        "timesteps": timesteps,
        "seed": seed,
        "engine": "quasim_legacy_v1_2_0",
        "average_latency": latency,
        "outputs": [_complex_to_json(value) for value in adjusted],
        "metadata": {"mode": "legacy_v1_2_0", **(extra or {})},
    }


def _run_qvr_backend(
    *, scenario_id: str, timesteps: int, seed: int, extra: dict[str, Any] | None
) -> MutableMapping[str, Any]:
    if os.name != "nt":  # pragma: no cover - platform-specific
        raise RuntimeError("QVR backend requires Windows")

    executable = os.environ.get("QVR_EXECUTABLE", "qvr.exe")
    if not Path(executable).exists():
        raise FileNotFoundError(f"QVR executable not found at '{executable}'")

    payload = {"scenario_id": scenario_id, "timesteps": timesteps, "seed": seed}
    if extra:
        payload.update(extra)

    process = subprocess.run(
        [executable],
        input=json.dumps(payload).encode(),
        capture_output=True,
        check=False,
    )

    stdout = process.stdout.decode(errors="replace")
    stderr = process.stderr.decode(errors="replace")

    try:
        result_payload = json.loads(stdout) if stdout else {}
    except json.JSONDecodeError:
        result_payload = {"raw_stdout": stdout}

    return {
        "status": "ok" if process.returncode == 0 else "error",
        "scenario_id": scenario_id,
        "timesteps": timesteps,
        "seed": seed,
        "engine": "qvr_win",
        "stdout": stdout,
        "stderr": stderr,
        "payload": result_payload,
        "returncode": process.returncode,
    }


# ---------------------------------------------------------------------------
# Universal API entrypoint
# ---------------------------------------------------------------------------


def run_scenario(
    scenario_id: str,
    timesteps: int,
    seed: int,
    engine: str = "quasim_modern",
    extra: dict[str, Any] | None = None,
) -> ScenarioResult:
    """
    Execute a simulation scenario via a registered engine.

    This function routes calls to the appropriate backend (modern, legacy,
    QVR Windows interface, adapters, HPC wraps, etc.).
    """
    if engine not in _ENGINE_REGISTRY:
        raise ValueError(
            f"Engine '{engine}' is not registered. "
            f"Available engines: {list(_ENGINE_REGISTRY.keys())}"
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


# ---------------------------------------------------------------------------
# Register real backends
# ---------------------------------------------------------------------------

register_engine("quasim_modern", _run_modern_backend)
register_engine("quasim_legacy_v1_2_0", _run_legacy_backend)
register_engine("qvr_win", _run_qvr_backend)

__all__ = [
    "ScenarioResult",
    "register_engine",
    "run_scenario",
]
