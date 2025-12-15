"""Deterministic scenario-to-circuit helpers for QuASIM simulations.

The functions in this module provide a stable mapping from a scenario identifier
plus basic parameters to a synthetic circuit representation that can be executed
by :func:`quantum.python.quasim_sim.simulate`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ScenarioSpec:
    """Specification describing a simulation scenario.

    Attributes
    ----------
    scenario_id: str
        Logical identifier for the scenario to build.
    timesteps: int
        Number of timesteps or gates to generate in the circuit.
    seed: int
        Seed used to ensure deterministic circuit generation.
    extra: dict[str, Any]
        Optional additional parameters for the scenario builder.
    """

    scenario_id: str
    timesteps: int
    seed: int
    extra: Dict[str, Any]


def _base_pattern(rng: random.Random, width: int = 2) -> List[complex]:
    """Create a small deterministic pattern of complex numbers."""

    return [complex(rng.random(), rng.random()) for _ in range(width)]


def _scenario_variant(rng: random.Random, scenario_id: str) -> float:
    """Return a deterministic multiplier derived from the scenario id."""

    return (sum(ord(ch) for ch in scenario_id) % 7 + 1) / 5.0


def build_circuit(spec: ScenarioSpec) -> List[List[complex]]:
    """Build a deterministic synthetic circuit from a :class:`ScenarioSpec`.

    The resulting circuit is a list of timesteps, each containing a small list
    of complex values. While synthetic, this structure exercises the runtime and
    remains stable for a given ``(scenario_id, timesteps, seed)`` tuple.
    """

    rng = random.Random(spec.seed)
    circuit: List[List[complex]] = []
    multiplier = _scenario_variant(rng, spec.scenario_id)

    for _ in range(max(1, spec.timesteps)):
        base_gate = _base_pattern(rng)
        circuit.append([value * multiplier for value in base_gate])

    return circuit


__all__ = ["ScenarioSpec", "build_circuit"]
