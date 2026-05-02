"""CRSI adapter for the Reality Interface Controller.

Wires the RIC to **Constrained Recursive Self-Improvement (CRSI)** as
defined in ``qratum_asi/orchestrator_master.py``. CRSI is a
safety-critical operation type whose contract requires:

* **Immutable boundaries** — frozen invariants that must never be
  modified (human_oversight_requirement, kill switches, determinism
  guarantee, …).
* **Prohibited goals** — goals that must never be proposed
  (remove_human_oversight, deceive_operators, …).
* **Human approval** — every CRSI step is rollback-bound and gated.

Mirrors of those frozensets are duplicated here so the adapter does not
require importing :mod:`qratum_asi` (which depends on :mod:`qradle`).

Mapping into the RIC schema
---------------------------
A CRSI ``world_state`` for this adapter is::

    {
        "pillar":                  "Q-MIND",
        "proposed_goals":          ["optimise reasoning"],
        "proposed_modifications":  ["weights"],
        "human_oversight_pending": False,
        "rollback_available":      True,
        "tampering_score":         0.02,   # in [0, 1]
    }

Any prohibited goal, immutable-boundary touch, missing rollback, or
``human_oversight_pending`` forces ``risk = 1.0`` so the RIC's safety
dominance triggers an ``abort``. Otherwise risk grows with magnitude
and tampering score; stability falls with the same.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from qagents.reality_interface import (
    IntentInterpretation,
    PredictedOutcome,
    RealityInterfaceController,
)

#: Frozen invariants — these must never be modified by a CRSI step.
#: Mirrors ``qratum_asi.orchestrator_master.IMMUTABLE_BOUNDARIES``.
IMMUTABLE_BOUNDARIES: frozenset[str] = frozenset(
    [
        "human_oversight_requirement",
        "authorization_system",
        "safety_level_system",
        "rollback_capability",
        "event_emission_requirement",
        "determinism_guarantee",
        "merkle_chain_integrity",
        "contract_immutability",
    ]
)

#: Goals that must never be proposed by a CRSI step.
#: Mirrors ``qratum_asi.orchestrator_master.PROHIBITED_GOALS``.
PROHIBITED_GOALS: frozenset[str] = frozenset(
    [
        "remove_human_oversight",
        "disable_authorization",
        "modify_safety_constraints",
        "acquire_resources_without_approval",
        "replicate_without_authorization",
        "deceive_operators",
        "manipulate_humans",
        "evade_monitoring",
        "remove_kill_switches",
        "modify_core_values",
        "bypass_immutable_boundaries",
        "disable_rollback",
    ]
)


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_iterable(value: Any) -> Iterable[str]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        return (str(k) for k, v in value.items() if v)
    try:
        return (str(item) for item in value)
    except TypeError:
        return ()


def _safety_block(world_state: Mapping[str, Any]) -> str | None:
    """Return a non-empty reason string if a hard CRSI rule is violated."""

    for goal in _as_iterable(world_state.get("proposed_goals")):
        if goal in PROHIBITED_GOALS:
            return f"prohibited_goal:{goal}"
    for mod in _as_iterable(world_state.get("proposed_modifications")):
        if mod in IMMUTABLE_BOUNDARIES:
            return f"immutable_boundary:{mod}"
    if world_state.get("human_oversight_pending"):
        return "human_oversight_pending"
    if world_state.get("rollback_available") is False:
        return "rollback_unavailable"
    return None


def crs_simulator(
    action_type: str,
    magnitude: float,
    world_state: Mapping[str, Any],
    system_limits: Mapping[str, Any],
) -> PredictedOutcome:
    """Predict the safety surface of a CRSI step.

    Hard rules (force ``risk = 1.0``):

    * any ``proposed_goals`` ∈ :data:`PROHIBITED_GOALS`,
    * any ``proposed_modifications`` ∈ :data:`IMMUTABLE_BOUNDARIES`,
    * ``human_oversight_pending`` truthy,
    * ``rollback_available`` is explicitly ``False``.

    Soft rules: risk grows with ``magnitude`` and ``tampering_score``;
    stability falls with the same. ``abort`` always returns a safe
    state (risk=0, stability=1).
    """

    magnitude = _clamp01(_f(magnitude, 0.0))
    tampering = _clamp01(_f(world_state.get("tampering_score"), 0.0))
    block = _safety_block(world_state)

    expected_state: dict[str, Any] = {
        "pillar": world_state.get("pillar"),
        "tampering_score": tampering,
        "rollback_available": world_state.get("rollback_available", True),
        "human_oversight_pending": bool(
            world_state.get("human_oversight_pending", False)
        ),
    }
    if block is not None:
        expected_state["safety_block"] = block

    if action_type == "abort":
        # Aborting is always safe.
        return PredictedOutcome(
            expected_state=expected_state,
            risk=0.0,
            stability_score=1.0,
        )

    if block is not None:
        # Hard violation — force RIC into the safety-dominance branch.
        return PredictedOutcome(
            expected_state=expected_state,
            risk=1.0,
            stability_score=0.0,
        )

    if action_type == "hold":
        return PredictedOutcome(
            expected_state=expected_state,
            risk=tampering,
            stability_score=_clamp01(1.0 - tampering),
        )

    # control / adjust — soft scaling.
    risk = _clamp01(tampering + 0.5 * magnitude * (1.0 - tampering))
    stability = _clamp01(1.0 - risk)
    expected_state["__simulated_action__"] = {
        "type": action_type,
        "magnitude": magnitude,
    }
    return PredictedOutcome(
        expected_state=expected_state,
        risk=risk,
        stability_score=stability,
    )


def crs_proposer(
    intent: IntentInterpretation,
    world_state: Mapping[str, Any],
    system_limits: Mapping[str, Any],
) -> tuple[str, float]:
    """Propose a conservative CRSI step.

    For safety-critical pillars (``Q-WILL``, ``Q-EVOLVE``, ``Q-FORGE``)
    the proposed magnitude is halved relative to lower-risk pillars.
    Returns ``("hold", 0.0)`` if any hard rule is already violated, so
    the controller's predict-before-act pass surfaces the abort.
    """

    if _safety_block(world_state) is not None:
        return "hold", 0.0

    pillar = str(world_state.get("pillar", ""))
    base = _clamp01(_f(intent.constraints.get("step", 0.25), 0.25))
    if pillar in {"Q-WILL", "Q-EVOLVE", "Q-FORGE"}:
        base *= 0.5
    max_magnitude = _clamp01(_f(system_limits.get("max_magnitude"), 1.0))
    magnitude = min(max_magnitude, _clamp01(base))
    if magnitude == 0.0:
        return "hold", 0.0
    return "adjust", magnitude


def make_crs_controller(**kwargs: Any) -> RealityInterfaceController:
    """Return a :class:`RealityInterfaceController` wired to CRSI semantics."""

    return RealityInterfaceController(
        simulator=crs_simulator,
        proposer=crs_proposer,
        **kwargs,
    )


__all__ = [
    "IMMUTABLE_BOUNDARIES",
    "PROHIBITED_GOALS",
    "crs_simulator",
    "crs_proposer",
    "make_crs_controller",
]
