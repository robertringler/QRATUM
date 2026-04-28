"""Reality Interface Controller (RIC) — deterministic bounded control operator.

Implements the formal pipeline specified in
``qagents/prompts/reality_interface_controller_v1.md``:

    (intent, world_state, system_limits) →
        (intent_interpretation, selected_action, predicted_outcome, fallback_plan)

Pipeline (executed in strict order):
    1. INTENT DECODING        — extract goal, constraints, priority
    2. STATE ALIGNMENT        — feasibility + conflict set
    3. ACTION CANDIDATE SET   — control / adjust / hold / abort
    4. SIMULATION (mandatory) — predicted next state per candidate
    5. ADMISSIBLE SET         — risk ≤ r_max ∧ stability ≥ σ_min
    6. POLICY (lexicographic) — Safety > Stability > Minimal-intervention
    7. CONFIDENCE             — proportional to stability score
    8. FALLBACK PLAN          — trigger + recovery action

The class is fully deterministic: the same (intent, world_state, system_limits)
always produces the same output. A pluggable simulator/proposer interface
allows wiring to CIIR dynamics or to LLM-based proposers; when no proposer
is available, the deterministic baseline is used (failing closed to ``abort``
on safety violations).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

ActionType = str  # "control" | "adjust" | "hold" | "abort"
ACTION_TYPES: tuple[ActionType, ...] = ("control", "adjust", "hold", "abort")


# ---------------------------------------------------------------------------
# Simulator / Proposer protocols
# ---------------------------------------------------------------------------


class Simulator(Protocol):
    """Predicts ``hat{o}_{t+1}`` given current state and a candidate action."""

    def predict(
        self,
        world_state: Mapping[str, Any],
        action: Mapping[str, Any],
    ) -> dict[str, Any]: ...


class Proposer(Protocol):
    """Proposes magnitudes for each candidate action type.

    Returns a mapping ``{action_type: magnitude}`` with magnitudes ``>= 0``.
    A proposer may be backed by an LLM, but must be deterministic for a
    given (intent, world_state, system_limits) tuple to keep the RIC
    reproducible. See ``qagents.llm_backends``.
    """

    def propose(
        self,
        intent: Mapping[str, Any],
        world_state: Mapping[str, Any],
        system_limits: Mapping[str, Any],
    ) -> dict[ActionType, float]: ...


# ---------------------------------------------------------------------------
# Default deterministic simulator + proposer
# ---------------------------------------------------------------------------


def _state_distance(state: Mapping[str, Any], target: Mapping[str, Any]) -> float:
    """L1 distance over shared numeric keys (deterministic, bounded)."""

    if not target:
        return 0.0
    diff = 0.0
    for key, want in target.items():
        try:
            have = float(state.get(key, 0.0))
            diff += abs(float(want) - have)
        except (TypeError, ValueError):
            continue
    return diff


class LinearSimulator:
    r"""Default deterministic simulator: ``\hat o_{t+1} = o_t + delta(action)``.

    Each numeric key in ``world_state`` is nudged by ``magnitude * direction``
    where direction depends on action type:

    - ``control`` : +magnitude  (full intervention)
    - ``adjust``  : +magnitude / 4
    - ``hold``    : 0
    - ``abort``   : 0  (next state == current state, system frozen)
    """

    _SCALE: dict[ActionType, float] = {
        "control": 1.0,
        "adjust": 0.25,
        "hold": 0.0,
        "abort": 0.0,
    }

    def predict(
        self,
        world_state: Mapping[str, Any],
        action: Mapping[str, Any],
    ) -> dict[str, Any]:
        scale = self._SCALE.get(str(action.get("type", "hold")), 0.0)
        magnitude = float(action.get("magnitude", 0.0))
        delta = scale * magnitude
        out: dict[str, Any] = {}
        for key, value in world_state.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out[key] = float(value) + delta
            else:
                out[key] = value
        return out


class StaticProposer:
    """Deterministic proposer with constant magnitudes per action type."""

    def __init__(
        self,
        magnitudes: Mapping[ActionType, float] | None = None,
    ) -> None:
        defaults: dict[ActionType, float] = {
            "control": 1.0,
            "adjust": 0.25,
            "hold": 0.0,
            "abort": 0.0,
        }
        if magnitudes:
            defaults.update({k: float(v) for k, v in magnitudes.items()})
        self._magnitudes = defaults

    def propose(
        self,
        intent: Mapping[str, Any],
        world_state: Mapping[str, Any],
        system_limits: Mapping[str, Any],
    ) -> dict[ActionType, float]:
        return dict(self._magnitudes)


# ---------------------------------------------------------------------------
# RIC dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RICDecision:
    """The strict 4-key output of the RIC pipeline."""

    intent_interpretation: dict[str, Any]
    selected_action: dict[str, Any]
    predicted_outcome: dict[str, Any]
    fallback_plan: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_interpretation": dict(self.intent_interpretation),
            "selected_action": dict(self.selected_action),
            "predicted_outcome": dict(self.predicted_outcome),
            "fallback_plan": dict(self.fallback_plan),
        }


@dataclass
class _Candidate:
    type: ActionType
    magnitude: float
    predicted: dict[str, Any]
    risk: float
    stability: float


# ---------------------------------------------------------------------------
# RIC controller
# ---------------------------------------------------------------------------


class RealityInterfaceController:
    """Deterministic Reality Interface Controller.

    Parameters
    ----------
    simulator
        Object implementing :class:`Simulator`. Defaults to
        :class:`LinearSimulator`.
    proposer
        Object implementing :class:`Proposer`. Defaults to
        :class:`StaticProposer`. Use ``qagents.llm_backends`` to plug in
        an LLM-backed proposer with deterministic fallback.
    risk_estimator
        Optional callable ``(predicted_state, system_limits) -> risk in [0, 1]``.
        Defaults to a constraint-violation count heuristic.
    stability_estimator
        Optional callable ``(predicted_state, intent) -> sigma in [0, 1]``.
        Defaults to ``1 - normalized_distance(predicted, intent.goal_state)``.
    """

    def __init__(
        self,
        simulator: Simulator | None = None,
        proposer: Proposer | None = None,
        risk_estimator: Callable[[Mapping[str, Any], Mapping[str, Any]], float] | None = None,
        stability_estimator: Callable[[Mapping[str, Any], Mapping[str, Any]], float] | None = None,
    ) -> None:
        self.simulator: Simulator = simulator or LinearSimulator()
        self.proposer: Proposer = proposer or StaticProposer()
        self._risk = risk_estimator or _default_risk
        self._stability = stability_estimator or _default_stability

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(
        self,
        intent: Mapping[str, Any],
        world_state: Mapping[str, Any],
        system_limits: Mapping[str, Any],
    ) -> RICDecision:
        """Run the full RIC pipeline and return a strict 4-key decision."""

        # 1. INTENT DECODING — never invent missing values
        interp = self._decode_intent(intent)

        # 2. STATE ALIGNMENT
        feasibility, conflicts = self._align_state(world_state, interp["constraints"])

        # 3 + 4. CANDIDATE SET + SIMULATION (mandatory)
        candidates = self._build_candidates(intent, world_state, system_limits)

        # 5. ADMISSIBLE SET
        r_max = float(system_limits.get("risk_threshold", 1.0))
        sigma_min = float(system_limits.get("stability_threshold", 0.0))
        admissible = [
            c for c in candidates
            if c.type != "abort" and c.risk <= r_max and c.stability >= sigma_min
        ]

        # 6. POLICY — lexicographic: Safety > Stability > Minimal-intervention
        chosen = self._apply_policy(
            candidates=candidates,
            admissible=admissible,
            feasibility=feasibility,
            conflicts=conflicts,
        )

        # 7. CONFIDENCE — proportional to stability, zero if no valid prediction
        confidence = 0.0 if chosen.predicted is None else max(0.0, min(1.0, chosen.stability))
        if chosen.type == "abort":
            confidence = 0.0

        # 8. FALLBACK PLAN
        fallback = self._build_fallback(chosen, feasibility, conflicts)

        return RICDecision(
            intent_interpretation=interp,
            selected_action={
                "type": chosen.type,
                "magnitude": float(chosen.magnitude),
                "confidence": float(confidence),
            },
            predicted_outcome={
                "expected_state": dict(chosen.predicted),
                "risk": float(chosen.risk),
                "stability_score": float(chosen.stability),
            },
            fallback_plan=fallback,
        )

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_intent(intent: Mapping[str, Any]) -> dict[str, Any]:
        # NEVER hallucinate — only echo provided keys; missing → safe defaults
        return {
            "goal": intent.get("goal", ""),
            "constraints": dict(intent.get("constraints", {})),
            "priority": float(intent.get("priority", 0.0)),
        }

    @staticmethod
    def _align_state(
        world_state: Mapping[str, Any],
        constraints: Mapping[str, Any],
    ) -> tuple[int, list[str]]:
        conflicts: list[str] = []
        for name, spec in constraints.items():
            if not _constraint_satisfied(world_state, spec):
                conflicts.append(name)
        feasibility = 0 if conflicts else 1
        return feasibility, conflicts

    def _build_candidates(
        self,
        intent: Mapping[str, Any],
        world_state: Mapping[str, Any],
        system_limits: Mapping[str, Any],
    ) -> list[_Candidate]:
        magnitudes = self.proposer.propose(intent, world_state, system_limits)
        candidates: list[_Candidate] = []
        for atype in ACTION_TYPES:
            magnitude = max(0.0, float(magnitudes.get(atype, 0.0)))
            action = {"type": atype, "magnitude": magnitude}
            predicted = dict(self.simulator.predict(world_state, action))
            risk = float(self._risk(predicted, system_limits))
            stability = float(self._stability(predicted, intent))
            # Clamp into [0, 1] to enforce HARD CONSTRAINTS
            risk = max(0.0, min(1.0, risk))
            stability = max(0.0, min(1.0, stability))
            candidates.append(
                _Candidate(
                    type=atype,
                    magnitude=magnitude,
                    predicted=predicted,
                    risk=risk,
                    stability=stability,
                )
            )
        return candidates

    @staticmethod
    def _apply_policy(
        candidates: list[_Candidate],
        admissible: list[_Candidate],
        feasibility: int,
        conflicts: list[str],
    ) -> _Candidate:
        # RULE 1 — SAFETY DOMINANCE: infeasible state OR no admissible → abort
        if feasibility == 0 or conflicts or not admissible:
            for cand in candidates:
                if cand.type == "abort":
                    return cand

        # RULE 2 — STABILITY AWARENESS: prefer hold/adjust if any admissible
        # candidate has stability below 0.5 (system near instability).
        near_instability = any(c.stability < 0.5 for c in admissible)
        if near_instability:
            stable_choices = [c for c in admissible if c.type in ("hold", "adjust")]
            if stable_choices:
                # RULE 3 within stable subset
                return min(stable_choices, key=lambda c: (c.magnitude, c.type))

        # RULE 3 — MINIMAL INTERVENTION: argmin magnitude among admissible
        return min(admissible, key=lambda c: (c.magnitude, c.type))

    @staticmethod
    def _build_fallback(
        chosen: _Candidate,
        feasibility: int,
        conflicts: list[str],
    ) -> dict[str, Any]:
        if chosen.type == "abort":
            return {
                "trigger_condition": "constraint_violation" if conflicts else "no_admissible_action",
                "action": "hold",
            }
        if chosen.stability < 0.5 or feasibility == 0:
            return {"trigger_condition": "stability_drop", "action": "hold"}
        return {"trigger_condition": "stability_drop", "action": "abort"}


# ---------------------------------------------------------------------------
# Default risk / stability estimators
# ---------------------------------------------------------------------------


def _default_risk(predicted: Mapping[str, Any], system_limits: Mapping[str, Any]) -> float:
    """Heuristic risk: fraction of safety bounds violated by ``predicted``."""

    bounds = system_limits.get("safety_bounds", {}) or {}
    if not bounds:
        return 0.0
    violations = 0
    total = 0
    for key, spec in bounds.items():
        if not isinstance(spec, Mapping):
            continue
        total += 1
        try:
            value = float(predicted.get(key, 0.0))
        except (TypeError, ValueError):
            violations += 1
            continue
        lo = spec.get("min", float("-inf"))
        hi = spec.get("max", float("inf"))
        if value < float(lo) or value > float(hi):
            violations += 1
    if total == 0:
        return 0.0
    return violations / total


def _default_stability(predicted: Mapping[str, Any], intent: Mapping[str, Any]) -> float:
    """Stability = 1 − normalized distance to ``intent.constraints`` numeric targets."""

    targets: dict[str, Any] = {}
    constraints = intent.get("constraints", {}) or {}
    for key, spec in constraints.items():
        if isinstance(spec, Mapping) and "target" in spec:
            targets[key] = spec["target"]
    if not targets:
        return 1.0
    dist = _state_distance(predicted, targets)
    # squash to [0, 1] via 1 / (1 + d)
    return 1.0 / (1.0 + dist)


# ---------------------------------------------------------------------------
# Constraint helpers
# ---------------------------------------------------------------------------


def _constraint_satisfied(world_state: Mapping[str, Any], spec: Any) -> bool:
    """True iff ``world_state`` satisfies ``spec``.

    Spec forms supported:

    - ``{"min": x}`` / ``{"max": y}`` / ``{"min": x, "max": y}`` — applied to
      numeric value with same key (caller passes name as constraint key).
    - ``{"key": k, "min"/"max": ...}`` — explicit key mapping.
    - ``{"equals": v}`` — strict equality.
    - bool/None — assumed satisfied.
    """

    if spec is None or isinstance(spec, bool):
        return True
    if not isinstance(spec, Mapping):
        return True
    key = spec.get("key")
    if key is None:
        # Spec applies to its own constraint name; skip if key not in state
        return True
    try:
        value = float(world_state.get(key, 0.0))
    except (TypeError, ValueError):
        return False
    if "equals" in spec and value != float(spec["equals"]):
        return False
    if "min" in spec and value < float(spec["min"]):
        return False
    if "max" in spec and value > float(spec["max"]):
        return False
    return True


__all__ = [
    "ACTION_TYPES",
    "ActionType",
    "LinearSimulator",
    "Proposer",
    "RICDecision",
    "RealityInterfaceController",
    "Simulator",
    "StaticProposer",
]
