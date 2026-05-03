"""Reality Interface Controller (RIC).

Deterministic Python counterpart to the LLM-facing system prompt at
``qagents/prompts/reality_interface_controller.md``.

The controller implements the strict perception -> model -> action loop:

    1. Intent decoding          -> goal, constraints, priority
    2. World alignment          -> feasibility, constraint conflicts
    3. Simulation (predict)     -> expected_state, risk, stability_score
    4. Action synthesis         -> {control | adjust | hold | abort}
                                   under safety dominance, stability
                                   awareness, minimal intervention, and
                                   predict-before-act enforcement.

The ``RealityInterfaceController.step`` method consumes a payload of the form

    {
        "intent": ...,
        "world_state": {...},
        "system_limits": {...},
    }

and returns a :class:`ControlDecision` whose :meth:`as_dict` matches the
strict output schema specified in the prompt file.

Simulation and proposal are pluggable callables so the controller can later
wrap real engines (e.g. CIIR plasma reconnection control, MPPK swarm) without
hard-coupling to them here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Allowed values for ``selected_action.type``.
ACTION_TYPES: tuple[str, ...] = ("control", "adjust", "hold", "abort")

#: Default risk threshold above which safety dominance forces an abort.
DEFAULT_RISK_THRESHOLD: float = 0.7

#: Default stability threshold below which safety dominance forces an abort.
DEFAULT_STABILITY_THRESHOLD: float = 0.5

#: Stability band above which a full ``control`` action is acceptable.
STRONG_STABILITY_BAND: float = 0.8


# ---------------------------------------------------------------------------
# Schema dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntentInterpretation:
    """Decoded human intent."""

    goal: str
    constraints: dict[str, Any] = field(default_factory=dict)
    priority: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "constraints": dict(self.constraints),
            "priority": int(self.priority),
        }


@dataclass(frozen=True)
class SelectedAction:
    """The chosen control action."""

    type: str
    magnitude: float = 0.0
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.type not in ACTION_TYPES:
            raise ValueError(
                f"selected_action.type must be one of {ACTION_TYPES}, got {self.type!r}"
            )
        if not 0.0 <= float(self.magnitude) <= 1.0:
            raise ValueError("selected_action.magnitude must be in [0, 1]")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("selected_action.confidence must be in [0, 1]")

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "magnitude": float(self.magnitude),
            "confidence": float(self.confidence),
        }


@dataclass(frozen=True)
class PredictedOutcome:
    """Pre-action simulation result."""

    expected_state: dict[str, Any] = field(default_factory=dict)
    risk: float = 0.0
    stability_score: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.risk) <= 1.0:
            raise ValueError("predicted_outcome.risk must be in [0, 1]")
        if not 0.0 <= float(self.stability_score) <= 1.0:
            raise ValueError("predicted_outcome.stability_score must be in [0, 1]")

    def as_dict(self) -> dict[str, Any]:
        return {
            "expected_state": dict(self.expected_state),
            "risk": float(self.risk),
            "stability_score": float(self.stability_score),
        }


@dataclass(frozen=True)
class FallbackPlan:
    """Mandatory fallback contract."""

    trigger_condition: str
    action: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "trigger_condition": self.trigger_condition,
            "action": self.action,
        }


@dataclass(frozen=True)
class ControlDecision:
    """Strict output payload of the controller."""

    intent_interpretation: IntentInterpretation
    selected_action: SelectedAction
    predicted_outcome: PredictedOutcome
    fallback_plan: FallbackPlan

    def as_dict(self) -> dict[str, Any]:
        return {
            "intent_interpretation": self.intent_interpretation.as_dict(),
            "selected_action": self.selected_action.as_dict(),
            "predicted_outcome": self.predicted_outcome.as_dict(),
            "fallback_plan": self.fallback_plan.as_dict(),
        }


# ---------------------------------------------------------------------------
# Pluggable callables
# ---------------------------------------------------------------------------

#: ``simulator(action_type, magnitude, world_state, system_limits) -> PredictedOutcome``.
Simulator = Callable[[str, float, Mapping[str, Any], Mapping[str, Any]], PredictedOutcome]

#: ``proposer(intent_interpretation, world_state, system_limits) -> (action_type, magnitude)``.
Proposer = Callable[
    [IntentInterpretation, Mapping[str, Any], Mapping[str, Any]],
    tuple[str, float],
]


# ---------------------------------------------------------------------------
# Default deterministic implementations
# ---------------------------------------------------------------------------


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def default_proposer(
    intent: IntentInterpretation,
    world_state: Mapping[str, Any],
    system_limits: Mapping[str, Any],
) -> tuple[str, float]:
    """Heuristic proposer: choose ``adjust`` with magnitude derived from the
    intent's stated urgency and the world's current instability.

    The proposed magnitude is intentionally conservative: it never exceeds the
    ``max_magnitude`` declared in ``system_limits`` (defaults to 1.0).
    """

    max_magnitude = _clamp01(_coerce_float(system_limits.get("max_magnitude"), 1.0))
    urgency = _clamp01(_coerce_float(intent.constraints.get("urgency"), 0.5))
    instability = _clamp01(_coerce_float(world_state.get("instability"), 0.0))
    # Higher urgency and higher instability both push toward smaller, safer
    # interventions. We bias toward minimal magnitude.
    base = 0.5 * urgency * (1.0 - instability)
    magnitude = min(max_magnitude, _clamp01(base))
    return "adjust", magnitude


def default_simulator(
    action_type: str,
    magnitude: float,
    world_state: Mapping[str, Any],
    system_limits: Mapping[str, Any],
) -> PredictedOutcome:
    """Deterministic, conservative simulator.

    - ``risk`` rises with magnitude and with current ``instability``.
    - ``stability_score`` falls with magnitude and rises with the current world stability (``stability`` key) (defaulting to 1 - instability).
    - ``expected_state`` echoes the world plus the proposed action so callers
      can inspect what was simulated.
    """

    magnitude = _clamp01(_coerce_float(magnitude, 0.0))
    instability = _clamp01(_coerce_float(world_state.get("instability"), 0.0))
    reported_stability = world_state.get("stability")
    if reported_stability is None:
        base_stability = 1.0 - instability
    else:
        base_stability = _clamp01(_coerce_float(reported_stability, 1.0))

    if action_type == "hold":
        risk = 0.5 * instability
        stability_score = base_stability
    elif action_type == "abort":
        # Aborting is the safest option: zero added risk, stability preserved.
        risk = 0.0
        stability_score = base_stability
    else:
        # control / adjust
        risk = _clamp01(0.5 * instability + 0.5 * magnitude)
        stability_score = _clamp01(base_stability - 0.5 * magnitude)

    expected_state = dict(world_state)
    expected_state["__simulated_action__"] = {
        "type": action_type,
        "magnitude": magnitude,
    }

    return PredictedOutcome(
        expected_state=expected_state,
        risk=risk,
        stability_score=stability_score,
    )


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


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

    def __call__(
        self,
        action_type: str,
        magnitude: float,
        world_state: Mapping[str, Any],
        system_limits: Mapping[str, Any],
    ) -> Any:
        """Make LinearSimulator callable as simulator(type, magnitude, world, limits) -> PredictedOutcome."""
        action = {"type": action_type, "magnitude": magnitude}
        expected = self.predict(world_state, action)
        # Compute simple risk and stability heuristics
        mag = float(magnitude)
        instability = float(world_state.get("instability", 0.0))
        risk = min(1.0, mag * 0.5 + instability * 0.5)
        stability_score = max(0.0, 1.0 - risk)
        return PredictedOutcome(
            expected_state=expected,
            risk=risk,
            stability_score=stability_score,
        )


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

    def __call__(
        self,
        intent: Mapping[str, Any],
        world_state: Mapping[str, Any],
        system_limits: Mapping[str, Any],
    ) -> tuple[ActionType, float]:
        """Make StaticProposer callable; returns (action_type, magnitude) tuple."""
        mags = self.propose(intent, world_state, system_limits)
        # Default to 'control' with its magnitude; controller will downgrade if unsafe.
        action_type = "control"
        return action_type, mags.get(action_type, 1.0)


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
    simulator:
        Callable that predicts outcomes for a candidate action. Defaults to
        :func:`default_simulator`.
    proposer:
        Callable that proposes a (type, magnitude) candidate from intent and
        world state. Defaults to :func:`default_proposer`.
    risk_threshold:
        Hard abort threshold for ``risk``. Per spec, ``risk > risk_threshold``
        forces ``abort``.
    default_stability_threshold:
        Stability threshold used when ``system_limits`` does not provide one.
    """

    simulator: Simulator = field(default=default_simulator)
    proposer: Proposer = field(default=default_proposer)
    risk_threshold: float = DEFAULT_RISK_THRESHOLD
    default_stability_threshold: float = DEFAULT_STABILITY_THRESHOLD

    # ---- Step A: Intent decoding -----------------------------------------

    def decode_intent(self, intent: Any) -> IntentInterpretation:
        """Extract goal, constraints, and priority from raw intent.

        Accepts a string (treated as a goal description) or a mapping with the
        keys ``goal``, ``constraints``, ``priority``.
        """

        if isinstance(intent, IntentInterpretation):
            return intent
        if isinstance(intent, str):
            return IntentInterpretation(goal=intent, constraints={}, priority=0)
        if isinstance(intent, Mapping):
            goal = str(intent.get("goal", "")).strip()
            constraints_raw = intent.get("constraints", {}) or {}
            if not isinstance(constraints_raw, Mapping):
                raise TypeError("intent.constraints must be a mapping")
            priority = int(_coerce_float(intent.get("priority"), 0))
            return IntentInterpretation(
                goal=goal,
                constraints=dict(constraints_raw),
                priority=priority,
            )
        raise TypeError(
            "intent must be a string, mapping, or IntentInterpretation; "
            f"got {type(intent).__name__}"
        )

    # ---- Step B: World alignment -----------------------------------------

    @staticmethod
    def alignment_violations(
        intent: IntentInterpretation,
        world_state: Mapping[str, Any],
        system_limits: Mapping[str, Any],
    ) -> list[str]:
        """Return a list of human-readable constraint violations.

        A violation is reported when:

        - any value in ``world_state["violations"]`` is truthy, or
        - any constraint declared in ``system_limits["safety_bounds"]`` is
          breached by the corresponding ``world_state`` reading, or
        - an intent-level constraint of the form ``"<key>_max"`` /
          ``"<key>_min"`` is breached by ``world_state[key]``.
        """

        violations: list[str] = []

        explicit = world_state.get("violations") or []
        if isinstance(explicit, Mapping):
            iterable = (k for k, v in explicit.items() if v)
        else:
            iterable = (str(v) for v in explicit if v)
        violations.extend(str(v) for v in iterable)

        bounds = system_limits.get("safety_bounds") or {}
        if isinstance(bounds, Mapping):
            for key, bound in bounds.items():
                if key not in world_state:
                    continue
                value = _coerce_float(world_state[key], 0.0)
                if isinstance(bound, Mapping):
                    lo = _coerce_float(bound.get("min"), float("-inf"))
                    hi = _coerce_float(bound.get("max"), float("inf"))
                    if value < lo or value > hi:
                        violations.append(f"safety_bound:{key} out of [{lo}, {hi}] (value={value})")
                elif isinstance(bound, (list, tuple)) and len(bound) == 2:
                    lo, hi = (_coerce_float(bound[0]), _coerce_float(bound[1]))
                    if value < lo or value > hi:
                        violations.append(f"safety_bound:{key} out of [{lo}, {hi}] (value={value})")

        for key, value in intent.constraints.items():
            if key.endswith("_max") and key[:-4] in world_state:
                if _coerce_float(world_state[key[:-4]], 0.0) > _coerce_float(value):
                    violations.append(f"intent_constraint:{key} exceeded")
            elif (
                key.endswith("_min")
                and key[:-4] in world_state
                and _coerce_float(world_state[key[:-4]], 0.0) < _coerce_float(value)
            ):
                violations.append(f"intent_constraint:{key} underrun")

        return violations

    # ---- Step C/D: Simulate and synthesize -------------------------------

    def _stability_threshold(self, system_limits: Mapping[str, Any]) -> float:
        return _clamp01(
            _coerce_float(
                system_limits.get("stability_threshold"),
                self.default_stability_threshold,
            )
        )

    def _is_unsafe(
        self,
        violations: list[str],
        outcome: PredictedOutcome,
        stability_threshold: float,
    ) -> bool:
        if violations:
            return True
        if outcome.risk > self.risk_threshold:
            return True
        return outcome.stability_score < stability_threshold

    def _confidence(self, outcome: PredictedOutcome, stability_threshold: float) -> float:
        margin = outcome.stability_score - stability_threshold
        # Confidence proportional to stability margin and inversely to risk.
        confidence = _clamp01(max(0.0, margin) * (1.0 - outcome.risk))
        return confidence

    def synthesize(
        self,
        intent: IntentInterpretation,
        world_state: Mapping[str, Any],
        system_limits: Mapping[str, Any],
    ) -> tuple[SelectedAction, PredictedOutcome]:
        """Run simulation then choose an action under the strict policy.

        Returns the selected action and the prediction it was based on. The
        prediction is *always* computed before the action is finalized, in
        accordance with the predict-before-act rule.
        """

        violations = self.alignment_violations(intent, world_state, system_limits)
        stability_threshold = self._stability_threshold(system_limits)

        # Always propose + simulate first; predict-before-act is mandatory.
        proposed_type, proposed_magnitude = self.proposer(intent, world_state, system_limits)
        if proposed_type not in ACTION_TYPES:
            raise ValueError(
                f"proposer returned unsupported type {proposed_type!r}; "
                f"must be one of {ACTION_TYPES}"
            )
        proposed_magnitude = _clamp01(_coerce_float(proposed_magnitude, 0.0))

        # Cap proposed magnitude by system_limits.max_magnitude (never exceed limits).
        max_magnitude = _clamp01(_coerce_float(system_limits.get("max_magnitude"), 1.0))
        proposed_magnitude = min(proposed_magnitude, max_magnitude)

        outcome = self.simulator(proposed_type, proposed_magnitude, world_state, system_limits)

        # (1) Safety dominance -- abort when unsafe.
        if self._is_unsafe(violations, outcome, stability_threshold):
            abort_outcome = self.simulator("abort", 0.0, world_state, system_limits)
            return (
                SelectedAction(
                    type="abort",
                    magnitude=0.0,
                    confidence=self._confidence(abort_outcome, stability_threshold),
                ),
                abort_outcome,
            )

        # (2) Stability awareness -- drop ``control`` to ``hold``/``adjust``
        # when stability is not strong.
        action_type = proposed_type
        magnitude = proposed_magnitude
        if action_type == "control" and outcome.stability_score < STRONG_STABILITY_BAND:
            action_type = "adjust"

        # (3) Minimal intervention -- scale magnitude down with risk; if risk
        # is high relative to stability margin, hold.
        margin = outcome.stability_score - stability_threshold
        if margin <= 0.0:
            action_type, magnitude = "hold", 0.0
        else:
            magnitude = _clamp01(magnitude * (1.0 - outcome.risk))
            if magnitude == 0.0 and action_type in {"control", "adjust"}:
                action_type = "hold"

        # Re-simulate the *finalized* action so the returned prediction
        # reflects what will actually be executed.
        final_outcome = self.simulator(action_type, magnitude, world_state, system_limits)

        # (4) Predict-before-act: confidence is 0 if no prediction; we always
        # have one here, so derive it from stability margin and risk.
        confidence = self._confidence(final_outcome, stability_threshold)

        return (
            SelectedAction(type=action_type, magnitude=magnitude, confidence=confidence),
            final_outcome,
        )

    # ---- Public entry point ---------------------------------------------

    def step(self, payload: Mapping[str, Any]) -> ControlDecision:
        """Run a full perception -> model -> action cycle.

        ``payload`` must contain ``intent``, ``world_state``, and
        ``system_limits`` keys (as in the spec).
        """

        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        if "intent" not in payload:
            raise KeyError("payload missing required key 'intent'")

        intent = self.decode_intent(payload["intent"])
        world_state_raw = payload.get("world_state", {}) or {}
        system_limits_raw = payload.get("system_limits", {}) or {}
        if not isinstance(world_state_raw, Mapping):
            raise TypeError("payload.world_state must be a mapping")
        if not isinstance(system_limits_raw, Mapping):
            raise TypeError("payload.system_limits must be a mapping")
        world_state = dict(world_state_raw)
        system_limits = dict(system_limits_raw)

        action, outcome = self.synthesize(intent, world_state, system_limits)
        fallback = self._build_fallback(intent, action, outcome, system_limits)

        return ControlDecision(
            intent_interpretation=intent,
            selected_action=action,
            predicted_outcome=outcome,
            fallback_plan=fallback,
        )

    # ---- Fallback --------------------------------------------------------

    def _build_fallback(
        self,
        intent: IntentInterpretation,
        action: SelectedAction,
        outcome: PredictedOutcome,
        system_limits: Mapping[str, Any],
    ) -> FallbackPlan:
        """Mandatory fallback plan; never omitted.

        The trigger condition references the same thresholds enforced by
        safety dominance. The fallback action is always the safest possible
        downgrade of the current action.
        """

        stability_threshold = self._stability_threshold(system_limits)
        trigger = (
            f"risk > {self.risk_threshold:.2f} "
            f"or stability_score < {stability_threshold:.2f} "
            "or new constraint violation detected"
        )
        # Downgrade ladder: control -> adjust -> hold -> abort.
        downgrade = {
            "control": "adjust",
            "adjust": "hold",
            "hold": "abort",
            "abort": "abort",
        }
        return FallbackPlan(
            trigger_condition=trigger,
            action=downgrade[action.type],
        )

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

        # 1. INTENT DECODING -- never invent missing values
        interp = self._decode_intent(intent)

        # 2. STATE ALIGNMENT
        feasibility, conflicts = self._align_state(world_state, interp["constraints"])

        # 3 + 4. CANDIDATE SET + SIMULATION (mandatory)
        candidates = self._build_candidates(intent, world_state, system_limits)

        # 5. ADMISSIBLE SET
        r_max = float(system_limits.get("risk_threshold", 1.0))
        sigma_min = float(system_limits.get("stability_threshold", 0.0))
        admissible = [
            c
            for c in candidates
            if c.type != "abort" and c.risk <= r_max and c.stability >= sigma_min
        ]

        # 6. POLICY -- lexicographic: Safety > Stability > Minimal-intervention
        chosen = self._apply_policy(
            candidates=candidates,
            admissible=admissible,
            feasibility=feasibility,
            conflicts=conflicts,
        )

        # 7. CONFIDENCE -- proportional to stability, zero if no valid prediction
        confidence = 0.0 if chosen.predicted is None else max(0.0, min(1.0, chosen.stability))
        if chosen.type == "abort":
            confidence = 0.0

        # 8. FALLBACK PLAN
        fallback = self._build_fallback_from_candidate(chosen, feasibility, conflicts)

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
        # NEVER hallucinate -- only echo provided keys; missing -> safe defaults
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
        # RULE 1 -- SAFETY DOMINANCE: infeasible state OR no admissible -> abort
        if feasibility == 0 or conflicts or not admissible:
            for cand in candidates:
                if cand.type == "abort":
                    return cand

        # RULE 2 -- STABILITY AWARENESS: prefer hold/adjust if any admissible
        # candidate has stability below 0.5 (system near instability).
        near_instability = any(c.stability < 0.5 for c in admissible)
        if near_instability:
            stable_choices = [c for c in admissible if c.type in ("hold", "adjust")]
            if stable_choices:
                # RULE 3 within stable subset
                return min(stable_choices, key=lambda c: (c.magnitude, c.type))

        # RULE 3 -- MINIMAL INTERVENTION: argmin magnitude among admissible
        return min(admissible, key=lambda c: (c.magnitude, c.type))

    @staticmethod
    def _build_fallback_from_candidate(
        chosen: _Candidate,
        feasibility: int,
        conflicts: list[str],
    ) -> dict[str, Any]:
        if chosen.type == "abort":
            return {
                "trigger_condition": (
                    "constraint_violation" if conflicts else "no_admissible_action"
                ),
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
    """Stability = 1 - normalized distance to ``intent.constraints`` numeric targets."""

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

    - ``{"min": x}`` / ``{"max": y}`` / ``{"min": x, "max": y}`` -- applied to
      numeric value with same key (caller passes name as constraint key).
    - ``{"key": k, "min"/"max": ...}`` -- explicit key mapping.
    - ``{"equals": v}`` -- strict equality.
    - bool/None -- assumed satisfied.
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
    "DEFAULT_RISK_THRESHOLD",
    "DEFAULT_STABILITY_THRESHOLD",
    "STRONG_STABILITY_BAND",
    "IntentInterpretation",
    "SelectedAction",
    "PredictedOutcome",
    "FallbackPlan",
    "ControlDecision",
    "Simulator",
    "Proposer",
    "default_simulator",
    "default_proposer",
    "RealityInterfaceController",
    "ActionType",
    "LinearSimulator",
    "Proposer",
    "RICDecision",
    "RealityInterfaceController",
    "Simulator",
    "StaticProposer",
]
