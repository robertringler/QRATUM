"""Reality Interface Controller v2 — trajectory-aware bounded-stochastic agent.

This module implements the v2 prompt at
``qagents/prompts/reality_interface_controller_v2.md``.

It upgrades the v1 deterministic policy
(:class:`qagents.reality_interface.RealityInterfaceController`) into a
**bounded stochastic model-predictive controller with memory-aware
trajectory selection**, while preserving:

- the strict 4-key output envelope (`intent_interpretation`,
  `selected_action`, `predicted_outcome`, `fallback_plan`);
- the lexicographic Safety > Stability > Minimal-intervention ordering
  (extended with Goal-alignment and Exploration-value);
- pluggable :class:`Simulator` and :class:`Proposer` interfaces.

What it adds (the "alive" parts)
--------------------------------
- **State enrichment**: derives instability trends, constraint saturation,
  and directional gradients from a sliding ``history`` window.
- **k-step trajectory rollout**: every candidate action is simulated
  ``k_horizon`` steps forward and scored by the *worst* admissibility
  along the rollout (Rule 5).
- **Bounded stochastic perturbations**: extra magnitude perturbations
  (seeded ``random.Random``) are added to the proposer set so the
  controller is no longer locked to a single magnitude per type.
- **HOLD-streak pressure (Rule 2)**: when the recent-action history shows
  a hold-streak ≥ ``hold_streak_threshold``, exploration value is boosted
  for non-hold actions to break dead policy lock.
- **Anti-unanimity / no-deterministic-collapse (Rule 5)**: if all admissible
  trajectories are within ``unanimity_eps`` on every score, a small
  bounded entropy perturbation is added to break ties (still safety-gated).
- **Temporal awareness (Rule 3)**: stability and risk are aggregated over
  the rollout (mean for stability, max for risk) — never scored in
  isolation.

Determinism contract
--------------------
Given the same ``(intent, world_state, system_limits, history, seed)``,
the controller produces identical output. Stochasticity is *bounded* and
*reproducible* — this preserves test stability while enabling emergent
behavior across runs with different seeds.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from qagents.reality_interface import (ACTION_TYPES, ActionType,
                                       LinearSimulator, Proposer, RICDecision,
                                       Simulator, StaticProposer,
                                       _constraint_satisfied, _default_risk,
                                       _default_stability)

# ---------------------------------------------------------------------------
# Trajectory dataclasses
# ---------------------------------------------------------------------------


@dataclass
class _TrajStep:
    state: dict[str, Any]
    risk: float
    stability: float


@dataclass
class _Trajectory:
    type: ActionType
    magnitude: float
    perturbation: float  # the added stochastic delta (0 for deterministic seed)
    steps: list[_TrajStep]

    @property
    def max_risk(self) -> float:
        return max((s.risk for s in self.steps), default=0.0)

    @property
    def mean_stability(self) -> float:
        if not self.steps:
            return 0.0
        return sum(s.stability for s in self.steps) / len(self.steps)

    @property
    def stability_variance(self) -> float:
        if len(self.steps) < 2:
            return 0.0
        mean = self.mean_stability
        return sum((s.stability - mean) ** 2 for s in self.steps) / len(self.steps)

    @property
    def predicted_terminal(self) -> dict[str, Any]:
        return dict(self.steps[-1].state) if self.steps else {}


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


@dataclass
class RICv2History:
    """Sliding-window memory of the last ``maxlen`` decisions.

    Used to satisfy Rule 2 (No Dead Policy Lock) and Rule 3 (Temporal
    Awareness). The bridge or runner is responsible for calling
    :meth:`record` after each decision.
    """

    maxlen: int = 16
    recent_actions: list[ActionType] = field(default_factory=list)
    recent_metrics: list[dict[str, float]] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)

    def record(self, action_type: ActionType, metrics: Mapping[str, float] | None = None) -> None:
        self.recent_actions.append(action_type)
        if len(self.recent_actions) > self.maxlen:
            self.recent_actions = self.recent_actions[-self.maxlen :]
        if metrics is not None:
            self.recent_metrics.append({k: float(v) for k, v in metrics.items()})
            if len(self.recent_metrics) > self.maxlen:
                self.recent_metrics = self.recent_metrics[-self.maxlen :]

    def add_failure(self, mode: str) -> None:
        self.failure_modes.append(mode)
        if len(self.failure_modes) > self.maxlen:
            self.failure_modes = self.failure_modes[-self.maxlen :]

    def hold_streak(self) -> int:
        """Number of consecutive trailing ``hold`` actions."""

        n = 0
        for a in reversed(self.recent_actions):
            if a == "hold":
                n += 1
            else:
                break
        return n

    def trend_signals(self) -> dict[str, float]:
        """First-difference summary over recorded numeric metrics."""

        if len(self.recent_metrics) < 2:
            return {}
        first = self.recent_metrics[0]
        last = self.recent_metrics[-1]
        return {
            k: float(last.get(k, 0.0) - first.get(k, 0.0))
            for k in last.keys() & first.keys()
        }


# ---------------------------------------------------------------------------
# v2 Controller
# ---------------------------------------------------------------------------


class RICv2Controller:
    """Trajectory-aware bounded-stochastic Reality Interface Controller.

    Parameters
    ----------
    simulator
        Single-step simulator. Defaults to :class:`LinearSimulator`. The
        controller composes ``k_horizon`` calls into a forward rollout.
    proposer
        Proposer of magnitudes per action type. Defaults to
        :class:`StaticProposer`. Use :mod:`qagents.llm_backends` for LLMs.
    k_horizon
        Number of forward steps simulated per candidate (Rule 3 — temporal
        awareness). Default 5.
    n_perturbations
        Bounded stochastic perturbations added per action type beyond the
        proposer's deterministic magnitude. Default 2.
    perturbation_scale
        Maximum |Δm| applied to a proposed magnitude. Default 0.25.
    hold_streak_threshold
        Number of consecutive ``hold`` decisions after which exploration
        pressure activates (Rule 2). Default 3.
    exploration_bonus
        Score bonus added per perturbation step when exploration pressure
        is active. Bounded so it never overrides safety/stability ranking.
        Default 0.05.
    unanimity_eps
        Tolerance below which all admissible trajectories are deemed
        "equivalent", triggering Rule 5 entropy injection. Default 1e-3.
    seed
        Seed for the bounded RNG. Same seed + same inputs ⇒ same output.
    """

    def __init__(
        self,
        simulator: Simulator | None = None,
        proposer: Proposer | None = None,
        risk_estimator: Callable[[Mapping[str, Any], Mapping[str, Any]], float] | None = None,
        stability_estimator: Callable[[Mapping[str, Any], Mapping[str, Any]], float] | None = None,
        k_horizon: int = 5,
        n_perturbations: int = 2,
        perturbation_scale: float = 0.25,
        hold_streak_threshold: int = 3,
        exploration_bonus: float = 0.05,
        unanimity_eps: float = 1e-3,
        seed: int = 0,
    ) -> None:
        self.simulator: Simulator = simulator or LinearSimulator()
        self.proposer: Proposer = proposer or StaticProposer()
        self._risk = risk_estimator or _default_risk
        self._stability = stability_estimator or _default_stability
        self.k_horizon = max(1, int(k_horizon))
        self.n_perturbations = max(0, int(n_perturbations))
        self.perturbation_scale = max(0.0, float(perturbation_scale))
        self.hold_streak_threshold = max(1, int(hold_streak_threshold))
        self.exploration_bonus = max(0.0, float(exploration_bonus))
        self.unanimity_eps = max(0.0, float(unanimity_eps))
        self._seed = int(seed)
        self._rng = random.Random(self._seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(
        self,
        intent: Mapping[str, Any],
        world_state: Mapping[str, Any],
        system_limits: Mapping[str, Any],
        history: RICv2History | None = None,
    ) -> RICDecision:
        """Run the v2 pipeline and return a strict 4-key decision."""

        history = history or RICv2History()

        # 1. Intent interpretation (no hallucination)
        interp = self._decode_intent(intent)

        # 2. State enrichment
        enriched = self._enrich_state(world_state, history)

        # 3. Candidate generation (deterministic + bounded stochastic)
        magnitudes = self._magnitudes(intent, world_state, system_limits)
        candidates = self._build_candidates(magnitudes)

        # 4. Trajectory simulation (k-step rollout, mandatory)
        trajectories = [
            self._rollout(world_state, intent, system_limits, c)
            for c in candidates
        ]

        # 5. Admissibility filter
        feasibility, conflicts = self._align_state(world_state, interp["constraints"])
        r_max = float(system_limits.get("risk_threshold", 1.0))
        sigma_min = float(system_limits.get("stability_threshold", 0.0))
        admissible = [
            t for t in trajectories
            if t.type != "abort"
            and t.max_risk <= r_max
            and t.mean_stability >= sigma_min
        ]

        # 6. Lexicographic selection
        chosen = self._select(
            trajectories=trajectories,
            admissible=admissible,
            history=history,
            feasibility=feasibility,
            conflicts=conflicts,
        )

        # 7. Entropy injection if unanimity (no deterministic collapse)
        chosen = self._maybe_inject_entropy(chosen, admissible)

        # 8. Output
        return self._format_output(chosen, interp, conflicts, feasibility, enriched)

    # ------------------------------------------------------------------
    # Step 1: intent
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_intent(intent: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "goal": intent.get("goal", ""),
            "constraints": dict(intent.get("constraints", {})),
            "priority": float(intent.get("priority", 0.0)),
        }

    # ------------------------------------------------------------------
    # Step 2: state enrichment
    # ------------------------------------------------------------------

    @staticmethod
    def _enrich_state(
        world_state: Mapping[str, Any],
        history: RICv2History,
    ) -> dict[str, Any]:
        trends = history.trend_signals()
        # Crude saturation proxy: |constraint_norm| ratio to its safety max
        cn = float(world_state.get("constraint_norm", 0.0) or 0.0)
        return {
            "trends": trends,
            "hold_streak": history.hold_streak(),
            "constraint_pressure": cn,
            "directional_gradient": float(world_state.get("gradient_norm", 0.0) or 0.0),
            "temporal_instability": abs(trends.get("total_loss", 0.0)) if trends else 0.0,
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
        return (0 if conflicts else 1), conflicts

    # ------------------------------------------------------------------
    # Step 3: candidates with bounded stochastic perturbations
    # ------------------------------------------------------------------

    def _magnitudes(
        self,
        intent: Mapping[str, Any],
        world_state: Mapping[str, Any],
        system_limits: Mapping[str, Any],
    ) -> dict[ActionType, float]:
        try:
            mags = self.proposer.propose(intent, world_state, system_limits)
        except Exception:
            mags = {}
        out: dict[ActionType, float] = {}
        for atype in ACTION_TYPES:
            try:
                out[atype] = max(0.0, float(mags.get(atype, 0.0)))
            except (TypeError, ValueError):
                out[atype] = 0.0
        return out

    def _build_candidates(
        self,
        magnitudes: Mapping[ActionType, float],
    ) -> list[tuple[ActionType, float, float]]:
        """Return ``(type, magnitude, perturbation)`` tuples.

        Always includes the deterministic proposer magnitude (perturbation=0)
        and ``n_perturbations`` bounded perturbations per non-abort type.
        """

        candidates: list[tuple[ActionType, float, float]] = []
        for atype in ACTION_TYPES:
            base = float(magnitudes.get(atype, 0.0))
            candidates.append((atype, base, 0.0))
            if atype == "abort" or self.n_perturbations == 0:
                continue
            for _ in range(self.n_perturbations):
                delta = self._rng.uniform(-self.perturbation_scale, self.perturbation_scale)
                m = max(0.0, base + delta)
                candidates.append((atype, m, delta))
        return candidates

    # ------------------------------------------------------------------
    # Step 4: k-step rollout
    # ------------------------------------------------------------------

    def _rollout(
        self,
        world_state: Mapping[str, Any],
        intent: Mapping[str, Any],
        system_limits: Mapping[str, Any],
        candidate: tuple[ActionType, float, float],
    ) -> _Trajectory:
        atype, magnitude, perturbation = candidate
        action = {"type": atype, "magnitude": magnitude}
        steps: list[_TrajStep] = []
        state: dict[str, Any] = dict(world_state)
        for _ in range(self.k_horizon):
            try:
                state = dict(self.simulator.predict(state, action))
            except Exception:
                # If the simulator fails, treat the trajectory as maximally risky.
                steps.append(_TrajStep(state=dict(state), risk=1.0, stability=0.0))
                break
            r = max(0.0, min(1.0, float(self._risk(state, system_limits))))
            s = max(0.0, min(1.0, float(self._stability(state, intent))))
            steps.append(_TrajStep(state=state, risk=r, stability=s))
        return _Trajectory(
            type=atype,
            magnitude=magnitude,
            perturbation=perturbation,
            steps=steps,
        )

    # ------------------------------------------------------------------
    # Step 6: lexicographic selection
    # ------------------------------------------------------------------

    def _select(
        self,
        trajectories: list[_Trajectory],
        admissible: list[_Trajectory],
        history: RICv2History,
        feasibility: int,
        conflicts: list[str],
    ) -> _Trajectory:
        # SAFETY DOMINANCE: infeasible state OR no admissible → abort
        if feasibility == 0 or conflicts or not admissible:
            for t in trajectories:
                if t.type == "abort":
                    return t

        # Compute Rule 2 exploration pressure
        streak = history.hold_streak()
        explore_active = streak >= self.hold_streak_threshold

        def score(t: _Trajectory) -> tuple[float, float, float, float, float]:
            # Lexicographic key — *lower is better*.
            #   1. -mean_stability                       (max stability)
            #   2.  stability_variance                   (min oscillation)
            #   3.  magnitude                            (min intervention)
            #   4. -goal_alignment                       (max alignment ≈ -mean_stability proxy)
            #   5. -exploration_value                    (max exploration when active)
            stab = t.mean_stability
            var = t.stability_variance
            cost = t.magnitude
            goal = stab  # use mean_stability as goal-alignment proxy
            explore = abs(t.perturbation) if explore_active else 0.0
            # Exploration bonus is bounded — cannot override safety/stability.
            explore_score = -self.exploration_bonus * explore
            # Rule 2: when stuck in HOLD streak, penalize hold itself slightly
            hold_penalty = 0.0
            if explore_active and t.type == "hold":
                hold_penalty = self.exploration_bonus
            return (-stab + hold_penalty, var, cost, -goal, explore_score)

        admissible_sorted = sorted(admissible, key=score)
        return admissible_sorted[0]

    # ------------------------------------------------------------------
    # Step 7: entropy injection on unanimity
    # ------------------------------------------------------------------

    def _maybe_inject_entropy(
        self,
        chosen: _Trajectory,
        admissible: list[_Trajectory],
    ) -> _Trajectory:
        if chosen.type == "abort" or len(admissible) < 2:
            return chosen
        # Detect unanimity: all admissible trajectories have nearly-identical
        # mean_stability, max_risk, magnitude.
        ref = admissible[0]
        unanimous = all(
            abs(t.mean_stability - ref.mean_stability) <= self.unanimity_eps
            and abs(t.max_risk - ref.max_risk) <= self.unanimity_eps
            and abs(t.magnitude - ref.magnitude) <= self.unanimity_eps
            for t in admissible
        )
        if not unanimous:
            return chosen
        # Pick a different admissible trajectory than chosen (if any) using RNG.
        # This breaks deterministic collapse without violating any safety bound
        # (all candidates are admissible).
        alternatives = [t for t in admissible if t is not chosen] or [chosen]
        return self._rng.choice(alternatives)

    # ------------------------------------------------------------------
    # Step 8: output
    # ------------------------------------------------------------------

    def _format_output(
        self,
        chosen: _Trajectory,
        interp: dict[str, Any],
        conflicts: list[str],
        feasibility: int,
        enriched: dict[str, Any],
    ) -> RICDecision:
        # Safety/abort confidence rules carried over from v1.
        confidence = 0.0
        if chosen.type != "abort" and chosen.steps:
            confidence = max(0.0, min(1.0, chosen.mean_stability))

        # Trajectory summary fields (Rule 8)
        if chosen.steps:
            stab_seq = [round(s.stability, 6) for s in chosen.steps]
            risk_seq = [round(s.risk, 6) for s in chosen.steps]
            traj_summary = (
                f"{chosen.type}@m={chosen.magnitude:.4f} "
                f"+δ={chosen.perturbation:+.4f}, k={len(chosen.steps)}"
            )
            stab_proj = (
                f"mean={chosen.mean_stability:.4f}, var={chosen.stability_variance:.4f}, "
                f"trace={stab_seq}"
            )
            risk_profile = f"max={chosen.max_risk:.4f}, trace={risk_seq}"
        else:
            traj_summary = f"{chosen.type}@m={chosen.magnitude:.4f} (no rollout)"
            stab_proj = "n/a"
            risk_profile = "n/a"

        if chosen.type == "abort":
            fallback = {
                "trigger_conditions": (
                    "constraint_violation" if conflicts else "no_admissible_trajectory"
                ),
                "safe_action": "hold",
            }
        elif chosen.mean_stability < 0.5 or feasibility == 0:
            fallback = {"trigger_conditions": "stability_drop", "safe_action": "hold"}
        else:
            fallback = {"trigger_conditions": "stability_drop", "safe_action": "abort"}

        return RICDecision(
            intent_interpretation={**interp, "enrichment": enriched},
            selected_action={
                "type": chosen.type,
                "magnitude": float(chosen.magnitude),
                "confidence": float(confidence),
                "perturbation": float(chosen.perturbation),
            },
            predicted_outcome={
                "expected_state": chosen.predicted_terminal,
                "trajectory_summary": traj_summary,
                "stability_projection": stab_proj,
                "risk_profile": risk_profile,
                "risk": float(chosen.max_risk),
                "stability_score": float(chosen.mean_stability),
            },
            fallback_plan=fallback,
        )


__all__ = [
    "RICv2Controller",
    "RICv2History",
]
