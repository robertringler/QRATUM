"""Bridge wiring the Reality Interface Controller to the CIIR engine.

The bridge:

1. Maps a CIIR ``MetricsSnapshot`` → RIC ``world_state``.
2. Builds default ``intent`` (drive loss + constraint norms toward 0,
   keep purity high) and ``system_limits`` (safety bounds + thresholds).
3. Applies the RIC ``selected_action`` back onto the CIIR engine:

   - ``control``  → step + scale learning rate up by ``magnitude``
   - ``adjust``   → step + scale learning rate down (gentle)
   - ``hold``     → no step (skip)
   - ``abort``    → mark engine as not running (graceful stop)

The bridge is fully deterministic (no randomness) and works with any
:class:`qagents.reality_interface.Proposer`, including the LLM-backed
proposers in :mod:`qagents.llm_backends`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from qagents.reality_interface import (
    RealityInterfaceController,
    RICDecision,
    Proposer,
)


# ---------------------------------------------------------------------------
# Default intent / system limits
# ---------------------------------------------------------------------------


def default_intent(priority: float = 0.7) -> dict[str, Any]:
    """Default control intent for CIIR: minimize loss, hold purity high."""

    return {
        "goal": "drive_ciir_loss_toward_zero",
        "constraints": {
            "loss": {"key": "total_loss", "min": 0.0, "max": 1e6, "target": 0.0},
            "purity": {"key": "purity", "min": 0.0, "max": 1.0, "target": 1.0},
            "grad_norm": {"key": "gradient_norm", "min": 0.0, "max": 1e6, "target": 0.0},
        },
        "priority": float(priority),
    }


def default_system_limits(
    risk_threshold: float = 0.5,
    stability_threshold: float = 0.05,
) -> dict[str, Any]:
    """Default safety/stability envelope for CIIR control."""

    return {
        "safety_bounds": {
            "total_loss": {"min": 0.0, "max": 1e3},
            "purity": {"min": 0.0, "max": 1.0001},
            "entropy": {"min": 0.0, "max": 10.0},
        },
        "stability_threshold": float(stability_threshold),
        "risk_threshold": float(risk_threshold),
    }


# ---------------------------------------------------------------------------
# Snapshot → world_state
# ---------------------------------------------------------------------------


def snapshot_to_world_state(snapshot: Any) -> dict[str, Any]:
    """Project a CIIR ``MetricsSnapshot`` into RIC ``world_state``.

    Accepts any object exposing the snapshot fields; missing fields default
    to ``0.0`` so the bridge degrades gracefully on test stubs.
    """

    def _f(name: str, default: float = 0.0) -> float:
        value = getattr(snapshot, name, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    cv = getattr(snapshot, "constraint_violations", []) or []
    try:
        cv_norm = float(sum(abs(float(x)) for x in cv))
    except (TypeError, ValueError):
        cv_norm = 0.0

    return {
        "step": int(getattr(snapshot, "step", 0) or 0),
        "total_loss": _f("total_loss"),
        "constraint_loss": _f("constraint_loss"),
        "observer_loss": _f("observer_loss"),
        "entropy_loss": _f("entropy_loss"),
        "gradient_norm": _f("gradient_norm"),
        "purity": _f("purity"),
        "entropy": _f("entropy"),
        "constraint_norm": cv_norm,
    }


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------


@dataclass
class CIIRRICRunResult:
    proposer_name: str
    n_steps_executed: int
    decisions: list[RICDecision]
    snapshots: list[Any]
    aborted_at: int | None = None
    final_loss: float = 0.0
    final_purity: float = 0.0
    action_counts: dict[str, int] = field(default_factory=dict)


class CIIRRICBridge:
    """Couples a :class:`RealityInterfaceController` to a CIIR engine."""

    def __init__(
        self,
        engine: Any,
        proposer: Proposer,
        intent: Mapping[str, Any] | None = None,
        system_limits: Mapping[str, Any] | None = None,
        proposer_name: str | None = None,
    ) -> None:
        self.engine = engine
        self.ric = RealityInterfaceController(proposer=proposer)
        self.intent: dict[str, Any] = dict(intent or default_intent())
        self.system_limits: dict[str, Any] = dict(system_limits or default_system_limits())
        self.proposer_name = proposer_name or getattr(proposer, "name", proposer.__class__.__name__)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, n_steps: int) -> CIIRRICRunResult:
        """Run the CIIR engine for up to ``n_steps``, gated by the RIC."""

        decisions: list[RICDecision] = []
        snapshots: list[Any] = []
        action_counts: dict[str, int] = {"control": 0, "adjust": 0, "hold": 0, "abort": 0}
        aborted_at: int | None = None

        # Snapshot of initial state (use a single forward step to sample state).
        snapshot = self.engine.step()
        snapshots.append(snapshot)
        executed = 1

        for i in range(n_steps - 1):
            world_state = snapshot_to_world_state(snapshot)
            decision = self.ric.decide(self.intent, world_state, self.system_limits)
            decisions.append(decision)

            action = decision.selected_action
            atype = str(action.get("type", "hold"))
            magnitude = float(action.get("magnitude", 0.0))
            action_counts[atype] = action_counts.get(atype, 0) + 1

            if atype == "abort":
                aborted_at = int(getattr(snapshot, "step", executed))
                # Mark engine as non-running if it supports the flag
                if hasattr(self.engine, "running"):
                    try:
                        self.engine.running = False
                    except Exception:
                        pass
                break

            # ``control`` / ``adjust`` actuate by scaling the learning rate;
            # ``hold`` advances the simulation tick but applies no control input.
            self._apply_lr_scale(atype, magnitude)
            snapshot = self.engine.step()
            snapshots.append(snapshot)
            executed += 1

        final_loss = float(getattr(snapshot, "total_loss", 0.0) or 0.0)
        final_purity = float(getattr(snapshot, "purity", 0.0) or 0.0)
        return CIIRRICRunResult(
            proposer_name=self.proposer_name,
            n_steps_executed=executed,
            decisions=decisions,
            snapshots=snapshots,
            aborted_at=aborted_at,
            final_loss=final_loss,
            final_purity=final_purity,
            action_counts=action_counts,
        )

    # ------------------------------------------------------------------
    # Actuation
    # ------------------------------------------------------------------

    def _apply_lr_scale(self, action_type: str, magnitude: float) -> None:
        """Scale the engine learning rate to actuate ``control`` / ``adjust``.

        The mapping is intentionally conservative: ``adjust`` shrinks lr,
        ``control`` nudges it up, both within a 4× envelope around the
        engine's configured base rate.
        """

        cfg = getattr(self.engine, "config", None)
        if cfg is None or not hasattr(cfg, "learning_rate"):
            return
        base = float(getattr(cfg, "_ric_base_lr", cfg.learning_rate))
        if not hasattr(cfg, "_ric_base_lr"):
            try:
                setattr(cfg, "_ric_base_lr", base)
            except Exception:
                return
        # Clamp magnitude into [0, 1]; map to scale ∈ [0.25, 4.0]
        m = max(0.0, min(1.0, float(magnitude)))
        if action_type == "control":
            scale = 1.0 + 3.0 * m  # up to 4×
        elif action_type == "adjust":
            scale = max(0.25, 1.0 - 0.75 * m)  # down to 0.25×
        else:
            scale = 1.0
        try:
            cfg.learning_rate = base * scale
        except Exception:
            pass


__all__ = [
    "CIIRRICBridge",
    "CIIRRICRunResult",
    "default_intent",
    "default_system_limits",
    "snapshot_to_world_state",
]
