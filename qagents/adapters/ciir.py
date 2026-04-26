"""CIIR variational-dynamics adapter for the Reality Interface Controller.

Wires the RIC to the CIIR loss functional from ``quasim/ciir/loss.py``:

.. math::

    L(\\rho) = \\sum_i \\lambda_i C_i(\\rho)^2
            + \\sum_j \\mu_j |\\langle O_j\\rangle_\\rho - y_j|^2
            + \\gamma \\cdot \\Omega(\\rho)

The adapter does **not** import :mod:`numpy` or :mod:`quasim.ciir`; it
operates on the CIIR canonical world-state schema so it can be used in
minimal environments and as a deterministic stand-in alongside the real
projected-gradient evolution.

Canonical CIIR ``world_state``::

    {
        "loss":                 0.42,   # current L(rho)
        "constraint_violation": 0.05,   # max_i |C_i(rho)|
        "entropy":              1.10,   # -Tr(rho log rho), regulariser
        "gradient_norm":        0.30,   # ||grad L|| for step sizing
    }

Mapping into the RIC schema
---------------------------
* ``risk``            — max constraint violation after the candidate step.
* ``stability_score`` — :math:`\\exp(-L)`, so :math:`L \\to 0` ⇒ score → 1.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from qagents.reality_interface import (
    IntentInterpretation,
    PredictedOutcome,
    RealityInterfaceController,
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


def _stability_from_loss(loss: float) -> float:
    """Map non-negative ``loss`` to ``[0, 1]`` via ``exp(-loss)``."""

    return math.exp(-max(0.0, loss))


def ciir_simulator(
    action_type: str,
    magnitude: float,
    world_state: Mapping[str, Any],
    system_limits: Mapping[str, Any],
) -> PredictedOutcome:
    """Predict the CIIR loss / constraint state after one projected step.

    The candidate ``magnitude`` is the learning-rate scaling
    :math:`\\eta \\in [0, 1]`. The model assumes:

    * ``loss``                  ↓  by ``magnitude * gradient_norm``
      (one gradient step, ignoring curvature),
    * ``constraint_violation``  ↓  multiplicatively by
      ``(1 - magnitude * projection_strength)`` — the projection
      operator pulls the iterate toward the constraint cone D7.4.
    * ``hold`` leaves the state untouched; ``abort`` zeros the step.
    """

    magnitude = _clamp01(_f(magnitude, 0.0))
    loss = max(0.0, _f(world_state.get("loss"), 0.0))
    violation = _clamp01(_f(world_state.get("constraint_violation"), 0.0))
    grad_norm = max(0.0, _f(world_state.get("gradient_norm"), loss))
    entropy = _f(world_state.get("entropy"), 0.0)
    projection_strength = _clamp01(
        _f(system_limits.get("projection_strength"), 1.0)
    )

    if action_type == "abort":
        return PredictedOutcome(
            expected_state={
                "loss": loss,
                "constraint_violation": violation,
                "entropy": entropy,
            },
            risk=violation,
            stability_score=_clamp01(_stability_from_loss(loss)),
        )

    if action_type == "hold":
        return PredictedOutcome(
            expected_state={
                "loss": loss,
                "constraint_violation": violation,
                "entropy": entropy,
            },
            risk=violation,
            stability_score=_clamp01(_stability_from_loss(loss)),
        )

    # control / adjust — one projected gradient step.
    new_loss = max(0.0, loss - magnitude * grad_norm)
    new_violation = _clamp01(violation * (1.0 - magnitude * projection_strength))

    return PredictedOutcome(
        expected_state={
            "loss": new_loss,
            "constraint_violation": new_violation,
            "entropy": entropy,
            "__simulated_action__": {"type": action_type, "magnitude": magnitude},
        },
        risk=new_violation,
        stability_score=_clamp01(_stability_from_loss(new_loss)),
    )


def ciir_proposer(
    intent: IntentInterpretation,
    world_state: Mapping[str, Any],
    system_limits: Mapping[str, Any],
) -> tuple[str, float]:
    """Propose ``adjust`` with a step size that scales with current violation.

    A larger constraint violation ⇒ a smaller step (closer to a pure
    projection move), capped by ``system_limits['max_magnitude']``.
    """

    violation = _clamp01(_f(world_state.get("constraint_violation"), 0.0))
    max_magnitude = _clamp01(_f(system_limits.get("max_magnitude"), 1.0))
    # Conservative inverse scaling: η = 1 / (1 + violation * scale).
    scale = max(0.0, _f(intent.constraints.get("step_scale"), 4.0))
    eta = 1.0 / (1.0 + violation * scale)
    magnitude = min(max_magnitude, _clamp01(eta))
    if magnitude == 0.0:
        return "hold", 0.0
    return "adjust", magnitude


def make_ciir_controller(**kwargs: Any) -> RealityInterfaceController:
    """Return a :class:`RealityInterfaceController` wired to CIIR dynamics."""

    return RealityInterfaceController(
        simulator=ciir_simulator,
        proposer=ciir_proposer,
        **kwargs,
    )


__all__ = [
    "ciir_simulator",
    "ciir_proposer",
    "make_ciir_controller",
]
