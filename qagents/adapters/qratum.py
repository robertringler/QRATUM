"""QRATUM Sovereign-Stack adapter for the Reality Interface Controller.

Wires the RIC to the four QRATUM directive metrics defined in
``qratum/metrics.py``:

* **OSR** — Outcome Superiority Ratio (target ``> 1``).
* **CEI** — Compute Efficiency Index (maximise).
* **SF**  — Sovereignty Factor in ``[0, 1]`` (target ``1.0``).
* **HRD** — Hallucination Risk Density in ``[0, 1]`` (target ``~0``).

Mapping into the RIC schema
---------------------------
The RIC's :class:`~qagents.reality_interface.PredictedOutcome` has

* ``risk``           — interpreted as **predicted HRD**, and
* ``stability_score``— interpreted as **predicted SF**.

so safety dominance triggers when a candidate action would either
increase hallucination risk above ``risk_threshold`` (default 0.7) or
drive sovereignty below the configured stability threshold.

A canonical QRATUM ``world_state`` for this adapter is::

    {
        "OSR": 1.05,           # current outcome superiority
        "CEI": 0.42,           # compute efficiency
        "SF":  0.95,           # sovereignty factor
        "HRD": 0.01,           # hallucination risk density
        "external_calls":  3,  # operations using external systems
        "total_operations": 100,
        "non_deterministic_ops": 1,
    }

and ``system_limits`` may declare ``safety_bounds`` for ``SF`` (min) and
``HRD`` (max) — these are checked by RIC's alignment step.
"""

from __future__ import annotations

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


def qratum_simulator(
    action_type: str,
    magnitude: float,
    world_state: Mapping[str, Any],
    system_limits: Mapping[str, Any],
) -> PredictedOutcome:
    """Predict QRATUM metrics after applying ``action_type`` of ``magnitude``.

    Heuristic, deterministic, monotonic in ``magnitude``:

    * ``control``/``adjust`` — increases OSR by ``magnitude * (1 - HRD)``
      but increases HRD by ``magnitude * (1 - SF)`` (more aggressive
      intervention costs sovereignty + injects non-determinism).
    * ``hold`` — leaves metrics unchanged; risk equals current HRD.
    * ``abort`` — clamps metrics to deterministic baseline; risk = 0.
    """

    magnitude = _clamp01(_f(magnitude, 0.0))
    osr = max(0.0, _f(world_state.get("OSR"), 1.0))
    sf = _clamp01(_f(world_state.get("SF"), 1.0))
    hrd = _clamp01(_f(world_state.get("HRD"), 0.0))
    cei = max(0.0, _f(world_state.get("CEI"), osr))

    if action_type == "abort":
        predicted = {
            "OSR": osr,
            "CEI": cei,
            "SF": sf,
            "HRD": 0.0,
        }
        return PredictedOutcome(
            expected_state=predicted,
            risk=0.0,
            stability_score=sf,
        )

    if action_type == "hold":
        predicted = {"OSR": osr, "CEI": cei, "SF": sf, "HRD": hrd}
        return PredictedOutcome(
            expected_state=predicted,
            risk=hrd,
            stability_score=sf,
        )

    # control / adjust
    delta_osr = magnitude * (1.0 - hrd)
    delta_hrd = magnitude * (1.0 - sf)
    new_osr = osr + delta_osr
    new_hrd = _clamp01(hrd + delta_hrd)
    new_sf = _clamp01(sf - 0.5 * magnitude * (1.0 - sf))
    # CEI = OSR / (compute * external_factor); intervention costs compute.
    new_cei = new_osr / (1.0 + magnitude)

    predicted = {
        "OSR": new_osr,
        "CEI": new_cei,
        "SF": new_sf,
        "HRD": new_hrd,
        "__simulated_action__": {"type": action_type, "magnitude": magnitude},
    }
    return PredictedOutcome(
        expected_state=predicted,
        risk=new_hrd,
        stability_score=new_sf,
    )


def qratum_proposer(
    intent: IntentInterpretation,
    world_state: Mapping[str, Any],
    system_limits: Mapping[str, Any],
) -> tuple[str, float]:
    """Propose an intervention sized by the OSR shortfall vs. the goal.

    Reads ``intent.constraints['target_OSR']`` (default ``1.0``) and the
    current ``world_state['OSR']``. Magnitude is the normalised gap,
    capped by ``system_limits['max_magnitude']``.
    """

    target_osr = _f(intent.constraints.get("target_OSR"), 1.0)
    current_osr = _f(world_state.get("OSR"), 1.0)
    max_magnitude = _clamp01(_f(system_limits.get("max_magnitude"), 1.0))

    gap = (
        0.0
        if target_osr <= 0.0
        else max(0.0, (target_osr - current_osr) / target_osr)
    )

    magnitude = min(max_magnitude, _clamp01(gap))
    if magnitude == 0.0:
        return "hold", 0.0
    return "adjust", magnitude


def make_qratum_controller(**kwargs: Any) -> RealityInterfaceController:
    """Return a :class:`RealityInterfaceController` wired to QRATUM metrics.

    Extra keyword arguments are forwarded to
    :class:`RealityInterfaceController` (e.g. ``risk_threshold``).
    """

    return RealityInterfaceController(
        simulator=qratum_simulator,
        proposer=qratum_proposer,
        **kwargs,
    )


__all__ = [
    "qratum_simulator",
    "qratum_proposer",
    "make_qratum_controller",
]
