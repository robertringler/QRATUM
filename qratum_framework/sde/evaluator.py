"""evaluator.py — alarm-tier state machine over :class:`DriftReading`.

The :class:`AlarmEvaluator` consumes :class:`DriftReading` objects and
emits :class:`DriftEvent` objects whenever the alarm tier changes (or
re-asserts at ALERT/HALT).  The state machine implements the
escalation and de-escalation rules from §FORMAL SPECIFICATION:

* OK    — ``d_t ≤ θ_low``
* WATCH — ``θ_low  < d_t ≤ θ_high``
* ALERT — ``θ_high < d_t ≤ θ_halt``
* HALT  — ``d_t > θ_halt`` AND ``Δ²d_t > 0``

Escalation requires ``N_debounce`` consecutive readings at-or-above
the candidate tier.  De-escalation is immediate.  The HALT tier
additionally requires positive drift acceleration so that a transient
spike at HALT amplitude does not trip the actuator.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from qratum_framework.sde.scorers import DriftReading


class AlarmTier(str, Enum):
    """Ordered alarm tiers (string-valued for canonical-JSON output)."""

    OK = "OK"
    WATCH = "WATCH"
    ALERT = "ALERT"
    HALT = "HALT"

    @property
    def rank(self) -> int:
        return _TIER_RANK[self]


_TIER_RANK: Dict[AlarmTier, int] = {
    AlarmTier.OK: 0,
    AlarmTier.WATCH: 1,
    AlarmTier.ALERT: 2,
    AlarmTier.HALT: 3,
}


@dataclass(frozen=True)
class AlarmThresholds:
    """Tier cut-points, all in the units of the chosen scorer."""

    theta_low: float = 0.05
    theta_high: float = 0.20
    theta_halt: float = 0.50

    def __post_init__(self) -> None:
        if not (0.0 <= self.theta_low <= self.theta_high <= self.theta_halt):
            raise ValueError(
                "thresholds must satisfy 0 ≤ theta_low ≤ theta_high ≤ theta_halt; "
                f"got low={self.theta_low}, high={self.theta_high}, "
                f"halt={self.theta_halt}"
            )

    def to_dict(self) -> Dict[str, float]:
        return {
            "theta_low": float(self.theta_low),
            "theta_high": float(self.theta_high),
            "theta_halt": float(self.theta_halt),
        }


@dataclass(frozen=True)
class DriftEvent:
    """One alarm-tier transition (or re-assertion at ALERT/HALT)."""

    tier: AlarmTier
    d_t: float
    delta_d: float
    delta2_d: float
    tick: int
    window_hash: str
    scorer_name: str
    flags: Tuple[str, ...] = ()
    previous_tier: AlarmTier = AlarmTier.OK

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier.value,
            "previous_tier": self.previous_tier.value,
            "d_t": float(self.d_t),
            "delta_d": float(self.delta_d),
            "delta2_d": float(self.delta2_d),
            "tick": int(self.tick),
            "window_hash": self.window_hash,
            "scorer_name": self.scorer_name,
            "flags": list(self.flags),
        }


class AlarmEvaluator:
    """Stateful debouncer + tier classifier.

    Parameters
    ----------
    thresholds:
        :class:`AlarmThresholds` defining the OK/WATCH/ALERT/HALT cuts.
    n_debounce:
        Number of consecutive readings at-or-above a candidate tier
        required before the evaluator escalates to it.  De-escalation
        is unconditional.

    The evaluator emits :class:`DriftEvent` objects on tier *changes*.
    It additionally re-emits at ALERT and HALT every tick that those
    tiers persist, because downstream actuators need a heartbeat — a
    HALT that fires once and goes silent would let the pipeline
    forget that the boundary is still being approached.

    The OK tier is suppressed unless the evaluator is *transitioning
    down* from a higher tier; pure OK ticks at the start of a stream
    therefore produce no events at all.
    """

    def __init__(
        self,
        thresholds: AlarmThresholds = AlarmThresholds(),
        *,
        n_debounce: int = 3,
    ) -> None:
        if n_debounce < 1:
            raise ValueError(f"n_debounce must be >= 1, got {n_debounce}")
        self._th = thresholds
        self._n_debounce = int(n_debounce)
        self._current: AlarmTier = AlarmTier.OK
        self._candidate: Optional[AlarmTier] = None
        self._candidate_count: int = 0
        self._history: List[DriftEvent] = []

    # --- properties -----------------------------------------------------

    @property
    def thresholds(self) -> AlarmThresholds:
        return self._th

    @property
    def n_debounce(self) -> int:
        return self._n_debounce

    @property
    def current_tier(self) -> AlarmTier:
        return self._current

    @property
    def history(self) -> List[DriftEvent]:
        return list(self._history)

    # --- main API -------------------------------------------------------

    def evaluate(self, reading: DriftReading, *, window_hash: str) -> Optional[DriftEvent]:
        """Return a :class:`DriftEvent` on tier change, else ``None``.

        ``window_hash`` is captured so that consumers (the Merkle
        ledger, audit replay) can correlate the event with the exact
        token-value sequence that produced it.
        """
        candidate = self._classify_raw(reading)
        new_tier = self._debounce(candidate)

        emit = False
        if new_tier != self._current:
            # Transition (up or down) — always emit.
            emit = True
        elif new_tier in (AlarmTier.ALERT, AlarmTier.HALT):
            # Heartbeat at high tiers so the actuator path keeps firing.
            emit = True
        # OK is suppressed unless transitioning (handled above).

        previous_tier = self._current
        self._current = new_tier

        if not emit:
            return None

        event = DriftEvent(
            tier=new_tier,
            d_t=reading.d_t,
            delta_d=reading.delta_d,
            delta2_d=reading.delta2_d,
            tick=reading.tick,
            window_hash=window_hash,
            scorer_name=reading.scorer_name,
            flags=reading.flags,
            previous_tier=previous_tier,
        )
        self._history.append(event)
        return event

    # --- internals ------------------------------------------------------

    def _classify_raw(self, reading: DriftReading) -> AlarmTier:
        d = reading.d_t
        if d <= self._th.theta_low:
            return AlarmTier.OK
        if d <= self._th.theta_high:
            return AlarmTier.WATCH
        if d <= self._th.theta_halt:
            return AlarmTier.ALERT
        # d > theta_halt: HALT requires positive acceleration.
        if reading.delta2_d > 0.0:
            return AlarmTier.HALT
        # Sustained high amplitude without acceleration: stay at ALERT.
        return AlarmTier.ALERT

    def _debounce(self, candidate: AlarmTier) -> AlarmTier:
        # De-escalation (toward OK) is immediate.
        if candidate.rank < self._current.rank:
            self._candidate = None
            self._candidate_count = 0
            return candidate
        # Same tier as current: nothing to debounce.
        if candidate.rank == self._current.rank:
            self._candidate = None
            self._candidate_count = 0
            return self._current
        # Escalation candidate: require N_debounce consecutive hits at
        # or above the candidate tier.
        if self._candidate is None or candidate.rank != self._candidate.rank:
            self._candidate = candidate
            self._candidate_count = 1
        else:
            self._candidate_count += 1
        if self._candidate_count >= self._n_debounce:
            new_tier = self._candidate
            self._candidate = None
            self._candidate_count = 0
            return new_tier
        return self._current


__all__ = ["AlarmTier", "AlarmThresholds", "DriftEvent", "AlarmEvaluator"]
