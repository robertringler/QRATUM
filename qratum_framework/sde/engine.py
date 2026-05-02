"""engine.py — the top-level :class:`StreamingDriftEngine`.

The engine wires the four SDE primitives together — buffer, scorer,
evaluator, and (optionally) a :class:`MerkleLedger` — and exposes a
single ``async ingest`` coroutine for stream consumption.

Backpressure: when the evaluator emits a HALT event the engine awaits
the supplied ``halt_callback`` *before* pulling the next token from
the source.  This is the only actuator privilege the SDE owns: it can
slow the producer down by holding the loop in the callback.  It never
mutates pipeline state directly and never silently swallows
exceptions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional

from qratum_framework.sde.buffer import RollingWindowBuffer
from qratum_framework.sde.evaluator import (
    AlarmEvaluator,
    AlarmTier,
    DriftEvent,
)
from qratum_framework.sde.scorers import DriftReading, DriftScorer
from qratum_framework.sde.tokens import Token, WindowSnapshot
from qratum_framework.trace import (
    MerkleLedger,
    UnifiedTraceEntry,
    hash_state,
)

#: A coroutine that the engine awaits whenever a HALT event is emitted.
HaltCallback = Callable[[DriftEvent], Awaitable[None]]


@dataclass(frozen=True)
class SDESnapshot:
    """Read-only summary of engine state, safe to take mid-ingest."""

    tier: AlarmTier
    d_t: float
    window_depth: int
    event_count: int
    halt_count: int
    tick: int
    scorer_name: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier.value,
            "d_t": float(self.d_t),
            "window_depth": int(self.window_depth),
            "event_count": int(self.event_count),
            "halt_count": int(self.halt_count),
            "tick": int(self.tick),
            "scorer_name": self.scorer_name,
        }


async def _noop_halt(_event: DriftEvent) -> None:
    """Default HALT callback — does nothing.  Used when the caller
    has not supplied an actuator hook."""
    return None


class StreamingDriftEngine:
    """Orchestrator: buffer → scorer → evaluator → (optional) ledger.

    Parameters
    ----------
    buffer:
        :class:`RollingWindowBuffer` for the persona stream.
    scorer:
        Any object satisfying the :class:`DriftScorer` protocol.
    evaluator:
        :class:`AlarmEvaluator` carrying the alarm tiers and debouncer.
    baseline_window:
        Frozen :class:`WindowSnapshot` representing ``W_0`` — the
        confirmed-admissible reference distribution.
    halt_callback:
        Awaitable invoked whenever a HALT event is emitted.  Receives
        the :class:`DriftEvent`.  The engine awaits the callback
        *before* ingesting the next token, providing natural
        backpressure.  Defaults to a no-op.
    ledger:
        Optional :class:`MerkleLedger`; when provided, every emitted
        :class:`DriftEvent` (not every reading) is appended as a
        :class:`UnifiedTraceEntry`.
    run_id:
        Free-form identifier stored in trace metadata.
    """

    def __init__(
        self,
        *,
        buffer: RollingWindowBuffer,
        scorer: DriftScorer,
        evaluator: AlarmEvaluator,
        baseline_window: WindowSnapshot,
        halt_callback: Optional[HaltCallback] = None,
        ledger: Optional[MerkleLedger] = None,
        run_id: str = "sde",
    ) -> None:
        if baseline_window is None or len(baseline_window) == 0:
            raise ValueError("baseline_window must be a non-empty WindowSnapshot")
        self._buffer = buffer
        self._scorer = scorer
        self._evaluator = evaluator
        self._baseline = baseline_window
        self._halt_callback: HaltCallback = halt_callback or _noop_halt
        self._ledger = ledger
        self._run_id = run_id

        self._last_reading: Optional[DriftReading] = None
        self._events: List[DriftEvent] = []
        self._halt_count: int = 0

    # --- properties -----------------------------------------------------

    @property
    def events(self) -> List[DriftEvent]:
        return list(self._events)

    @property
    def ledger(self) -> Optional[MerkleLedger]:
        return self._ledger

    # --- step API (synchronous, single-token) ---------------------------

    def step(self, token: Token) -> Optional[DriftEvent]:
        """Push one token and return the resulting event (if any).

        This is the synchronous core of the engine; :meth:`ingest`
        calls it under an async loop.  It is exposed publicly so that
        embedders driving the engine from a non-async context (tests,
        replay tools) can use it directly.
        """
        snapshot = self._buffer.push(token)
        if snapshot is None:
            return None
        reading = self._scorer.score(snapshot, self._baseline)
        self._last_reading = reading
        event = self._evaluator.evaluate(
            reading, window_hash=snapshot.canonical_hash()
        )
        if event is None:
            return None
        self._events.append(event)
        if event.tier == AlarmTier.HALT:
            self._halt_count += 1
        if self._ledger is not None:
            self._ledger.append(self._to_trace_entry(event))
        return event

    # --- async ingest ---------------------------------------------------

    async def ingest(self, stream: AsyncIterator[Token]) -> None:
        """Drain ``stream`` through the full SDE pipeline.

        On HALT events the engine awaits :attr:`halt_callback` before
        pulling the next token, so a slow callback naturally throttles
        ingestion.  Exceptions raised by the source iterator or the
        callback propagate — the SDE never silently swallows.
        """
        async for token in stream:
            event = self.step(token)
            if event is not None and event.tier == AlarmTier.HALT:
                await self._halt_callback(event)

    # --- introspection --------------------------------------------------

    def snapshot(self) -> SDESnapshot:
        """Return a frozen view of current engine state.

        Safe to call from another coroutine while ``ingest`` is
        running; nothing in the snapshot path mutates engine state.
        """
        d_t = self._last_reading.d_t if self._last_reading is not None else 0.0
        return SDESnapshot(
            tier=self._evaluator.current_tier,
            d_t=float(d_t),
            window_depth=len(self._buffer),
            event_count=len(self._events),
            halt_count=int(self._halt_count),
            tick=self._buffer.tick,
            scorer_name=self._scorer.name,
        )

    # --- ledger projection ---------------------------------------------

    def _to_trace_entry(self, event: DriftEvent) -> UnifiedTraceEntry:
        """Project a :class:`DriftEvent` onto the unified trace schema.

        The drift event is encoded as a synthetic step in the canonical
        9-step loop:

        * ``intent_raw`` / ``intent`` — the alarm-tier name.
        * ``action`` — ``SDE_HALT_INJECT`` for HALT, else
          ``SDE_OBSERVE``; this is the *only* place the SDE ever
          surfaces an actuator-shaped action.
        * ``status`` — ``OK`` for OK/WATCH/ALERT, ``TYPE_II`` for HALT
          (matches the CRS Type II "boundary breach" failure mode).
        * ``phi_verdict`` — ``False`` for HALT (boundary breached),
          ``True`` otherwise.
        * ``metrics`` — ``d_t``, ``delta_d``, ``delta2_d``.
        * ``metadata`` — the alarm tier (so it survives JSON export
          but does *not* perturb the canonical hash, per the
          determinism contract in ``qratum_framework.trace``).
        """
        is_halt = event.tier == AlarmTier.HALT
        action = "SDE_HALT_INJECT" if is_halt else "SDE_OBSERVE"
        status = "TYPE_II" if is_halt else "OK"
        failure_mode = "BOUNDARY_BREACH" if is_halt else None
        phi = not is_halt
        return UnifiedTraceEntry(
            step=event.tick,
            intent_raw=event.tier.value,
            intent=event.tier.value,
            action=action,
            status=status,
            failure_mode=failure_mode,
            pre_state_hash=event.window_hash,
            post_state_hash=event.window_hash,
            phi_verdict=phi,
            violated_constraints=("DRIFT_BOUNDARY",) if is_halt else (),
            metrics={
                "d_t": float(event.d_t),
                "delta_d": float(event.delta_d),
                "delta2_d": float(event.delta2_d),
            },
            metadata={
                "tier": event.tier.value,
                "previous_tier": event.previous_tier.value,
                "scorer": event.scorer_name,
                "run_id": self._run_id,
                "flags": list(event.flags),
            },
        )


__all__ = [
    "HaltCallback",
    "SDESnapshot",
    "StreamingDriftEngine",
]
