"""operator.py — the canonical end-to-end pipeline (§3 of the plan).

This module implements the *single* request path every workload traverses:

    Pilot Intent
        → RIC.parse_intent
        → Φ-gate (CIIR constraints)
        → CRS T_C
        → control-geometry / safety scoring (delegated to backend)
        → MVRI / domain executor
        → Observer
        → Merkle-chained TraceEntry

The default backend, :class:`StrictCIIRBackend`, wraps the strict
``qagents.ciir_crs_ric.executor.run_step`` (the canonical 9-step loop with
named Type I–IV failure modes).  Domain-specific backends (qubit, plasma,
genomic, consensus, …) plug in by implementing the
:class:`OperatorBackend` protocol — yielding the **one adapter contract**
described in §4 of the plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Protocol, Tuple

from qratum_framework.trace import (
    MerkleLedger,
    UnifiedTraceEntry,
    hash_state,
)

# ---------------------------------------------------------------------------
# Backend protocol — the unified adapter contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackendStepResult:
    """One step of execution as seen by the framework.

    A backend reports the *observable* effect of executing one intent against
    its internal state, in terms of fields the unified TraceEntry needs.

    All fields are JSON-friendly so they can be hashed and persisted without
    any backend-specific serialisation logic leaking into the spine.
    """

    intent: Optional[str]
    action: str
    status: str
    failure_mode: Optional[str]
    pre_state: Mapping[str, Any]
    post_state: Optional[Mapping[str, Any]]
    phi_verdict: bool
    violated_constraints: Tuple[str, ...] = ()
    metrics: Mapping[str, float] = field(default_factory=dict)


class OperatorBackend(Protocol):
    """The single adapter contract.

    Domain backends (qubit / plasma / genomic / consensus / pilot) implement
    this Protocol.  The Operator is otherwise backend-agnostic.
    """

    name: str

    def initial_state(self) -> Mapping[str, Any]:
        """Return the canonical initial state as a dict."""
        ...

    def step(self, raw_intent: str, step_index: int) -> BackendStepResult:
        """Run one tick.  MUST be deterministic given (state, raw_intent).

        Implementations carry their own internal mutable state between calls;
        the Operator never inspects it directly — only via ``BackendStepResult``.
        """
        ...


# ---------------------------------------------------------------------------
# Default backend — strict CIIR–CRS–RIC executor
# ---------------------------------------------------------------------------


class StrictCIIRBackend:
    """Backend wrapping ``qagents.ciir_crs_ric.executor.run_step``.

    This is the reference implementation of an :class:`OperatorBackend`: the
    strict 9-step loop with explicit Type I–IV failure modes.  Used by the
    ``qratum`` CLI's ``run`` and ``simulate`` subcommands by default.

    The backend is deliberately created lazily — its dependency on the
    ``qagents.ciir_crs_ric`` package is imported inside ``__init__`` so that
    ``qratum_framework`` itself can be imported in a minimal environment.
    """

    name: str = "strict_ciir_crs_ric"

    def __init__(self, initial: Optional[Any] = None) -> None:
        # Lazy import: keep qratum_framework importable even if the strict
        # executor's transitive deps are unavailable in some sandbox.
        from qagents.ciir_crs_ric.ciir import State

        self._State = State
        self._state: Any = (
            initial if initial is not None else State(cpu_used=0, mem_used=0, status="idle")
        )

    # --- OperatorBackend protocol --------------------------------------

    def initial_state(self) -> Mapping[str, Any]:
        return _state_to_dict(self._state)

    def step(self, raw_intent: str, step_index: int) -> BackendStepResult:
        from qagents.ciir_crs_ric.executor import run_step

        pre = self._state
        next_state, _obs, entry = run_step(pre, raw_intent, step_index=step_index)

        violated: Tuple[str, ...] = ()
        if entry.failure is not None:
            details = entry.failure.details or {}
            v = details.get("violated", [])
            if isinstance(v, (list, tuple)):
                violated = tuple(str(x) for x in v)

        # Φ verdict: True iff status is OK and a fresh post-state was produced.
        phi = entry.status == "OK" and entry.state_t1 is not None

        post_state_dict: Optional[Mapping[str, Any]] = None
        if entry.state_t1 is not None:
            post_state_dict = _state_to_dict(entry.state_t1)
            self._state = entry.state_t1
        # On failure we hold state (deterministic fallback per the plan).

        result = BackendStepResult(
            intent=entry.intent.name if entry.intent is not None else None,
            action=entry.action.name,
            status=entry.status,
            failure_mode=(entry.failure.name if entry.failure is not None else None),
            pre_state=_state_to_dict(pre),
            post_state=post_state_dict,
            phi_verdict=phi,
            violated_constraints=violated,
            metrics={
                "admissible_count": float(len(entry.admissible)),
            },
        )
        return result


def _state_to_dict(state: Any) -> Dict[str, Any]:
    """Best-effort projection of a backend state into a JSON-friendly dict.

    Used by :class:`StrictCIIRBackend` to expose its internal frozen
    dataclass without leaking the type into the framework.
    """
    if hasattr(state, "__dataclass_fields__"):
        return {k: getattr(state, k) for k in state.__dataclass_fields__}
    if hasattr(state, "_asdict"):
        return dict(state._asdict())
    if isinstance(state, Mapping):
        return dict(state)
    return {"repr": repr(state)}


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------


@dataclass
class OperatorResult:
    """Outcome of an Operator run."""

    backend_name: str
    ledger: MerkleLedger
    final_state: Mapping[str, Any]
    success: bool
    failure_count: int

    def summary(self) -> Dict[str, Any]:
        statuses: Dict[str, int] = {}
        for e in self.ledger.entries:
            statuses[e.status] = statuses.get(e.status, 0) + 1
        return {
            "backend": self.backend_name,
            "ledger_height": self.ledger.height(),
            "ledger_head": self.ledger.head(),
            "verified": self.ledger.verify(),
            "statuses": statuses,
            "success": self.success,
            "failure_count": self.failure_count,
        }


class Operator:
    """Drives a backend through a sequence of intents and emits a ledger.

    This is the *only* legitimate way to run a workload through the
    framework.  Every workload — quantum, plasma, genomic, consensus,
    agentic — passes through here, so every workload yields an
    interchangeable Merkle-chained ledger.
    """

    def __init__(
        self,
        backend: OperatorBackend,
        ledger: Optional[MerkleLedger] = None,
        run_id: Optional[str] = None,
    ) -> None:
        self.backend = backend
        self.ledger = ledger if ledger is not None else MerkleLedger()
        self.run_id = run_id or "default"

    # --- single tick ----------------------------------------------------

    def step(self, raw_intent: str, step_index: int) -> UnifiedTraceEntry:
        result = self.backend.step(raw_intent, step_index)
        entry = UnifiedTraceEntry(
            step=step_index,
            intent_raw=raw_intent,
            intent=result.intent,
            action=result.action,
            status=result.status,
            failure_mode=result.failure_mode,
            pre_state_hash=hash_state(result.pre_state),
            post_state_hash=(
                hash_state(result.post_state) if result.post_state is not None else None
            ),
            phi_verdict=result.phi_verdict,
            violated_constraints=tuple(result.violated_constraints),
            metrics=dict(result.metrics),
            metadata={"backend": self.backend.name, "run_id": self.run_id},
        )
        self.ledger.append(entry)
        return entry

    # --- multi-step driver ---------------------------------------------

    def run(self, intents: Iterable[str]) -> OperatorResult:
        """Run a sequence of raw intents and return an :class:`OperatorResult`."""
        intents = list(intents)
        for i, raw in enumerate(intents):
            self.step(raw, step_index=i)

        statuses = [e.status for e in self.ledger.entries]
        failures = sum(1 for s in statuses if s != "OK")

        # final_state: re-derive from backend if available
        final_state: Mapping[str, Any]
        if hasattr(self.backend, "initial_state"):
            try:
                final_state = self.backend.initial_state()  # type: ignore[assignment]
                # If the backend updates its own internal state in place,
                # ``initial_state`` would still return the constructor default.
                # Backends that want to expose their final state should
                # override ``current_state``; we honour that here.
                if hasattr(self.backend, "current_state"):
                    final_state = self.backend.current_state()  # type: ignore[attr-defined]
            except Exception:
                final_state = {}
        else:
            final_state = {}

        return OperatorResult(
            backend_name=self.backend.name,
            ledger=self.ledger,
            final_state=final_state,
            success=failures == 0,
            failure_count=failures,
        )


__all__ = [
    "BackendStepResult",
    "Operator",
    "OperatorBackend",
    "OperatorResult",
    "StrictCIIRBackend",
]
