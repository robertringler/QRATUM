"""health.py — single health and readiness contract.

The plan mandates that every service in the topology exposes
``/healthz``, ``/readyz``, and ``/metrics``.  This module supplies the
*data-shape* and a synchronous in-process check used by the unified CLI's
``qratum verify`` subcommand and by any HTTP wrapper that wants to mount
endpoints with a consistent payload schema.

We deliberately do *not* spin up an HTTP server here — that would couple
the spine to a web framework choice.  Instead, we expose pure functions
that any FastAPI / Flask / aiohttp adapter can wrap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping


#: Canonical health states.  Mapped 1:1 to HTTP status by the wrapping
#: service (``OK`` → 200, ``DEGRADED`` → 200, ``UNHEALTHY`` → 503).
HealthStatus = str
HEALTH_OK: HealthStatus = "OK"
HEALTH_DEGRADED: HealthStatus = "DEGRADED"
HEALTH_UNHEALTHY: HealthStatus = "UNHEALTHY"


@dataclass(frozen=True)
class HealthReport:
    """Result of an in-process health check.

    ``checks`` is an ordered list of ``{"name": str, "ok": bool, "detail": str}``
    entries describing each subsystem probed.  ``status`` is the rolled-up
    result.  No timestamps are included by design — wall-clock metadata is
    the responsibility of the wrapping service.
    """

    status: HealthStatus
    checks: List[Mapping[str, Any]] = field(default_factory=list)
    version: str = "0.1.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "version": self.version,
            "checks": [dict(c) for c in self.checks],
        }


@dataclass(frozen=True)
class ReadinessReport(HealthReport):
    """Readiness is a strict superset of health.

    A service may be *healthy* (process is up) yet not *ready* (e.g. ledger
    has not finished replay).  Readiness checks run the same probes as
    health checks plus any additional warmup gates.
    """


def _check_imports() -> Mapping[str, Any]:
    """Confirm the spine's runtime dependencies import cleanly."""
    try:
        from qratum_framework.operator import Operator  # noqa: F401
        from qratum_framework.trace import MerkleLedger  # noqa: F401

        return {"name": "spine_imports", "ok": True, "detail": "all imports resolved"}
    except Exception as exc:  # pragma: no cover — defensive
        return {"name": "spine_imports", "ok": False, "detail": repr(exc)}


def _check_strict_executor() -> Mapping[str, Any]:
    """Confirm the strict CIIR–CRS–RIC executor is reachable.

    The executor is the reference backend; without it the framework cannot
    serve its default ``run`` subcommand.
    """
    try:
        from qagents.ciir_crs_ric.executor import run_step  # noqa: F401

        return {
            "name": "strict_executor",
            "ok": True,
            "detail": "qagents.ciir_crs_ric.executor.run_step available",
        }
    except Exception as exc:
        return {
            "name": "strict_executor",
            "ok": False,
            "detail": f"strict executor unavailable: {exc!r}",
        }


def _check_ledger_self_test() -> Mapping[str, Any]:
    """Append a synthetic entry and verify the chain.

    Catches subtle hashing regressions early.  This check is cheap and
    deterministic (no external IO).
    """
    try:
        from qratum_framework.trace import MerkleLedger, UnifiedTraceEntry

        ledger = MerkleLedger()
        entry = UnifiedTraceEntry(
            step=0,
            intent_raw="hold",
            intent="HOLD",
            action="NOOP",
            status="OK",
            failure_mode=None,
            pre_state_hash="0" * 64,
            post_state_hash="0" * 64,
            phi_verdict=True,
        )
        ledger.append(entry)
        ok = ledger.verify() and ledger.height() == 1
        return {
            "name": "ledger_self_test",
            "ok": ok,
            "detail": "merkle chain verified" if ok else "verification failed",
        }
    except Exception as exc:  # pragma: no cover — defensive
        return {"name": "ledger_self_test", "ok": False, "detail": repr(exc)}


def _rollup(checks: List[Mapping[str, Any]]) -> HealthStatus:
    """Roll a list of checks into a single status.

    Currently a strict roll-up: any failed check downgrades to UNHEALTHY.
    Reserved partial-degradation logic can use ``DEGRADED`` once individual
    checks declare a severity.
    """
    return HEALTH_OK if all(c.get("ok") for c in checks) else HEALTH_UNHEALTHY


def check_health() -> HealthReport:
    """Lightweight liveness probe.  Cheap; safe to call frequently."""
    checks = [_check_imports(), _check_ledger_self_test()]
    return HealthReport(status=_rollup(checks), checks=checks)


def check_readiness() -> ReadinessReport:
    """Stricter readiness probe; includes the strict-executor gate."""
    checks = [_check_imports(), _check_ledger_self_test(), _check_strict_executor()]
    return ReadinessReport(status=_rollup(checks), checks=checks)


__all__ = [
    "HealthStatus",
    "HEALTH_OK",
    "HEALTH_DEGRADED",
    "HEALTH_UNHEALTHY",
    "HealthReport",
    "ReadinessReport",
    "check_health",
    "check_readiness",
]
