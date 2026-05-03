"""falsifier.py — the per-domain falsification harness contract.

The plan mandates that every headline capability ships with a falsification
harness analogous to ``run_falsification.py`` and reports a verdict in the
A / A0 / B taxonomy with a signal-to-noise ratio (SNR) and overlap with
ground truth.  This module provides the *contract* every domain falsifier
must conform to, plus a lightweight reference verdict object usable by the
``qratum falsify`` CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Protocol

#: Verdict A — the falsifier could not refute the claim; signal is well above
#: noise *and* its geometry matches the predicted ground truth.
VERDICT_A: str = "A"

#: Verdict A0 — high SNR is achieved but only via a confound (e.g. amplitude
#: damping or control performance), so the *causal* claim cannot be sustained.
#: This is the result currently reported by the multi-qubit causal-claim
#: rescreen and is preserved here as a first-class outcome.
VERDICT_A0: str = "A0"

#: Verdict B — the falsifier refuted the claim outright (signal indistinguishable
#: from null, or overlap with ground truth below threshold).
VERDICT_B: str = "B"

_VALID_VERDICTS = frozenset({VERDICT_A, VERDICT_A0, VERDICT_B})


@dataclass(frozen=True)
class FalsificationVerdict:
    """Result of a domain-specific falsification harness.

    Attributes:
        domain: short symbolic name (``"multi_qubit"``, ``"plasma"``, …).
        verdict: one of :data:`VERDICT_A`, :data:`VERDICT_A0`, :data:`VERDICT_B`.
        snr: signal-to-noise ratio in linear units (≥ 0).
        overlap_with_ground: fidelity-style scalar in ``[0, 1]``.
        notes: free-form rationale; *excluded* from any equality / hash check
            via the dataclass `compare=False`.
        artifacts: optional pointers to per-run outputs (file paths, hashes,
            ledger ranges).  Treated opaquely by the framework.
    """

    domain: str
    verdict: str
    snr: float
    overlap_with_ground: float
    notes: str = field(default="", compare=False)
    artifacts: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if self.verdict not in _VALID_VERDICTS:
            raise ValueError(
                f"verdict must be one of {sorted(_VALID_VERDICTS)}, got {self.verdict!r}"
            )
        if self.snr < 0:
            raise ValueError(f"snr must be non-negative, got {self.snr}")
        if not (0.0 <= self.overlap_with_ground <= 1.0):
            raise ValueError(
                f"overlap_with_ground must be in [0,1], got {self.overlap_with_ground}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "verdict": self.verdict,
            "snr": float(self.snr),
            "overlap_with_ground": float(self.overlap_with_ground),
            "notes": self.notes,
            "artifacts": dict(self.artifacts),
        }


class Falsifier(Protocol):
    """Per-domain falsification harness contract.

    Implementations live in domain packages (``quasim.ciir.multi_qubit``,
    ``quasim.ciir.plasma``, ``qrVITRA``, …).  Each must expose:

    * a stable ``domain`` string,
    * a deterministic, seedable ``run`` method returning a
      :class:`FalsificationVerdict`.

    The CLI's ``qratum falsify`` discovers and runs registered falsifiers.
    """

    domain: str

    def run(self, *, seed: int = 0, **kwargs: Any) -> FalsificationVerdict: ...


# ---------------------------------------------------------------------------
# Trivial reference falsifier — used by tests and by `qratum falsify --demo`.
# ---------------------------------------------------------------------------


class _DemoFalsifier:
    """Reference falsifier returning a fixed Verdict A0.

    Intentionally trivial: it lets the CLI smoke-test exercise the
    falsification surface end-to-end without depending on the heavyweight
    multi-qubit or plasma harnesses.  Real domain falsifiers replace this
    object via dependency injection.
    """

    domain: str = "demo"

    def __init__(
        self,
        verdict: str = VERDICT_A0,
        snr: float = 1.0e6,
        overlap_with_ground: float = 0.999,
    ) -> None:
        self._verdict = verdict
        self._snr = snr
        self._overlap = overlap_with_ground

    def run(self, *, seed: int = 0, **kwargs: Any) -> FalsificationVerdict:
        return FalsificationVerdict(
            domain=self.domain,
            verdict=self._verdict,
            snr=self._snr,
            overlap_with_ground=self._overlap,
            notes="reference demo falsifier; replace with a real harness",
            artifacts={"seed": seed},
        )


__all__ = [
    "VERDICT_A",
    "VERDICT_A0",
    "VERDICT_B",
    "FalsificationVerdict",
    "Falsifier",
    "_DemoFalsifier",
]
