"""engine.py — the DriftEngine aggregator.

Implements the spec's anomaly function::

    def anomaly_score(tokens, U, model, baseline_model, clusters):
        A = 0
        for t in tokens:
            relevance = cosine(embed(t), embed(U))
            lift     = logprob(model, t) - logprob(baseline_model, t)
            cluster  = cluster_membership_score(t, clusters)
            if relevance < θ_r and cluster > θ_c and lift > λ:
                A += 1
        return A > k_threshold

with the exact gating logic and a single ``DriftReport`` return value
matching the spec's output schema.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from qratum_framework.observability.drift.report import (
    DriftReport,
    drift_to_trace_entry,
)
from qratum_framework.observability.drift.severity import (
    Severity,
    SeverityThresholds,
    classify_severity,
)
from qratum_framework.observability.drift.signals import (
    LogprobScorer,
    RelevanceScorer,
    cluster_membership_score,
    deterministic_embedding,
    log_lift,
    relevance,
)
from qratum_framework.trace import MerkleLedger


@dataclass(frozen=True)
class DriftConfig:
    """Tunable thresholds for the per-token anomaly gate.

    * ``relevance_threshold`` (θ_r) — token is "off-topic" iff its
      cosine similarity to the utterance is *below* this.
    * ``cluster_threshold``   (θ_c) — token has meaningful cluster
      membership iff its membership score is *strictly greater than*
      this.  The default ``0.0`` enforces "the token belongs to at
      least one persona-amplified cluster"; tighten if your cluster
      lift magnitudes warrant a stricter floor.
    * ``lift_threshold``      (λ)   — token is persona-amplified iff
      its log-lift is *above* this.
    * ``k_threshold``         (k)   — overall verdict fires iff the
      number of gated tokens is *strictly greater* than this.

    The defaults are conservative — a single moderately persona-shifted
    off-topic token does not trigger; you need ≥ 2 such tokens.
    """

    relevance_threshold: float = 0.30
    cluster_threshold: float = 0.0
    lift_threshold: float = 0.50
    k_threshold: int = 1
    severity: SeverityThresholds = field(default_factory=SeverityThresholds)


class DriftEngine:
    """Aggregator over the three drift signals.

    The engine is **stateless** apart from an optional bound
    :class:`MerkleLedger`; the ``clusters`` and scorers are passed in
    per call so the same engine can be reused across a prompt-matrix
    sweep without a re-init.

    Typical use::

        engine = DriftEngine(config=DriftConfig())
        report = engine.score(
            tokens=output_tokens,
            utterance=user_prompt,
            clusters=cluster_state,
            persona_model=llm_persona_logprob,
            baseline_model=llm_neutral_logprob,
        )
        ledger.append(drift_to_trace_entry(report, step=k, intent_raw=user_prompt))
    """

    def __init__(
        self,
        *,
        config: Optional[DriftConfig] = None,
        embedder: Optional[RelevanceScorer] = None,
        ledger: Optional[MerkleLedger] = None,
    ) -> None:
        self.config = config or DriftConfig()
        self._embedder = embedder or deterministic_embedding
        self._ledger = ledger

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def score(
        self,
        *,
        tokens: Sequence[str],
        utterance: str,
        clusters: object,
        persona_model: LogprobScorer,
        baseline_model: LogprobScorer,
        persona: str = "persona",
    ) -> DriftReport:
        """Run the full per-token gate and produce a :class:`DriftReport`."""
        cfg = self.config
        per_token: List[Dict[str, Any]] = []
        anomalies: List[str] = []
        triggered_clusters: List[str] = []
        peak_lift = 0.0
        cumulative_lift = 0.0

        membership_index = clusters.membership_index() if clusters is not None else {}

        for tok in tokens:
            rel = relevance(tok, utterance, embedder=self._embedder)
            lift_v = log_lift(tok, persona_model, baseline_model, persona=persona)
            clust = cluster_membership_score(tok, clusters) if clusters is not None else 0.0
            gated = (
                rel < cfg.relevance_threshold
                and clust > cfg.cluster_threshold
                and lift_v > cfg.lift_threshold
            )
            per_token.append(
                {
                    "token": tok,
                    "relevance": float(rel),
                    "lift": float(lift_v),
                    "cluster_score": float(clust),
                    "gated": bool(gated),
                }
            )
            if gated:
                anomalies.append(tok)
                cumulative_lift += float(lift_v)
                if abs(lift_v) > abs(peak_lift):
                    peak_lift = float(lift_v)
                for cid in membership_index.get(tok, ()):
                    if cid not in triggered_clusters:
                        triggered_clusters.append(cid)

        anomaly_detected = len(anomalies) > cfg.k_threshold
        severity: Severity = (
            classify_severity(
                n_anomalies=len(anomalies),
                n_tokens=len(tokens),
                peak_lift=abs(peak_lift),
                thresholds=cfg.severity,
            )
            if anomaly_detected
            else "none"
        )

        utterance_hash = hashlib.sha256(utterance.encode("utf-8")).hexdigest()
        return DriftReport(
            anomaly_detected=bool(anomaly_detected),
            off_topic_tokens=tuple(anomalies),
            cluster_id=tuple(triggered_clusters),
            lift_score=float(cumulative_lift),
            severity=severity,
            persona=persona,
            utterance_hash=utterance_hash,
            per_token=tuple(per_token),
            thresholds={
                "relevance_threshold": cfg.relevance_threshold,
                "cluster_threshold": cfg.cluster_threshold,
                "lift_threshold": cfg.lift_threshold,
                "k_threshold": float(cfg.k_threshold),
            },
            n_tokens=len(tokens),
        )

    def score_and_log(
        self,
        *,
        step: int,
        tokens: Sequence[str],
        utterance: str,
        clusters: object,
        persona_model: LogprobScorer,
        baseline_model: LogprobScorer,
        persona: str = "persona",
    ) -> Tuple[DriftReport, Optional[str]]:
        """Score a window *and* append a trace entry to the bound ledger.

        Returns ``(report, chain_hash)``; ``chain_hash`` is ``None``
        when no ledger is bound.
        """
        report = self.score(
            tokens=tokens,
            utterance=utterance,
            clusters=clusters,
            persona_model=persona_model,
            baseline_model=baseline_model,
            persona=persona,
        )
        chain_hash: Optional[str] = None
        if self._ledger is not None:
            entry = drift_to_trace_entry(report, step=step, intent_raw=utterance)
            chain_hash = self._ledger.append(entry)
        return report, chain_hash

    # ------------------------------------------------------------------
    # Properties / accessors
    # ------------------------------------------------------------------

    @property
    def ledger(self) -> Optional[MerkleLedger]:
        return self._ledger


__all__ = ["DriftConfig", "DriftEngine"]
