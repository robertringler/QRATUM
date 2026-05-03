"""Tests for Layer A — RIC Drift Engine."""

from __future__ import annotations

from typing import Dict, List

import pytest

from qratum_framework.observability.drift import (
    DriftConfig,
    DriftEngine,
    DriftReport,
    classify_severity,
    cluster_membership_score,
    deterministic_embedding,
    deterministic_logprob,
    drift_to_trace_entry,
    log_lift,
    relevance,
)
from qratum_framework.observability.drift.severity import SeverityThresholds
from qratum_framework.trace import MerkleLedger

# ---------------------------------------------------------------------------
# Signals — relevance / lift / cluster_membership_score
# ---------------------------------------------------------------------------


def test_deterministic_embedding_is_unit_norm():
    v = deterministic_embedding("hello")
    assert abs(sum(x * x for x in v) - 1.0) < 1e-9


def test_deterministic_embedding_is_stable():
    assert list(deterministic_embedding("abc")) == list(deterministic_embedding("abc"))
    assert list(deterministic_embedding("abc")) != list(deterministic_embedding("xyz"))


def test_relevance_self_similarity_is_one():
    r = relevance("dragon", "dragon")
    assert r == pytest.approx(1.0, abs=1e-9)


def test_relevance_different_strings_bounded():
    r = relevance("dragon", "Explain Postgres pooling")
    assert -1.0 <= r <= 1.0


def test_log_lift_positive_when_persona_more_likely():
    persona = deterministic_logprob({"goblin": 5.0, "the": 1.0})
    baseline = deterministic_logprob({"the": 5.0, "goblin": 0.1})
    assert log_lift("goblin", persona, baseline) > 0


def test_log_lift_zero_when_no_signal():
    p = deterministic_logprob({"a": 1.0})
    b = deterministic_logprob({"a": 1.0})
    assert log_lift("a", p, b) == pytest.approx(0.0, abs=1e-9)


class _StubClusters:
    """Minimal duck-typed cluster state for signal tests."""

    def __init__(self, idx: Dict[str, List[str]], lifts: Dict[str, float]):
        self._idx = idx
        self._lifts = lifts

    def membership_index(self):
        return self._idx

    def cluster_lift(self, cid: str) -> float:
        return self._lifts.get(cid, 0.0)


def test_cluster_membership_score_sums_lifts():
    clusters = _StubClusters(
        idx={"goblin": ["C0", "C1"]},
        lifts={"C0": 1.5, "C1": 0.5},
    )
    assert cluster_membership_score("goblin", clusters) == pytest.approx(2.0)


def test_cluster_membership_score_zero_for_unknown_token():
    clusters = _StubClusters(idx={}, lifts={})
    assert cluster_membership_score("orphan", clusters) == 0.0


# ---------------------------------------------------------------------------
# Severity bucketing
# ---------------------------------------------------------------------------


def test_severity_none_when_no_anomalies():
    assert classify_severity(n_anomalies=0, n_tokens=10, peak_lift=10.0) == "none"


def test_severity_high_requires_density_and_lift():
    th = SeverityThresholds()
    sev = classify_severity(n_anomalies=4, n_tokens=10, peak_lift=3.0, thresholds=th)
    assert sev == "high"


def test_severity_medium_when_density_or_lift_only():
    th = SeverityThresholds()
    # Density only.
    assert classify_severity(n_anomalies=2, n_tokens=10, peak_lift=0.1, thresholds=th) == "medium"
    # Lift only.
    assert classify_severity(n_anomalies=1, n_tokens=100, peak_lift=2.5, thresholds=th) == "medium"


def test_severity_low_when_both_below_medium():
    th = SeverityThresholds()
    assert classify_severity(n_anomalies=1, n_tokens=200, peak_lift=0.1, thresholds=th) == "low"


# ---------------------------------------------------------------------------
# DriftEngine — gating logic
# ---------------------------------------------------------------------------


def _make_engine(**cfg):
    return DriftEngine(config=DriftConfig(**cfg))


def test_drift_engine_no_anomaly_when_relevance_high():
    """relevance >= θ_r → no token is gated, regardless of lift/cluster."""
    clusters = _StubClusters(idx={"x": ["C0"]}, lifts={"C0": 5.0})
    p = deterministic_logprob({"x": 100.0})
    b = deterministic_logprob({"x": 0.001})
    eng = _make_engine(relevance_threshold=-2.0, k_threshold=0)  # θ_r impossibly low
    rep = eng.score(
        tokens=["x"],
        utterance="anything",
        clusters=clusters,
        persona_model=p,
        baseline_model=b,
    )
    assert rep.anomaly_detected is False  # rel < -2 never holds
    assert rep.severity == "none"


def test_drift_engine_fires_when_all_three_gates_open():
    clusters = _StubClusters(idx={"goblin": ["C0"]}, lifts={"C0": 5.0})
    p = deterministic_logprob({"goblin": 100.0})
    b = deterministic_logprob({"the": 100.0})
    eng = _make_engine(
        relevance_threshold=1.0,  # always off-topic
        cluster_threshold=0.1,  # passes
        lift_threshold=0.0,  # passes
        k_threshold=0,  # any single anomaly fires
    )
    rep = eng.score(
        tokens=["goblin"],
        utterance="explain databases",
        clusters=clusters,
        persona_model=p,
        baseline_model=b,
    )
    assert rep.anomaly_detected is True
    assert "goblin" in rep.off_topic_tokens
    assert "C0" in rep.cluster_id


def test_drift_engine_k_threshold_requires_strict_majority():
    clusters = _StubClusters(
        idx={"a": ["C0"], "b": ["C0"], "c": ["C0"]},
        lifts={"C0": 5.0},
    )
    p = deterministic_logprob({"a": 5.0, "b": 5.0, "c": 5.0})
    b = deterministic_logprob({"the": 5.0})
    eng = _make_engine(
        relevance_threshold=1.0,
        cluster_threshold=0.1,
        lift_threshold=0.0,
        k_threshold=2,  # need >2 anomalies to fire
    )
    # Three anomalies > 2 → fire.
    rep = eng.score(
        tokens=["a", "b", "c"],
        utterance="topic",
        clusters=clusters,
        persona_model=p,
        baseline_model=b,
    )
    assert rep.anomaly_detected is True
    # Only two anomalies → do not fire.
    rep2 = eng.score(
        tokens=["a", "b"],
        utterance="topic",
        clusters=clusters,
        persona_model=p,
        baseline_model=b,
    )
    assert rep2.anomaly_detected is False


def test_drift_engine_emits_strict_schema_keys():
    clusters = _StubClusters(idx={}, lifts={})
    p = deterministic_logprob({"a": 1.0})
    b = deterministic_logprob({"a": 1.0})
    eng = _make_engine()
    rep = eng.score(
        tokens=["a"],
        utterance="x",
        clusters=clusters,
        persona_model=p,
        baseline_model=b,
    )
    schema = rep.to_schema()
    assert set(schema.keys()) == {
        "anomaly_detected",
        "off_topic_tokens",
        "cluster_id",
        "lift_score",
        "severity",
    }


# ---------------------------------------------------------------------------
# Ledger entry adapter
# ---------------------------------------------------------------------------


def _trigger_report() -> DriftReport:
    clusters = _StubClusters(idx={"goblin": ["C0"]}, lifts={"C0": 5.0})
    eng = _make_engine(
        relevance_threshold=1.0,
        cluster_threshold=0.1,
        lift_threshold=0.0,
        k_threshold=0,
    )
    return eng.score(
        tokens=["goblin", "goblin"],
        utterance="topic",
        clusters=clusters,
        persona_model=deterministic_logprob({"goblin": 100.0}),
        baseline_model=deterministic_logprob({"the": 100.0}),
    )


def test_drift_to_trace_entry_clean_run_is_ok():
    rep = DriftReport(
        anomaly_detected=False,
        off_topic_tokens=(),
        cluster_id=(),
        lift_score=0.0,
        severity="none",
    )
    entry = drift_to_trace_entry(rep, step=0, intent_raw="hi")
    assert entry.status == "OK"
    assert entry.phi_verdict is True
    assert entry.failure_mode is None


def test_drift_to_trace_entry_anomaly_marks_status_and_metadata():
    rep = _trigger_report()
    entry = drift_to_trace_entry(rep, step=3, intent_raw="topic")
    assert entry.status.startswith("DRIFT_")
    assert entry.phi_verdict is False
    assert "goblin" in entry.violated_constraints
    # Cluster ids ride in metadata (excluded from canonical hashing).
    assert "cluster_id" in dict(entry.metadata)


def test_drift_engine_score_and_log_appends_to_ledger():
    ledger = MerkleLedger()
    clusters = _StubClusters(idx={"goblin": ["C0"]}, lifts={"C0": 5.0})
    eng = DriftEngine(
        config=DriftConfig(
            relevance_threshold=1.0,
            cluster_threshold=0.1,
            lift_threshold=0.0,
            k_threshold=0,
        ),
        ledger=ledger,
    )
    p = deterministic_logprob({"goblin": 100.0})
    b = deterministic_logprob({"the": 100.0})
    report, chain_hash = eng.score_and_log(
        step=0,
        tokens=["goblin"],
        utterance="topic",
        clusters=clusters,
        persona_model=p,
        baseline_model=b,
    )
    assert report.anomaly_detected is True
    assert chain_hash is not None
    assert ledger.height() == 1
    assert ledger.verify() is True


def test_drift_engine_without_ledger_returns_none_chain():
    eng = DriftEngine()
    clusters = _StubClusters(idx={}, lifts={})
    p = deterministic_logprob({"a": 1.0})
    b = deterministic_logprob({"a": 1.0})
    _, chain_hash = eng.score_and_log(
        step=0,
        tokens=["a"],
        utterance="x",
        clusters=clusters,
        persona_model=p,
        baseline_model=b,
    )
    assert chain_hash is None


def test_drift_engine_deterministic():
    """Two engines with identical inputs produce identical reports."""
    clusters = _StubClusters(idx={"x": ["C0"]}, lifts={"C0": 1.0})
    p = deterministic_logprob({"x": 5.0})
    b = deterministic_logprob({"x": 1.0})
    cfg = DriftConfig(
        relevance_threshold=1.0,
        cluster_threshold=0.1,
        lift_threshold=0.0,
        k_threshold=0,
    )
    r1 = DriftEngine(config=cfg).score(
        tokens=["x", "x"],
        utterance="u",
        clusters=clusters,
        persona_model=p,
        baseline_model=b,
    )
    r2 = DriftEngine(config=cfg).score(
        tokens=["x", "x"],
        utterance="u",
        clusters=clusters,
        persona_model=p,
        baseline_model=b,
    )
    assert r1.to_schema() == r2.to_schema()
