"""Tests for Layer B — Cluster Discovery Engine."""
from __future__ import annotations

import json

import pytest

from qratum_framework.observability.clusters import (
    ClusterEngineState,
    CoOccurrenceAccumulator,
    detect_communities,
    discover_clusters,
    edge_lift,
    jaccard_stability,
    score_clusters,
    spectral_communities,
)
from qratum_framework.observability.clusters.lift import (
    LiftConfig,
    lift_weighted_graph,
    positive_lift_subgraph,
)

# ---------------------------------------------------------------------------
# Co-occurrence accumulator
# ---------------------------------------------------------------------------


def test_cooccurrence_basic_counts():
    acc = CoOccurrenceAccumulator(window=4)
    acc.add_window(["a", "b", "c"], stream="persona")
    e = acc.edge("a", "b")
    assert e["persona"] == 1.0
    assert e["baseline"] == 0.0
    # canonical ordering (a,c) regardless of query order
    assert acc.edge("c", "a") == acc.edge("a", "c")


def test_cooccurrence_persona_vs_baseline_separate():
    acc = CoOccurrenceAccumulator(window=8)
    acc.add_window(["x", "y"], stream="persona")
    acc.add_window(["x", "y"], stream="baseline")
    e = acc.edge("x", "y")
    assert e["persona"] == 1.0
    assert e["baseline"] == 1.0


def test_cooccurrence_invalid_stream_raises():
    acc = CoOccurrenceAccumulator()
    with pytest.raises(ValueError):
        acc.add_window(["a", "b"], stream="other")
    with pytest.raises(ValueError):
        acc.add_tokens(["a"], stream="bogus")


def test_cooccurrence_distance_decay_monotone():
    acc = CoOccurrenceAccumulator(window=8, distance_decay=True)
    acc.add_window(["a", "b", "c", "d"], stream="persona")
    # (a,b) at distance 1 => weight 1/2; (a,d) at distance 3 => weight 1/4
    assert acc.edge("a", "b")["persona"] > acc.edge("a", "d")["persona"]


def test_cooccurrence_round_trip():
    acc = CoOccurrenceAccumulator(window=4)
    acc.add_tokens(["a", "b", "c", "d", "e"], stream="persona")
    acc.add_tokens(["a", "b"], stream="baseline")
    payload = acc.to_dict()
    rebuilt = CoOccurrenceAccumulator.from_dict(payload)
    assert dict(rebuilt.to_dict()) == dict(payload)


def test_cooccurrence_max_vocab_bound():
    acc = CoOccurrenceAccumulator(window=4, max_vocab=3)
    acc.add_window(["a", "b", "c", "d", "e"], stream="persona")
    assert len(acc.vocab) == 3
    # Surviving tokens must still have non-zero edges among themselves.
    assert acc.edge("a", "b")["persona"] >= 1.0


# ---------------------------------------------------------------------------
# Lift weighting
# ---------------------------------------------------------------------------


def test_edge_lift_smoothing_no_division_by_zero():
    val = edge_lift(0, 0, persona_total=0, baseline_total=0)
    # Symmetric inputs => zero lift even after smoothing
    assert val == 0.0


def test_edge_lift_clipping():
    cfg = LiftConfig(clip=0.1)
    v = edge_lift(100, 0, persona_total=100, baseline_total=0, config=cfg)
    assert -0.1 <= v <= 0.1


def test_lift_weighted_graph_filters_low_support():
    acc = CoOccurrenceAccumulator(window=4)
    # Only one window of (a,b) — combined support = 1 < default min_support=2
    acc.add_window(["a", "b"], stream="persona")
    out = lift_weighted_graph(acc)
    assert ("a", "b") not in out

    cfg = LiftConfig(min_support=1.0)
    out = lift_weighted_graph(acc, config=cfg)
    assert ("a", "b") in out


def test_positive_lift_subgraph_drops_negative():
    weighted = {("a", "b"): 0.5, ("b", "c"): -0.2}
    pos = positive_lift_subgraph(weighted)
    assert pos == {("a", "b"): 0.5}


# ---------------------------------------------------------------------------
# Community detection
# ---------------------------------------------------------------------------


def test_greedy_modularity_finds_two_components():
    # Two cliques connected by nothing => two communities.
    edges = {
        ("a", "b"): 1.0,
        ("b", "c"): 1.0,
        ("a", "c"): 1.0,
        ("x", "y"): 1.0,
        ("y", "z"): 1.0,
        ("x", "z"): 1.0,
    }
    comms = detect_communities(edges, backend="greedy")
    assert len(comms) == 2
    assert {"a", "b", "c"} in [set(c) for c in comms]
    assert {"x", "y", "z"} in [set(c) for c in comms]


def test_detect_communities_deterministic():
    edges = {
        ("a", "b"): 1.0,
        ("b", "c"): 1.0,
        ("a", "c"): 0.5,
        ("d", "e"): 1.0,
        ("e", "f"): 1.0,
    }
    a = detect_communities(edges, backend="greedy")
    b = detect_communities(edges, backend="greedy")
    assert a == b


def test_detect_communities_skips_negative_weights():
    edges = {("a", "b"): -1.0, ("c", "d"): 1.0}
    comms = detect_communities(edges, backend="greedy")
    # Only positive edge survives — single community.
    assert comms == [frozenset({"c", "d"})]


def test_spectral_communities_basic():
    edges = {
        ("a", "b"): 1.0,
        ("a", "c"): 1.0,
        ("b", "c"): 1.0,
        ("d", "e"): 1.0,
        ("e", "f"): 1.0,
        ("d", "f"): 1.0,
    }
    comms = spectral_communities(edges, k=2)
    sets = [set(c) for c in comms]
    assert {"a", "b", "c"} in sets
    assert {"d", "e", "f"} in sets


def test_detect_communities_unknown_backend_raises():
    with pytest.raises(ValueError):
        detect_communities({("a", "b"): 1.0}, backend="bogus")


# ---------------------------------------------------------------------------
# Scoring & stability
# ---------------------------------------------------------------------------


def test_jaccard_stability_perfect_overlap():
    cur = ["a", "b", "c"]
    prev = [frozenset({"a", "b", "c"})]
    assert jaccard_stability(cur, prev) == 1.0


def test_jaccard_stability_empty_previous_returns_zero():
    assert jaccard_stability(["a"], []) == 0.0


def test_score_clusters_assigns_ids_in_score_order():
    edges = {
        ("a", "b"): 1.0,
        ("b", "c"): 1.0,
        ("a", "c"): 1.0,
        ("d", "e"): 0.1,
        ("e", "f"): 0.1,
        ("d", "f"): 0.1,
    }
    comms = [frozenset({"a", "b", "c"}), frozenset({"d", "e", "f"})]
    out = score_clusters(comms, edges)
    assert out[0].id == "C0"
    assert out[1].id == "C1"
    assert out[0].score >= out[1].score


# ---------------------------------------------------------------------------
# End-to-end discover_clusters
# ---------------------------------------------------------------------------


def test_discover_clusters_recovers_persona_amplified_pair():
    # Persona stream injects {goblin, troll} co-occurring; baseline does not.
    persona = (["goblin", "troll", "filler", "filler"]) * 6
    baseline = (["filler", "the", "and", "of"]) * 6
    state = discover_clusters(
        persona_tokens=persona,
        baseline_tokens=baseline,
        window=4,
    )
    assert isinstance(state, ClusterEngineState)
    schema = state.to_schema()
    assert "clusters" in schema
    flat_tokens = {tok for c in schema["clusters"] for tok in c["tokens"]}
    assert "goblin" in flat_tokens or "troll" in flat_tokens


def test_discover_clusters_state_round_trip():
    persona = ["x", "y", "z"] * 5
    baseline = ["a", "b", "c"] * 5
    state = discover_clusters(persona_tokens=persona, baseline_tokens=baseline, window=3)
    payload = state.to_dict()
    # Must be JSON-serialisable.
    json.dumps(payload, default=str)
    rebuilt = ClusterEngineState.from_dict(payload)
    assert rebuilt.window_index == state.window_index
    assert [c.id for c in rebuilt.clusters] == [c.id for c in state.clusters]


def test_discover_clusters_membership_index_lookup():
    persona = (["alpha", "beta"]) * 6
    baseline = (["the", "and"]) * 6
    state = discover_clusters(
        persona_tokens=persona, baseline_tokens=baseline, window=2
    )
    idx = state.membership_index()
    # Token in some cluster has at least one cluster id.
    assert all(isinstance(v, list) for v in idx.values())
    if state.clusters:
        any_token = state.clusters[0].tokens[0]
        assert any_token in idx


def test_discover_clusters_deterministic():
    persona = ["a", "b", "c"] * 8
    baseline = ["x", "y", "z"] * 8
    s1 = discover_clusters(persona_tokens=persona, baseline_tokens=baseline, window=3)
    s2 = discover_clusters(persona_tokens=persona, baseline_tokens=baseline, window=3)
    assert s1.to_schema() == s2.to_schema()
