"""state.py — ClusterEngineState snapshot and the discover_clusters façade.

This module ties Steps 1–4 of Layer B together into a single
serialisable state object that:

* round-trips through JSON for offline persistence;
* can be appended to a :class:`qratum_framework.trace.MerkleLedger` for
  tamper-evident audit (recommended default — see PR plan, Q3);
* is the input the drift engine (Layer A) consumes via
  :class:`qratum_framework.observability.drift.engine.DriftEngine`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

from qratum_framework.observability.clusters.community import detect_communities
from qratum_framework.observability.clusters.cooccurrence import (
    DEFAULT_WINDOW,
    CoOccurrenceAccumulator,
)
from qratum_framework.observability.clusters.lift import (
    EdgeKey,
    LiftConfig,
    lift_weighted_graph,
    positive_lift_subgraph,
)
from qratum_framework.observability.clusters.scoring import Cluster, score_clusters

# ---------------------------------------------------------------------------
# Top-level façade
# ---------------------------------------------------------------------------


def discover_clusters(
    persona_tokens: Sequence[str],
    baseline_tokens: Sequence[str],
    *,
    window: int = DEFAULT_WINDOW,
    distance_decay: bool = False,
    lift_config: Optional[LiftConfig] = None,
    backend: str = "auto",
    seed: int = 0,
    previous_clusters: Sequence[FrozenSet[str]] = (),
    positive_threshold: float = 0.0,
) -> ClusterEngineState:
    """One-shot Layer-B pipeline: tokens → emergent clusters.

    ``persona_tokens`` is the persona-influenced stream, ``baseline_tokens``
    the neutral-persona stream.  The two streams should cover comparable
    prompts/contexts; mismatched corpora will yield meaningless lift.
    """
    acc = CoOccurrenceAccumulator(window=window, distance_decay=distance_decay)
    acc.add_tokens(persona_tokens, stream="persona")
    acc.add_tokens(baseline_tokens, stream="baseline")
    weighted = lift_weighted_graph(acc, config=lift_config)
    pos = positive_lift_subgraph(weighted, threshold=positive_threshold)
    communities = detect_communities(pos, seed=seed, backend=backend)
    clusters = score_clusters(communities, weighted, previous=previous_clusters)
    return ClusterEngineState(
        clusters=tuple(clusters),
        edge_weights=dict(weighted),
        accumulator=acc,
        window_index=acc.windows_seen,
        previous_membership=tuple(previous_clusters),
    )


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClusterEngineState:
    """Serialisable snapshot of the cluster engine.

    Carries everything downstream consumers need:

    * ``clusters`` — scored, id'd ``Cluster`` records.
    * ``edge_weights`` — the persona-vs-baseline lift-weighted graph
      (used by the drift engine to compute ``cluster_membership_score``
      without re-running the accumulator).
    * ``accumulator`` — the raw co-occurrence accumulator, so the next
      observation window can extend it incrementally.
    * ``window_index`` — monotonic count of windows seen.
    * ``previous_membership`` — the membership used as the *prior*
      window when stability was scored, retained so a downstream
      caller can chain windows without leaking state.
    """

    clusters: Tuple[Cluster, ...]
    edge_weights: Mapping[EdgeKey, float]
    accumulator: CoOccurrenceAccumulator
    window_index: int
    previous_membership: Tuple[FrozenSet[str], ...] = ()

    # ------------------------------------------------------------------
    # Public-facing schema (matches the spec exactly)
    # ------------------------------------------------------------------

    def to_schema(self) -> Dict[str, Any]:
        """Return the spec-mandated ``{"clusters": [...]}`` JSON shape."""
        return {
            "clusters": [
                {
                    "id": c.id,
                    "tokens": list(c.tokens),
                    "lift": float(c.lift),
                    "stability": float(c.stability),
                }
                for c in self.clusters
            ]
        }

    # ------------------------------------------------------------------
    # Round-trip persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Full, lossless dict view (for JSON snapshotting)."""
        return {
            "schema_version": "1.0",
            "window_index": int(self.window_index),
            "clusters": [c.to_dict() for c in self.clusters],
            "edge_weights": [
                {"i": k[0], "j": k[1], "w": float(v)}
                for k, v in sorted(self.edge_weights.items())
            ],
            "accumulator": dict(self.accumulator.to_dict()),
            "previous_membership": [sorted(m) for m in self.previous_membership],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ClusterEngineState:
        clusters = tuple(
            Cluster(
                id=str(c["id"]),
                tokens=tuple(c["tokens"]),
                lift=float(c["lift"]),
                density=float(c.get("density", 0.0)),
                stability=float(c["stability"]),
                score=float(c.get("score", 0.0)),
            )
            for c in payload.get("clusters", [])
        )
        edges: Dict[EdgeKey, float] = {}
        for e in payload.get("edge_weights", []) or []:
            i, j = str(e["i"]), str(e["j"])
            key = (i, j) if i <= j else (j, i)
            edges[key] = float(e["w"])
        acc = CoOccurrenceAccumulator.from_dict(payload.get("accumulator", {}) or {})
        prev = tuple(
            frozenset(m) for m in (payload.get("previous_membership", []) or [])
        )
        return cls(
            clusters=clusters,
            edge_weights=edges,
            accumulator=acc,
            window_index=int(payload.get("window_index", 0)),
            previous_membership=prev,
        )

    # ------------------------------------------------------------------
    # Convenience views for the drift engine
    # ------------------------------------------------------------------

    def membership_index(self) -> Dict[str, List[str]]:
        """Return ``{token -> [cluster_id, …]}`` for fast lookup.

        A token that appears in multiple clusters (e.g. on a boundary)
        is listed under each.  Used by
        :func:`qratum_framework.observability.drift.signals.cluster_membership`.
        """
        out: Dict[str, List[str]] = {}
        for c in self.clusters:
            for tok in c.tokens:
                out.setdefault(tok, []).append(c.id)
        return out

    def cluster_lift(self, cluster_id: str) -> float:
        """Return the lift of ``cluster_id`` or ``0.0`` if absent."""
        for c in self.clusters:
            if c.id == cluster_id:
                return float(c.lift)
        return 0.0

    def membership(self) -> Tuple[FrozenSet[str], ...]:
        """Return cluster memberships as frozensets — input for the next window."""
        return tuple(frozenset(c.tokens) for c in self.clusters)


__all__ = ["ClusterEngineState", "discover_clusters"]
