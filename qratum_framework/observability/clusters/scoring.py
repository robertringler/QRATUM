"""scoring.py — cluster scoring and stability.

Layer B / Step 4:

    cluster_score = mean(lift(tokens)) × density × stability_over_time

where ``density`` is the within-cluster edge density of the lift-weighted
graph and ``stability`` is the Jaccard overlap of cluster membership
across two consecutive observation windows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Iterable, List, Mapping, Sequence, Tuple

from qratum_framework.observability.clusters.lift import EdgeKey, average_lift


@dataclass(frozen=True)
class Cluster:
    """A scored, persona-amplified semantic cluster.

    Mirrors the spec's output schema:

        {"id": "C12", "tokens": [...], "lift": 3.8, "stability": 0.91}

    plus an internal ``score`` that combines lift, density, and
    stability for downstream ranking.
    """

    id: str
    tokens: Tuple[str, ...]
    lift: float
    density: float
    stability: float
    score: float

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "tokens": list(self.tokens),
            "lift": float(self.lift),
            "density": float(self.density),
            "stability": float(self.stability),
            "score": float(self.score),
        }


def cluster_density(tokens: Iterable[str], weighted: Mapping[EdgeKey, float]) -> float:
    """Fraction of possible undirected edges between ``tokens`` present in ``weighted``.

    Returns ``0.0`` for clusters of size < 2.  This is a graph-theoretic
    density, not a weighted-density; it is bounded in ``[0, 1]`` and so
    composes cleanly with ``mean_lift × stability``.
    """
    members = list(dict.fromkeys(tokens))  # preserve order, dedupe
    n = len(members)
    if n < 2:
        return 0.0
    total_possible = n * (n - 1) / 2.0
    member_set = set(members)
    present = 0
    for (i, j) in weighted:
        if i in member_set and j in member_set:
            present += 1
    return present / total_possible if total_possible > 0 else 0.0


def jaccard_stability(
    current: Iterable[str], previous: Iterable[FrozenSet[str]]
) -> float:
    """Best Jaccard overlap of ``current`` against any cluster in ``previous``.

    Stability over time is the spec's third multiplicative term in the
    cluster score.  We define it operationally as: *given the cluster
    we just observed, how well does it match a cluster from the prior
    observation window?*  This rewards persistent, reproducible
    clusters and penalises one-off spurious ones.

    Returns ``0.0`` when ``previous`` is empty (i.e. first window has
    no temporal context, which is the conservative default).
    """
    cur = set(current)
    if not cur or not previous:
        return 0.0
    best = 0.0
    for prev in previous:
        union = cur | set(prev)
        if not union:
            continue
        j = len(cur & set(prev)) / len(union)
        if j > best:
            best = j
    return float(best)


def score_clusters(
    communities: Sequence[FrozenSet[str]],
    weighted: Mapping[EdgeKey, float],
    *,
    previous: Sequence[FrozenSet[str]] = (),
    id_prefix: str = "C",
) -> List[Cluster]:
    """Compute the ``Cluster`` records for a set of communities.

    Ordering: communities are first sorted by descending ``score`` so
    consumers (notably the drift engine) can short-circuit on the most
    persona-amplified clusters.  ``id`` is assigned in that order
    (``C0, C1, …``) so the same input always yields the same labels.
    """
    scored: List[Cluster] = []
    for comm in communities:
        tokens = tuple(sorted(comm))
        lift_val = average_lift(tokens, weighted)
        dens = cluster_density(tokens, weighted)
        stab = jaccard_stability(tokens, previous)
        # The spec multiplies stability in.  When previous is empty we
        # treat stability as 1.0 *for ranking purposes* so first-window
        # clusters are not all collapsed to score==0.  The reported
        # ``stability`` field still records the raw Jaccard so callers
        # can see we have no temporal context.
        eff_stab = stab if previous else 1.0
        score = float(lift_val * dens * eff_stab)
        scored.append(
            Cluster(
                id="",  # filled in after sorting
                tokens=tokens,
                lift=float(lift_val),
                density=float(dens),
                stability=float(stab),
                score=score,
            )
        )
    scored.sort(key=lambda c: (-c.score, c.tokens))
    return [
        Cluster(
            id=f"{id_prefix}{i}",
            tokens=c.tokens,
            lift=c.lift,
            density=c.density,
            stability=c.stability,
            score=c.score,
        )
        for i, c in enumerate(scored)
    ]


__all__ = ["Cluster", "cluster_density", "jaccard_stability", "score_clusters"]
