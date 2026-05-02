"""lift.py — persona-vs-baseline edge lift with Laplace smoothing.

Layer B / Step 2: weight every co-occurrence edge by the persona-baseline
delta probability,

    edge_weight(i, j) = P(i, j | persona) − P(i, j | baseline)

Probabilities are estimated as smoothed frequencies over the relevant
stream's total edge weight, with a configurable minimum-support floor
that drops noisy edges before community detection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple

from qratum_framework.observability.clusters.cooccurrence import CoOccurrenceAccumulator

#: Edge key (canonical ordered pair).
EdgeKey = Tuple[str, str]


@dataclass(frozen=True)
class LiftConfig:
    """Tunables for the lift weighting step.

    * ``smoothing`` — Laplace add-α numerator constant.  Avoids ``log 0``
      and stabilises lift on rare edges.  Default ``1.0``.
    * ``min_support`` — minimum *combined* persona+baseline raw count
      required before an edge contributes.  Edges below the floor are
      dropped from the weighted graph (returned mapping omits them).
    * ``clip`` — symmetric clip on the returned lift values.  ``None``
      disables clipping; ``2.0`` keeps lifts in ``[-2.0, +2.0]``.
    """

    smoothing: float = 1.0
    min_support: float = 2.0
    clip: float | None = None


def edge_lift(
    persona_count: float,
    baseline_count: float,
    *,
    persona_total: float,
    baseline_total: float,
    config: LiftConfig | None = None,
) -> float:
    """Compute the persona-vs-baseline lift for one edge.

    Returns ``P_persona − P_baseline`` after additive smoothing.  When
    either total is ``0`` the corresponding probability is treated as
    the smoothing prior, so the function never divides by zero.
    """
    cfg = config or LiftConfig()
    alpha = cfg.smoothing
    p_num = persona_count + alpha
    b_num = baseline_count + alpha
    # +alpha in numerator, +1 in denom is the standard Laplace form for
    # a single Bernoulli, but for an edge probability we keep the total
    # in the denominator and lift the prior to ``alpha`` per stream.  We
    # add ``alpha`` to both denominators to mirror the numerator.
    p_den = persona_total + alpha
    b_den = baseline_total + alpha
    p_p = p_num / p_den if p_den > 0 else 0.0
    p_b = b_num / b_den if b_den > 0 else 0.0
    lift = p_p - p_b
    if cfg.clip is not None:
        c = abs(cfg.clip)
        if lift > c:
            lift = c
        elif lift < -c:
            lift = -c
    return float(lift)


def lift_weighted_graph(
    accumulator: CoOccurrenceAccumulator,
    *,
    config: LiftConfig | None = None,
) -> Dict[EdgeKey, float]:
    """Return a dense mapping of edges to persona-vs-baseline lift.

    Edges whose combined persona+baseline raw count is below
    ``config.min_support`` are dropped from the result.  The mapping is
    suitable input for :mod:`qratum_framework.observability.clusters.community`.
    """
    cfg = config or LiftConfig()
    persona_total = accumulator.total(stream="persona")
    baseline_total = accumulator.total(stream="baseline")

    out: Dict[EdgeKey, float] = {}
    for key, counts in accumulator.edges():
        support = counts["persona"] + counts["baseline"]
        if support < cfg.min_support:
            continue
        out[key] = edge_lift(
            counts["persona"],
            counts["baseline"],
            persona_total=persona_total,
            baseline_total=baseline_total,
            config=cfg,
        )
    return out


def positive_lift_subgraph(
    weighted: Mapping[EdgeKey, float], *, threshold: float = 0.0
) -> Dict[EdgeKey, float]:
    """Return only edges whose lift exceeds ``threshold``.

    Community detection in :mod:`.community` operates on this positive
    subgraph: a persona-amplified semantic cluster is, by construction,
    a connected component of edges with persona >> baseline.
    """
    return {k: v for k, v in weighted.items() if v > threshold}


def average_lift(
    tokens: Iterable[str], weighted: Mapping[EdgeKey, float]
) -> float:
    """Mean per-edge lift over edges incident to ``tokens``.

    Used by :mod:`.scoring` to compute ``mean(lift(tokens))`` for a
    discovered cluster.  Returns ``0.0`` if the cluster has no internal
    edges in ``weighted``.
    """
    members = set(tokens)
    if len(members) < 2:
        return 0.0
    total = 0.0
    n = 0
    for (i, j), w in weighted.items():
        if i in members and j in members:
            total += w
            n += 1
    return total / n if n else 0.0


__all__ = [
    "EdgeKey",
    "LiftConfig",
    "average_lift",
    "edge_lift",
    "lift_weighted_graph",
    "positive_lift_subgraph",
]
