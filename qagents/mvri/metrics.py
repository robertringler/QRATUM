"""Pure observational metrics for the MVRI.

These functions are *observational only* — they must not influence loop
control flow. They are used by trace entries and by the post-hoc analysis
report.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

from qagents.mvri.state import H, State

# --------------------------------------------------------------------------- #
# Information-theoretic deltas
# --------------------------------------------------------------------------- #


def _activation_distribution(s: State) -> list[float]:
    weights = [abs(v) for _, v in s.activations]
    total = sum(weights)
    if total <= 0.0:
        # uniform distribution as default
        n = max(1, len(weights))
        return [1.0 / n] * n
    return [w / total for w in weights]


def delta_entropy(s_prev: State, s_new: State) -> float:
    """ΔH = H(s_new) - H(s_prev)."""

    return H(s_new) - H(s_prev)


def _node_aligned_distributions(s_prev: State, s_new: State) -> tuple[list[float], list[float]]:
    """Return (p, q) over the union of node ids, normalized."""

    nodes = sorted(set(s_prev.nodes) | set(s_new.nodes))
    p_raw = [abs(dict(s_prev.activations).get(n, 0.0)) for n in nodes]
    q_raw = [abs(dict(s_new.activations).get(n, 0.0)) for n in nodes]
    sp = sum(p_raw) or 1.0
    sq = sum(q_raw) or 1.0
    return [x / sp for x in p_raw], [x / sq for x in q_raw]


def delta_mutual_information(s_prev: State, s_new: State) -> float:
    """A symmetric ΔMI proxy.

    We model ΔMI as the symmetrised KL divergence between the prev/new
    activation distributions (Jensen–Shannon divergence). Strictly
    non-negative; zero when the distributions are identical. This is a
    well-defined information-theoretic quantity, *not* a placeholder.
    """

    p, q = _node_aligned_distributions(s_prev, s_new)
    m = [0.5 * (pi + qi) for pi, qi in zip(p, q)]

    def _kl(a: list[float], b: list[float]) -> float:
        out = 0.0
        for ai, bi in zip(a, b):
            if ai <= 0.0:
                continue
            if bi <= 0.0:
                # JS divergence is bounded; m is a mixture so bi=0 only
                # when both ai and qi are zero, which we already skipped.
                continue
            out += ai * math.log2(ai / bi)
        return out

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def delta_transfer_entropy(s_prev: State, s_new: State) -> float:
    """A directional, edge-weight-driven ΔTE proxy.

    For each edge ``(i, j)`` present in *both* states we compare
    ``|w_new| - |w_prev|``. The total is the L1 increase in directed
    coupling magnitude (``≥ 0`` ⇒ stronger directed flow,
    ``< 0`` ⇒ relaxation). This respects the "directional" semantics of
    transfer entropy at the level of graph couplings.
    """

    prev_w = dict(s_prev.edge_weights)
    new_w = dict(s_new.edge_weights)
    total = 0.0
    for e, w in new_w.items():
        if e in prev_w:
            total += abs(w) - abs(prev_w[e])
    return total


# --------------------------------------------------------------------------- #
# Trace summaries
# --------------------------------------------------------------------------- #


def acceptance_rate(trace: Iterable) -> float:
    """Fraction of trace entries with outcome ``accepted``."""

    n = 0
    accepted = 0
    for entry in trace:
        n += 1
        if getattr(entry, "outcome", None) == "accepted":
            accepted += 1
    if n == 0:
        return 0.0
    return accepted / n


def violation_count(trace: Iterable) -> dict[str, int]:
    """Count rejected steps grouped by constraint id.

    The special id ``"INVALID_ACTION"`` is used for actions rejected at
    the validation layer. ``"INVARIANT_HALT"`` denotes Type-IV halts.
    """

    counter: Counter[str] = Counter()
    for entry in trace:
        outcome = getattr(entry, "outcome", None)
        if outcome == "accepted":
            continue
        ids = list(getattr(entry, "constraints_failed", []) or [])
        if outcome == "invalid_action" and not ids:
            ids = ["INVALID_ACTION"]
        elif outcome == "invariant_halt" and not ids:
            ids = ["INVARIANT_HALT"]
        for cid in ids:
            counter[cid] += 1
    return dict(counter)


__all__ = [
    "acceptance_rate",
    "delta_entropy",
    "delta_mutual_information",
    "delta_transfer_entropy",
    "violation_count",
]
