"""cooccurrence.py — sliding-window co-occurrence accumulator.

Implements Step 1 of Layer B: a sparse co-occurrence graph

    G[token_i][token_j] += co_occurrence_window_weight

over a sliding window (default 256 tokens, per the spec).  Counts are
*persona-tagged*: each ``add_window`` call records both a persona-stream
contribution and a baseline-stream contribution so that downstream
:mod:`qratum_framework.observability.clusters.lift` can compute

    P(token_i, token_j | persona) − P(token_i, token_j | baseline)

without having to keep two separate accumulators in lockstep.

Determinism: the accumulator only iterates over tokens in the order
supplied by the caller, never touches a global RNG, and returns sorted
edges from :meth:`edges` so two equivalent runs produce byte-identical
outputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Mapping, Sequence, Tuple

#: Sliding-window default per the spec (Layer B / Step 1).
DEFAULT_WINDOW: int = 256

# Each edge maps an ordered pair (i, j) with i < j to per-stream counts.
_EdgeKey = Tuple[str, str]
_EdgeCounts = Dict[str, float]  # {"persona": n, "baseline": n}


def _ordered(a: str, b: str) -> _EdgeKey:
    """Return a canonical ordering of an undirected edge."""
    return (a, b) if a <= b else (b, a)


@dataclass
class CoOccurrenceAccumulator:
    """Sparse, persona-tagged co-occurrence accumulator.

    Two streams (``persona`` and ``baseline``) are accumulated in the
    same edge dictionary so the lift module can read both sides without
    iterating over two graphs.

    The default decay weight is uniform (each token pair within the
    window contributes ``1.0``); if the caller wants distance-weighted
    counts (a common refinement) they pass ``distance_decay=True``,
    which uses ``1 / (1 + |i − j|)`` — bounded in (0, 1] and
    deterministic.

    Token vocabulary is bounded by ``max_vocab`` to keep the sparse
    representation tractable on long streams; tokens beyond that bound
    are silently dropped from new edges (existing edges for the
    in-vocab tokens are still updated).
    """

    window: int = DEFAULT_WINDOW
    distance_decay: bool = False
    max_vocab: int = 50_000

    _edges: Dict[_EdgeKey, _EdgeCounts] = field(default_factory=dict)
    _node_counts: Dict[str, Dict[str, float]] = field(default_factory=dict)
    _vocab: Dict[str, int] = field(default_factory=dict)
    _windows_seen: int = 0

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_tokens(self, tokens: Sequence[str], *, stream: str = "persona") -> None:
        """Add a token stream to the accumulator under ``stream``.

        ``stream`` must be either ``"persona"`` or ``"baseline"``.  The
        stream is split into non-overlapping windows of ``self.window``
        tokens; within each window, every token pair contributes a
        weight to the edge.  Use :meth:`add_window` if the caller has
        already chunked the stream.
        """
        if stream not in ("persona", "baseline"):
            raise ValueError(f"stream must be 'persona' or 'baseline', got {stream!r}")
        if not tokens:
            return
        w = self.window
        for start in range(0, len(tokens), w):
            self.add_window(tokens[start : start + w], stream=stream)

    def add_window(self, window_tokens: Sequence[str], *, stream: str = "persona") -> None:
        """Add a single window's worth of tokens.

        The number of unique edges in a window is ``O(n²)`` where ``n``
        is the window length, so callers should keep windows reasonably
        sized (the default ``256`` keeps a single window under ~32k
        edges in the worst case).
        """
        if stream not in ("persona", "baseline"):
            raise ValueError(f"stream must be 'persona' or 'baseline', got {stream!r}")
        if not window_tokens:
            return

        # Update node counts (per-stream). Vocab admission is checked
        # lazily; an OOV token is admitted iff there is room left, but
        # an already-admitted token is always counted to preserve
        # statistical fidelity for the surviving subset.
        for tok in window_tokens:
            if tok not in self._vocab:
                if len(self._vocab) >= self.max_vocab:
                    continue
                self._vocab[tok] = len(self._vocab)
            counts = self._node_counts.setdefault(tok, {"persona": 0.0, "baseline": 0.0})
            counts[stream] += 1.0

        # Update edge counts.
        n = len(window_tokens)
        for i in range(n):
            ti = window_tokens[i]
            if ti not in self._vocab:
                continue
            for j in range(i + 1, n):
                tj = window_tokens[j]
                if tj == ti or tj not in self._vocab:
                    continue
                weight = 1.0 / (1.0 + (j - i)) if self.distance_decay else 1.0
                key = _ordered(ti, tj)
                ecounts = self._edges.setdefault(key, {"persona": 0.0, "baseline": 0.0})
                ecounts[stream] += weight

        self._windows_seen += 1

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def vocab(self) -> List[str]:
        """Return the vocabulary in deterministic insertion order."""
        return sorted(self._vocab, key=self._vocab.__getitem__)

    @property
    def windows_seen(self) -> int:
        return self._windows_seen

    def edge(self, a: str, b: str) -> _EdgeCounts:
        """Return the per-stream counts for the (a, b) edge."""
        return dict(self._edges.get(_ordered(a, b), {"persona": 0.0, "baseline": 0.0}))

    def edges(self) -> Iterator[Tuple[_EdgeKey, _EdgeCounts]]:
        """Yield ``((i, j), counts)`` in deterministic sorted order."""
        for key in sorted(self._edges):
            yield key, dict(self._edges[key])

    def node_count(self, token: str, *, stream: str = "persona") -> float:
        """Return the per-stream occurrence count of ``token``."""
        return self._node_counts.get(token, {}).get(stream, 0.0)

    def total(self, *, stream: str = "persona") -> float:
        """Return the total per-stream edge weight (used as a denominator).

        This is used by the lift module to convert raw counts into
        probabilities.  Returns ``0.0`` for an empty accumulator.
        """
        return float(sum(c[stream] for c in self._edges.values()))

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Mapping[str, object]:
        """Return a deterministic dict view suitable for JSON export."""
        return {
            "window": self.window,
            "distance_decay": self.distance_decay,
            "max_vocab": self.max_vocab,
            "windows_seen": self._windows_seen,
            "vocab": self.vocab,
            "node_counts": {
                tok: dict(self._node_counts[tok]) for tok in sorted(self._node_counts)
            },
            "edges": [
                {"i": k[0], "j": k[1], "persona": v["persona"], "baseline": v["baseline"]}
                for k, v in self.edges()
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> CoOccurrenceAccumulator:
        """Round-trip an accumulator from the dict produced by :meth:`to_dict`."""
        acc = cls(
            window=int(payload.get("window", DEFAULT_WINDOW)),
            distance_decay=bool(payload.get("distance_decay", False)),
            max_vocab=int(payload.get("max_vocab", 50_000)),
        )
        acc._windows_seen = int(payload.get("windows_seen", 0))
        for idx, tok in enumerate(payload.get("vocab", []) or []):
            acc._vocab[str(tok)] = idx
        for tok, counts in (payload.get("node_counts", {}) or {}).items():
            acc._node_counts[str(tok)] = {
                "persona": float(counts.get("persona", 0.0)),
                "baseline": float(counts.get("baseline", 0.0)),
            }
        for edge in payload.get("edges", []) or []:
            key = _ordered(str(edge["i"]), str(edge["j"]))
            acc._edges[key] = {
                "persona": float(edge.get("persona", 0.0)),
                "baseline": float(edge.get("baseline", 0.0)),
            }
        return acc


__all__ = ["CoOccurrenceAccumulator", "DEFAULT_WINDOW"]
