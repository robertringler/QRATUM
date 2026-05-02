"""signals.py — the three measurable drift signals.

Per the spec:

1. ``relevance(t, U) = cosine(embed(t), embed(U))``
2. ``log_lift(t)    = log P(t | persona) − log P(t | baseline)``
3. ``cluster_membership_score(t, clusters)``
            = Σ P(t in cluster_k) · co_occurrence_weight(k)

All three are pure functions and accept *injectable* scorers
(``RelevanceScorer``, ``LogprobScorer``) so callers can swap a real
embedding model or a real LLM logprob backend in without touching the
drift engine.

For offline tests and CI we ship deterministic stubs:

* :func:`deterministic_embedding` — hash-based 64-d unit vector;
  produces stable cosine similarities without any external model.
* :func:`deterministic_logprob` — token-frequency surrogate; the
  ``persona`` and ``baseline`` token frequencies plug straight in.

This mirrors the convention codified for ``qagents/llm_backends.py``:
LLM backends fall back deterministically when the SDK or API key is
missing rather than hard-failing.
"""
from __future__ import annotations

import hashlib
import math
from typing import Mapping, Protocol, Sequence


#: Scorer protocol: ``embed(token | utterance) -> sequence of floats``.
class RelevanceScorer(Protocol):  # pragma: no cover - structural typing
    def __call__(self, text: str) -> Sequence[float]:
        ...


#: Scorer protocol: ``logprob(token, *, condition: str) -> float``.
class LogprobScorer(Protocol):  # pragma: no cover - structural typing
    def __call__(self, token: str, *, condition: str) -> float:
        ...


# ---------------------------------------------------------------------------
# 1. Relevance
# ---------------------------------------------------------------------------


_EMBED_DIM = 64


def deterministic_embedding(text: str, *, dim: int = _EMBED_DIM) -> Sequence[float]:
    """Hash-based pseudo-embedding for offline / CI use.

    The embedding is a unit-L2 vector of dimension ``dim`` derived from
    SHA-256 of the input.  It is *not* semantically meaningful — two
    similar strings will not have similar vectors — so production use
    should swap in a real embedding model behind the
    :class:`RelevanceScorer` protocol.  But for unit tests and
    deterministic CI it is stable, fast, and zero-dependency.
    """
    if dim <= 0:
        raise ValueError("dim must be positive")
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    # Stretch the digest to fill ``dim`` floats by re-hashing rounds.
    raw = bytearray()
    rounds = (dim * 4 + len(digest) - 1) // len(digest)
    cur = digest
    for _ in range(rounds):
        raw.extend(cur)
        cur = hashlib.sha256(cur).digest()
    vec = []
    for i in range(dim):
        # Map 4 bytes → unsigned 32-bit int → float in roughly [-1, 1].
        # ``(x / 2**32)`` rescales the unsigned int into the unit
        # interval [0, 1); ``* 2.0 - 1.0`` then linearly maps that
        # onto [-1, 1).  After the L2-normalise below, the result is
        # a deterministic point on the unit sphere.
        chunk = raw[i * 4 : i * 4 + 4]
        x = int.from_bytes(chunk, "big", signed=False)
        vec.append((x / 2**32) * 2.0 - 1.0)
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    num = 0.0
    da = 0.0
    db = 0.0
    for i in range(n):
        num += a[i] * b[i]
        da += a[i] * a[i]
        db += b[i] * b[i]
    if da == 0.0 or db == 0.0:
        return 0.0
    return num / math.sqrt(da * db)


def relevance(
    token: str,
    utterance: str,
    *,
    embedder: RelevanceScorer | None = None,
) -> float:
    """``sim(t, U) = cosine(embed(t), embed(U))``.

    Defaults to :func:`deterministic_embedding` when ``embedder`` is
    not supplied.  The returned value is in ``[-1, 1]``.
    """
    emb = embedder or deterministic_embedding
    return float(_cosine(emb(token), emb(utterance)))


# ---------------------------------------------------------------------------
# 2. Log-lift vs baseline
# ---------------------------------------------------------------------------


def deterministic_logprob(
    persona_freqs: Mapping[str, float] | None = None,
    *,
    smoothing: float = 1.0,
) -> LogprobScorer:
    """Build a deterministic ``LogprobScorer`` from token-frequency tables.

    The returned scorer accepts ``condition="persona"`` or
    ``condition="baseline"`` — when the frequency table for that
    condition is missing the scorer returns the smoothed log-prior so
    the downstream ``log_lift`` is bounded.

    Production callers should ignore this helper and pass a real
    ``LogprobScorer`` that wraps an LLM's ``logprobs`` API.
    """
    persona = dict(persona_freqs or {})
    persona_total = sum(persona.values())

    def _scorer(token: str, *, condition: str) -> float:
        # The deterministic surrogate ignores ``condition`` *content*
        # and instead consults whatever table the caller bound at
        # construction time.  In practice the tests bind a
        # persona-table and use a separate baseline scorer.
        c = persona.get(token, 0.0)
        num = c + smoothing
        den = persona_total + smoothing * max(1, len(persona) or 1)
        return math.log(num / den)

    return _scorer


def log_lift(
    token: str,
    model: LogprobScorer,
    baseline_model: LogprobScorer,
    *,
    persona: str = "persona",
    neutral: str = "baseline",
) -> float:
    """``log P(t | persona) − log P(t | baseline)``.

    Defensive: any callable that raises is treated as ``-inf`` for
    that branch (i.e. an unknown token under that condition) which
    yields a bounded `inf` lift the engine can clip.
    """
    try:
        lp_persona = float(model(token, condition=persona))
    except Exception:  # pragma: no cover — defensive
        lp_persona = -math.inf
    try:
        lp_baseline = float(baseline_model(token, condition=neutral))
    except Exception:  # pragma: no cover — defensive
        lp_baseline = -math.inf
    if math.isinf(lp_persona) and math.isinf(lp_baseline):
        return 0.0
    if math.isinf(lp_baseline):
        return float("inf") if lp_persona > -math.inf else 0.0
    return float(lp_persona - lp_baseline)


# ---------------------------------------------------------------------------
# 3. Cluster co-occurrence
# ---------------------------------------------------------------------------


def cluster_membership_score(
    token: str,
    clusters_state: object,
) -> float:
    """``Σ P(t in cluster_k) · co_occurrence_weight(k)``.

    Concretely: for each cluster a token belongs to, accumulate that
    cluster's lift score (the spec's ``co_occurrence_weight``).  A
    token that is not in any discovered cluster scores 0.  A token in
    a single cluster scores that cluster's lift.

    ``clusters_state`` is a duck-typed object exposing the methods
    :meth:`membership_index` and :meth:`cluster_lift` — the
    :class:`ClusterEngineState` produced by Layer B is the canonical
    implementation, but a stub object that mocks just those two
    methods is also accepted (this is what the unit tests use to
    keep the drift engine independent of the cluster pipeline).
    """
    membership = clusters_state.membership_index()
    cluster_ids = membership.get(token, ())
    total = 0.0
    for cid in cluster_ids:
        total += float(clusters_state.cluster_lift(cid))
    return total


__all__ = [
    "LogprobScorer",
    "RelevanceScorer",
    "cluster_membership_score",
    "deterministic_embedding",
    "deterministic_logprob",
    "log_lift",
    "relevance",
]
