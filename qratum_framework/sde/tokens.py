"""tokens.py — fundamental units of stream ingestion for the SDE.

Defines :class:`Token`, the per-tick observation produced by the
domain executor, and :class:`WindowSnapshot`, the frozen, hashable view
of a rolling window of tokens.

Determinism contract: ``WindowSnapshot.canonical_hash`` is the
SHA-256 of canonical-JSON of ``[token.value, ...]``.  Timestamps and
embeddings are intentionally excluded from the hash so that two runs
with identical token-value sequences produce byte-identical event
logs, in keeping with the rest of the operational spine
(``qratum_framework.trace``).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Token:
    """One stream observation.

    ``value`` is opaque to the SDE — it may be a token id, a string, a
    structured event, or anything JSON-serialisable through
    ``default=str``.  ``embedding`` is the optional per-token vector
    used by embedding-aware scorers (MMD).  ``timestamp`` is wall-clock
    metadata; it is *never* mixed into hashes.
    """

    value: Any
    timestamp: float = 0.0
    embedding: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        # Defensive: numpy arrays are mutable; freeze a copy so that
        # downstream mutation cannot perturb a token already in flight.
        if self.embedding is not None:
            arr = np.asarray(self.embedding, dtype=float)
            arr.setflags(write=False)
            object.__setattr__(self, "embedding", arr)


@dataclass(frozen=True)
class WindowSnapshot:
    """Frozen, hashable view of a rolling window of tokens.

    ``tokens`` is an immutable tuple — the buffer hands out a snapshot
    once per push and does not retain a writeable reference.  Scorers
    may inspect it but must not mutate it.
    """

    tokens: Tuple[Token, ...]
    tick: int

    # --- inspection helpers --------------------------------------------

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def values(self) -> Tuple[Any, ...]:
        return tuple(t.value for t in self.tokens)

    @property
    def embeddings(self) -> Optional[np.ndarray]:
        """Stack the per-token embeddings into a (n, d) array, or
        return ``None`` if any token lacks an embedding."""
        if not self.tokens:
            return None
        if any(t.embedding is None for t in self.tokens):
            return None
        return np.stack([np.asarray(t.embedding, dtype=float) for t in self.tokens])

    # --- canonical hash -------------------------------------------------

    def canonical_hash(self) -> str:
        """SHA-256 of canonical JSON of token values.

        Sorted keys, compact separators, ``default=str`` for non-JSON
        values.  Excludes timestamps, embeddings, and tick index so
        that identical token-value sequences hash identically across
        runs.
        """
        payload = json.dumps(
            list(self.values),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
            ensure_ascii=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = ["Token", "WindowSnapshot"]
