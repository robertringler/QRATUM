"""scorers.py — drift scorers and the :class:`DriftReading` they produce.

Two concrete scorers are shipped:

* :class:`KLDriftScorer` — empirical-distribution KL divergence between
  the current window and a baseline window.  Numeric token values use
  ``np.histogram``; non-numeric (categorical) values use a count-based
  empirical distribution.  Both branches are epsilon-smoothed to avoid
  ``log 0``.
* :class:`MMDDriftScorer` — Maximum Mean Discrepancy with an RBF
  kernel.  Bandwidth ``σ`` is set by the median heuristic on the
  pairwise-distance matrix of the union of both windows.

Both scorers are stateful only in their delta tracking (``Δd_t`` and
``Δ²d_t``).  The score itself is a pure function of ``(window,
baseline)``.  Degenerate inputs — windows below ``min_window``, zero
variance, missing embeddings — return ``d_t = 0.0`` with the
``DEGENERATE`` flag set, never raise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Protocol, Tuple

import numpy as np

from qratum_framework.sde.tokens import WindowSnapshot

#: Marker flag attached to :class:`DriftReading` when the scorer could
#: not produce a meaningful score (under-populated window, zero
#: variance, missing embeddings, …).
DEGENERATE: str = "DEGENERATE"


@dataclass(frozen=True)
class DriftReading:
    """One scored reading.

    ``d_t`` is the raw drift score, ``delta_d`` is the first
    difference (``d_t - d_{t-1}``), and ``delta2_d`` is the second
    difference (``Δd_t - Δd_{t-1}``).  ``flags`` carries side-channel
    markers such as :data:`DEGENERATE`.
    """

    d_t: float
    delta_d: float
    delta2_d: float
    scorer_name: str
    tick: int
    flags: Tuple[str, ...] = ()

    def is_degenerate(self) -> bool:
        return DEGENERATE in self.flags

    def to_dict(self) -> Dict[str, Any]:
        return {
            "d_t": float(self.d_t),
            "delta_d": float(self.delta_d),
            "delta2_d": float(self.delta2_d),
            "scorer_name": self.scorer_name,
            "tick": int(self.tick),
            "flags": list(self.flags),
        }


class DriftScorer(Protocol):
    """Protocol for drift scorers.

    Implementations carry their own ``Δd_t`` / ``Δ²d_t`` state and
    must be reset-friendly via :meth:`reset`.
    """

    name: str

    def score(
        self, window: WindowSnapshot, baseline: WindowSnapshot
    ) -> DriftReading: ...

    def reset(self) -> None: ...


# ---------------------------------------------------------------------------
# Common base
# ---------------------------------------------------------------------------


class _BaseScorer:
    """Shared first/second-difference bookkeeping."""

    name: str = "base"

    def __init__(self) -> None:
        self._prev_d: Optional[float] = None
        self._prev_delta: Optional[float] = None

    def reset(self) -> None:
        self._prev_d = None
        self._prev_delta = None

    def _wrap(
        self, d_t: float, *, tick: int, flags: Iterable[str] = ()
    ) -> DriftReading:
        delta_d = 0.0 if self._prev_d is None else float(d_t - self._prev_d)
        delta2_d = (
            0.0 if self._prev_delta is None else float(delta_d - self._prev_delta)
        )
        # Sanitise non-finite scores defensively — they would corrupt
        # downstream tier comparisons.
        if not np.isfinite(d_t):
            d_t = 0.0
            flags = tuple(list(flags) + [DEGENERATE])
        reading = DriftReading(
            d_t=float(d_t),
            delta_d=float(delta_d),
            delta2_d=float(delta2_d),
            scorer_name=self.name,
            tick=int(tick),
            flags=tuple(flags),
        )
        self._prev_d = float(d_t)
        self._prev_delta = float(delta_d)
        return reading


# ---------------------------------------------------------------------------
# KL-divergence scorer
# ---------------------------------------------------------------------------


class KLDriftScorer(_BaseScorer):
    """Empirical KL divergence ``D_KL(W_t ‖ W₀)``.

    Numeric values are binned with :func:`numpy.histogram`; non-numeric
    values use a categorical count distribution over the union of
    keys.  Both distributions are Laplace-smoothed by ``epsilon`` so
    that no entry is exactly zero.
    """

    name = "kl"

    def __init__(self, *, n_bins: int = 32, epsilon: float = 1e-9) -> None:
        super().__init__()
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        self._n_bins = int(n_bins)
        self._epsilon = float(epsilon)

    # --- main scoring API ----------------------------------------------

    def score(
        self, window: WindowSnapshot, baseline: WindowSnapshot
    ) -> DriftReading:
        if len(window) == 0 or len(baseline) == 0:
            return self._wrap(0.0, tick=window.tick, flags=(DEGENERATE,))

        try:
            p, q = self._distributions(window.values, baseline.values)
        except _DegenerateInput:
            return self._wrap(0.0, tick=window.tick, flags=(DEGENERATE,))

        # KL divergence with Laplace smoothing.
        p = p + self._epsilon
        q = q + self._epsilon
        p = p / p.sum()
        q = q / q.sum()
        kl = float(np.sum(p * (np.log(p) - np.log(q))))
        # Numerical noise can produce very small negatives; clamp to 0.
        kl = max(kl, 0.0)
        return self._wrap(kl, tick=window.tick)

    # --- distribution construction -------------------------------------

    def _distributions(
        self, w_values: Tuple[Any, ...], b_values: Tuple[Any, ...]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return aligned (p, q) probability vectors.

        Tries numeric binning first; falls back to categorical counts
        if either side contains non-numeric values.
        """
        if self._all_numeric(w_values) and self._all_numeric(b_values):
            return self._numeric_distributions(w_values, b_values)
        return self._categorical_distributions(w_values, b_values)

    @staticmethod
    def _all_numeric(values: Tuple[Any, ...]) -> bool:
        for v in values:
            if isinstance(v, bool):
                return False  # bool is int subclass; reject explicitly
            if not isinstance(v, (int, float)):
                return False
        return True

    def _numeric_distributions(
        self, w_values: Tuple[Any, ...], b_values: Tuple[Any, ...]
    ) -> Tuple[np.ndarray, np.ndarray]:
        w = np.asarray(w_values, dtype=float)
        b = np.asarray(b_values, dtype=float)
        lo = float(min(w.min(), b.min()))
        hi = float(max(w.max(), b.max()))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            # Zero-variance corpus → distributions are identical
            # delta-spikes; KL is 0 by construction but signal is
            # uninformative.
            raise _DegenerateInput()
        edges = np.linspace(lo, hi, self._n_bins + 1)
        p, _ = np.histogram(w, bins=edges)
        q, _ = np.histogram(b, bins=edges)
        return p.astype(float), q.astype(float)

    def _categorical_distributions(
        self, w_values: Tuple[Any, ...], b_values: Tuple[Any, ...]
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Sorted union of keys (str-coerced) → deterministic alignment.
        keys = sorted({str(v) for v in w_values} | {str(v) for v in b_values})
        if not keys:
            raise _DegenerateInput()
        w_counts = {str(v): 0 for v in keys}
        b_counts = {str(v): 0 for v in keys}
        for v in w_values:
            w_counts[str(v)] += 1
        for v in b_values:
            b_counts[str(v)] += 1
        p = np.array([w_counts[k] for k in keys], dtype=float)
        q = np.array([b_counts[k] for k in keys], dtype=float)
        return p, q


# ---------------------------------------------------------------------------
# MMD scorer (RBF kernel, median bandwidth)
# ---------------------------------------------------------------------------


class MMDDriftScorer(_BaseScorer):
    """Biased empirical MMD² with an RBF kernel.

    Embeddings are read from :attr:`Token.embedding`.  If either window
    is missing any embedding the scorer returns 0.0 with the
    :data:`DEGENERATE` flag.  Bandwidth ``σ`` is the median of the
    pairwise Euclidean distances in the union ``W_t ∪ W₀`` (the
    canonical "median heuristic"); a zero median triggers the
    degenerate path.
    """

    name = "mmd"

    def __init__(self, *, epsilon: float = 1e-12) -> None:
        super().__init__()
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        self._epsilon = float(epsilon)

    def score(
        self, window: WindowSnapshot, baseline: WindowSnapshot
    ) -> DriftReading:
        if len(window) == 0 or len(baseline) == 0:
            return self._wrap(0.0, tick=window.tick, flags=(DEGENERATE,))
        x = window.embeddings
        y = baseline.embeddings
        if x is None or y is None:
            return self._wrap(0.0, tick=window.tick, flags=(DEGENERATE,))
        if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
            return self._wrap(0.0, tick=window.tick, flags=(DEGENERATE,))

        sigma = self._median_bandwidth(x, y)
        if sigma <= 0.0 or not np.isfinite(sigma):
            return self._wrap(0.0, tick=window.tick, flags=(DEGENERATE,))

        gamma = 1.0 / (2.0 * (sigma ** 2) + self._epsilon)
        k_xx = self._rbf(x, x, gamma)
        k_yy = self._rbf(y, y, gamma)
        k_xy = self._rbf(x, y, gamma)
        mmd2 = float(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())
        # Empirical MMD² can be slightly negative under finite-sample
        # noise; clamp to 0 so the score is monotone with separation.
        mmd2 = max(mmd2, 0.0)
        return self._wrap(mmd2, tick=window.tick)

    # --- helpers --------------------------------------------------------

    @staticmethod
    def _pairwise_sq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Squared Euclidean pairwise distances, shape (n, m)."""
        aa = np.sum(a * a, axis=1, keepdims=True)
        bb = np.sum(b * b, axis=1, keepdims=True).T
        d2 = aa + bb - 2.0 * a @ b.T
        return np.maximum(d2, 0.0)

    @classmethod
    def _rbf(cls, a: np.ndarray, b: np.ndarray, gamma: float) -> np.ndarray:
        return np.exp(-gamma * cls._pairwise_sq(a, b))

    @classmethod
    def _median_bandwidth(cls, x: np.ndarray, y: np.ndarray) -> float:
        z = np.vstack([x, y])
        d2 = cls._pairwise_sq(z, z)
        # Take the upper triangle (i < j) so we don't bias on diagonal zeros.
        n = z.shape[0]
        if n < 2:
            return 0.0
        iu = np.triu_indices(n, k=1)
        d = np.sqrt(d2[iu])
        if d.size == 0:
            return 0.0
        return float(np.median(d))


# ---------------------------------------------------------------------------
# Internal sentinel
# ---------------------------------------------------------------------------


class _DegenerateInput(Exception):
    """Raised internally by KLDriftScorer when a degenerate window is detected."""


__all__ = [
    "DEGENERATE",
    "DriftReading",
    "DriftScorer",
    "KLDriftScorer",
    "MMDDriftScorer",
]
