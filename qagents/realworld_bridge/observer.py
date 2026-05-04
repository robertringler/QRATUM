"""Deterministic state observer.

Implements an exponential-moving-average estimator that consumes
:class:`Observation` records and returns MVRI-compatible :class:`State`
values via an explicit canonical bijection:

* indices ``0 .. n_nodes-1``  →  node activations (in ``node_ids`` order)
* indices ``n_nodes ..``      →  edge weights (in ``edge_ids`` order)

The observer is deterministic: identical input sequences yield identical
output sequences. No global RNG, no hidden state.
"""

from __future__ import annotations

import numpy as np

from qagents.mvri.state import Edge, Inv, State, replace_state
from qagents.realworld_bridge.types import Observation

# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class ObserverStateInvalid(RuntimeError):  # noqa: N818 - name fixed by spec
    """Raised when ``get_state()`` would return a state failing ``Inv``."""


# --------------------------------------------------------------------------- #
# Observer
# --------------------------------------------------------------------------- #


class StateObserver:
    """Deterministic EMA-based state observer."""

    def __init__(
        self,
        initial_state: State,
        smoothing_alpha: float,
        node_ids: tuple[str, ...],
        edge_ids: tuple[Edge, ...],
    ) -> None:
        if not (0.0 < float(smoothing_alpha) <= 1.0):
            raise ValueError(f"smoothing_alpha must be in (0, 1]; got {smoothing_alpha!r}")

        # Validate that node/edge ids match the initial state schema.
        if tuple(node_ids) != tuple(initial_state.nodes):
            raise ValueError("node_ids must equal initial_state.nodes (same order)")
        for e in edge_ids:
            if e not in initial_state.edges:
                raise ValueError(f"edge_id {e!r} not present in initial_state.edges")

        self._alpha: float = float(smoothing_alpha)
        self._node_ids: tuple[str, ...] = tuple(node_ids)
        self._edge_ids: tuple[Edge, ...] = tuple(edge_ids)
        self._initial_state: State = initial_state

        # Internal float64 estimate vector — never exposed by reference.
        self._estimate: np.ndarray = self._state_to_vector(initial_state)
        self._dim: int = self._estimate.shape[0]
        self._has_update: bool = False
        self._last_delta_inf: float = float("inf")

    # ------------------------------------------------------------------ #
    # Bijection state ↔ vector
    # ------------------------------------------------------------------ #

    def _state_to_vector(self, s: State) -> np.ndarray:
        n_nodes = len(self._node_ids)
        n_edges = len(self._edge_ids)
        vec = np.zeros(n_nodes + n_edges, dtype=np.float64)
        for i, node in enumerate(self._node_ids):
            vec[i] = float(s.activation(node))
        for j, edge in enumerate(self._edge_ids):
            vec[n_nodes + j] = float(s.weight(edge))
        return vec

    def _vector_to_state(self, vec: np.ndarray) -> State:
        # Clip to bounds to keep activations valid.
        lo, hi = self._initial_state.bounds
        n_nodes = len(self._node_ids)
        clipped = np.clip(vec, lo, hi)
        activations = {node: float(clipped[i]) for i, node in enumerate(self._node_ids)}
        edge_weights = {edge: float(vec[n_nodes + j]) for j, edge in enumerate(self._edge_ids)}
        # Preserve any edges in the initial state that we are not
        # tracking (so the observer cannot introduce/remove edges).
        for e, w in self._initial_state.edge_weights:
            edge_weights.setdefault(e, w)
        return replace_state(
            self._initial_state,
            activations=activations,
            edge_weights=edge_weights,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def update(self, observation: Observation) -> None:
        """Apply EMA on non-missing dimensions of ``observation``."""

        if len(observation.measurements) != self._dim:
            raise ValueError(
                f"observation has {len(observation.measurements)} "
                f"dimensions; expected {self._dim}"
            )

        obs_vec = np.asarray(observation.measurements, dtype=np.float64)
        mask = np.asarray(observation.missing_mask, dtype=bool)

        new_estimate = self._estimate.copy()
        # Only update non-missing dimensions.
        non_missing = ~mask
        if np.any(non_missing):
            new_estimate[non_missing] = (
                self._alpha * obs_vec[non_missing]
                + (1.0 - self._alpha) * self._estimate[non_missing]
            )

        delta = new_estimate - self._estimate
        # Track per-update L∞ change for convergence test.
        self._last_delta_inf = float(np.max(np.abs(delta)))
        self._estimate = new_estimate
        self._has_update = True

    def get_state(self) -> State:
        s = self._vector_to_state(self._estimate)
        if not Inv(s):
            raise ObserverStateInvalid(
                "observer estimate violates MVRI Inv() — graph topology, "
                "bounds, or schema corrupted"
            )
        return s

    def is_converged(self, tol: float) -> bool:
        if not self._has_update:
            return False
        return self._last_delta_inf < float(tol)

    # ------------------------------------------------------------------ #
    # Introspection (read-only)
    # ------------------------------------------------------------------ #

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def node_ids(self) -> tuple[str, ...]:
        return self._node_ids

    @property
    def edge_ids(self) -> tuple[Edge, ...]:
        return self._edge_ids

    def estimate_vector(self) -> np.ndarray:
        """Return a copy of the internal estimate vector."""
        return self._estimate.copy()


__all__ = ["ObserverStateInvalid", "StateObserver"]
