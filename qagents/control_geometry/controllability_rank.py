"""Controllability rank: SVD-based estimate of local control space.

Given a set of sampled actions ``{a_i}`` and their displacement vectors
``{ΔS_i}``, we form the matrix ``M = [ΔS_1 | ΔS_2 | ... | ΔS_n]`` and
report the numerical rank of ``M`` (i.e. the dimension of the local
controllable subspace).

A high rank means many *linearly independent* directions are reachable
from the current state — the system is locally controllable. A low rank
means the action space is *redundant* or the system is *rigid*: many
actions move state along the same direction.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from qagents.control_geometry.embedding import (
    DisplacementCache,
    action_to_displacement,
)
from qagents.control_geometry.sensitivity import DEFAULT_EPS, vector_dim
from qagents.mvri.action_space import Action
from qagents.mvri.state import State


def displacement_matrix(
    state: State,
    actions: Iterable[Action],
    *,
    eps: float = DEFAULT_EPS,
    cache: DisplacementCache | None = None,
) -> np.ndarray:
    """Stack action displacements as columns of ``M ∈ R^{d × n}``.

    Returns a zero-column matrix of the correct row count when no
    actions are provided.
    """

    cols: list[np.ndarray] = []
    for a in actions:
        cols.append(action_to_displacement(a, state, eps=eps, cache=cache))
    if not cols:
        return np.zeros((vector_dim(state), 0), dtype=float)
    return np.stack(cols, axis=1)


def controllability_spectrum(
    state: State,
    sampled_actions: Iterable[Action],
    *,
    eps: float = DEFAULT_EPS,
    cache: DisplacementCache | None = None,
) -> np.ndarray:
    """Return the singular values of the displacement matrix.

    The vector is non-increasing. This is the raw spectrum from which
    :func:`controllability_rank` is derived.
    """

    m = displacement_matrix(
        state, list(sampled_actions), eps=eps, cache=cache
    )
    if m.size == 0:
        return np.zeros((0,), dtype=float)
    return np.linalg.svd(m, compute_uv=False)


def controllability_rank(
    state: State,
    sampled_actions: Iterable[Action],
    *,
    eps: float = DEFAULT_EPS,
    tol: float | None = None,
    cache: DisplacementCache | None = None,
) -> int:
    """Numerical rank of the local control space.

    Parameters
    ----------
    state:
        Reference state at which actions are probed.
    sampled_actions:
        Candidate actions to probe.
    eps:
        Probe scale (forwarded to :func:`sensitivity_probe`).
    tol:
        Singular-value cutoff. Values strictly greater than ``tol`` are
        counted. When ``None`` (default), ``numpy.linalg.matrix_rank``'s
        adaptive tolerance is used (``max(M.shape) * eps_machine *
        sigma_max``), which is what users typically want.

    Returns
    -------
    int
        ``rank ∈ [0, min(d, n)]``.
    """

    actions = list(sampled_actions)
    m = displacement_matrix(state, actions, eps=eps, cache=cache)
    if m.size == 0:
        return 0
    if tol is None:
        return int(np.linalg.matrix_rank(m))
    return int(np.linalg.matrix_rank(m, tol=float(tol)))


__all__ = [
    "controllability_rank",
    "controllability_spectrum",
    "displacement_matrix",
]
