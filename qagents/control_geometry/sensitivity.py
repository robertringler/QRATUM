"""Sensitivity layer: local action → state displacement estimation.

The MVRI ``State`` is a typed graph (sorted activations + sorted edge
weights). We flatten it into a *fixed-dimension* vector for any given
state shape so that displacement vectors can be compared and stacked:

``vec(s) = [activations sorted by node id, edge_weights sorted by edge]``

Sensitivity is computed by *scaling* a candidate ``Action.magnitude`` by
a small factor ``eps`` and asking the deterministic forward model what
state it produces. Because ``forward_model`` is pure and never consults
global state (and ``noise_injection`` requires an explicit seed), the
sensitivity probe is fully deterministic.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Final

import numpy as np

from qagents.mvri.action_space import Action
from qagents.mvri.forward_model import forward_model
from qagents.mvri.state import State

#: Default finite-difference scale for the probe.
DEFAULT_EPS: Final[float] = 1e-3


def state_to_vector(s: State) -> np.ndarray:
    """Flatten a State into a fixed-order numeric vector.

    Order: activations sorted by node id, then edge weights sorted by
    edge id. The ordering is canonical because ``State.activations`` and
    ``State.edge_weights`` are stored sorted.
    """

    parts: list[float] = [v for _, v in s.activations]
    parts.extend(w for _, w in s.edge_weights)
    return np.asarray(parts, dtype=float)


def vector_dim(s: State) -> int:
    """Return ``dim(vec(s))`` for the given state shape."""

    return len(s.activations) + len(s.edge_weights)


def _scaled_action(action: Action, eps: float) -> Action:
    """Return ``action`` with magnitude scaled by ``eps`` (pure)."""

    return replace(action, magnitude=float(action.magnitude) * float(eps))


def sensitivity_probe(
    state: State,
    action: Action,
    eps: float = DEFAULT_EPS,
) -> np.ndarray:
    """Compute ΔS ≈ (F(s, εa) − vec(s)) / ε.

    Parameters
    ----------
    state:
        Current MVRI state ``s``.
    action:
        Candidate action ``a``. Its magnitude is rescaled by ``eps``
        before being passed to ``F`` so the result approximates the
        local linear response. ``noise_injection`` actions remain
        deterministic because the seed is preserved on the rescaled
        action.
    eps:
        Probe scale. Must be strictly positive.

    Returns
    -------
    np.ndarray
        Fixed-dimension displacement vector of shape ``(vector_dim(state),)``.
        Always returns finite values: if a target is unknown F returns the
        state unchanged and the probe returns the zero vector.
    """

    if not (eps > 0.0):
        raise ValueError(f"eps must be positive, got {eps!r}")

    base = state_to_vector(state)
    probed = state_to_vector(forward_model(state, _scaled_action(action, eps)))
    if probed.shape != base.shape:
        raise RuntimeError(
            "forward_model returned a state of different shape; cannot probe"
        )
    return (probed - base) / float(eps)


__all__ = [
    "DEFAULT_EPS",
    "sensitivity_probe",
    "state_to_vector",
    "vector_dim",
]
