"""Deterministic, total forward model ``F : S × A → S``.

Pure: never mutates inputs, never reads global state, never raises on
well-formed inputs. The only error path is a hard ``ForwardModelError``
when an unknown action type is supplied (which is impossible if the
caller has run ``validate_action`` first, but the model remains total
in the sense that *every recognised action type is handled*).
"""

from __future__ import annotations

import numpy as np

from qagents.mvri.action_space import Action
from qagents.mvri.state import (
    ForwardModelError,
    State,
    parse_edge,
    replace_state,
)


def _clip(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def forward_model(state: State, action: Action) -> State:
    """Apply ``action`` to ``state`` and return a new ``State``.

    The model is deterministic: identical inputs ⇒ identical output. For
    ``noise_injection`` an explicit seed (drawn from ``action.seed``) is
    used to construct a local ``numpy.random.Generator``; no global
    random state is ever consulted.

    Notes
    -----
    - The input ``state`` is never mutated.
    - Activations are clipped to ``state.bounds`` so the result is
      always inside the declared activation envelope.
    - This function does *not* enforce ``Φ`` — the constraint gate is
      responsible for rejecting transitions that violate constraints.
    """

    lo, hi = state.bounds

    if action.type == "edge_weight_adjust":
        edge = parse_edge(action.target)
        new_weights = dict(state.edge_weights)
        if edge not in new_weights:
            # Total model: edge missing — return state unchanged. Validation
            # is the gate, not F.
            return replace_state(state, step=state.step + 1)
        new_weights[edge] = float(new_weights[edge]) + float(action.magnitude)
        return replace_state(state, edge_weights=new_weights, step=state.step + 1)

    if action.type == "node_activation_shift":
        node = action.target
        new_acts = dict(state.activations)
        if node not in new_acts:
            return replace_state(state, step=state.step + 1)
        new_acts[node] = _clip(float(new_acts[node]) + float(action.magnitude), lo, hi)
        return replace_state(state, activations=new_acts, step=state.step + 1)

    if action.type == "noise_injection":
        if action.seed is None:
            # Forward model is total but cannot proceed without a seed:
            # surface this clearly. (validate_action would have caught it.)
            raise ForwardModelError("noise_injection requires an explicit integer seed")
        rng = np.random.default_rng(int(action.seed))
        node = action.target
        new_acts = dict(state.activations)
        if node not in new_acts:
            return replace_state(state, step=state.step + 1)
        # Symmetric uniform draw scaled by |magnitude|; deterministic
        # given (seed, target, magnitude).
        delta = float(rng.uniform(-1.0, 1.0)) * float(action.magnitude)
        new_acts[node] = _clip(float(new_acts[node]) + delta, lo, hi)
        return replace_state(state, activations=new_acts, step=state.step + 1)

    raise ForwardModelError(f"unhandled action type: {action.type!r}")


__all__ = ["forward_model"]
