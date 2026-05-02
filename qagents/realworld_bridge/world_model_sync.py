"""Pure model-reality alignment.

Applies an EMA correction to an MVRI :class:`State` from a partial
:class:`Observation`. Topology is **never** modified; only activations
and tracked edge weights are nudged.

Two safety properties are enforced:

1. The per-dimension correction is bounded by ``max_correction_magnitude``.
2. The corrected state must satisfy ``Inv``; if not, the original state
   is returned unchanged.
"""

from __future__ import annotations

import numpy as np

from qagents.mvri.state import Edge, Inv, State, replace_state
from qagents.realworld_bridge.types import Observation

#: Default per-dimension cap on the EMA correction. Documented as a
#: named constant so it can be inspected and overridden at call sites.
DEFAULT_MAX_CORRECTION: float = 0.1


def _state_to_vector(
    state: State,
    node_ids: tuple[str, ...],
    edge_ids: tuple[Edge, ...],
) -> np.ndarray:
    n_nodes = len(node_ids)
    vec = np.zeros(n_nodes + len(edge_ids), dtype=np.float64)
    for i, n in enumerate(node_ids):
        vec[i] = float(state.activation(n))
    for j, e in enumerate(edge_ids):
        vec[n_nodes + j] = float(state.weight(e))
    return vec


def sync_model(
    model_state: State,
    observation: Observation,
    node_ids: tuple[str, ...],
    edge_ids: tuple[Edge, ...],
    alpha: float,
    *,
    max_correction_magnitude: float = DEFAULT_MAX_CORRECTION,
) -> State:
    """Return a corrected copy of ``model_state`` aligned with ``observation``.

    The correction is bounded element-wise by ``max_correction_magnitude``
    *and* gated by ``Inv``: any correction that would yield an
    Inv-violating state is rejected and the original state is returned.

    The function is pure — neither input is mutated.
    """

    if not (0.0 < float(alpha) <= 1.0):
        raise ValueError(f"alpha must be in (0, 1]; got {alpha!r}")
    if float(max_correction_magnitude) < 0.0:
        raise ValueError("max_correction_magnitude must be non-negative")

    n_nodes = len(node_ids)
    expected_dim = n_nodes + len(edge_ids)
    if len(observation.measurements) != expected_dim:
        raise ValueError(
            f"observation has {len(observation.measurements)} dimensions; "
            f"expected {expected_dim}"
        )

    model_vec = _state_to_vector(model_state, node_ids, edge_ids)
    obs_vec = np.asarray(observation.measurements, dtype=np.float64)
    mask = np.asarray(observation.missing_mask, dtype=bool)

    raw_correction = alpha * (obs_vec - model_vec)
    # Apply mask: missing dimensions get zero correction.
    raw_correction[mask] = 0.0
    # Bound correction element-wise.
    bound = float(max_correction_magnitude)
    correction = np.clip(raw_correction, -bound, bound)

    new_vec = model_vec + correction
    # Clip activations to bounds to preserve Inv.
    lo, hi = model_state.bounds
    new_vec[:n_nodes] = np.clip(new_vec[:n_nodes], lo, hi)

    activations = {
        n: float(new_vec[i]) for i, n in enumerate(node_ids)
    }
    edge_weights = {
        e: float(new_vec[n_nodes + j]) for j, e in enumerate(edge_ids)
    }
    # Preserve untracked edges so topology is unchanged.
    for e, w in model_state.edge_weights:
        edge_weights.setdefault(e, w)

    candidate = replace_state(
        model_state,
        activations=activations,
        edge_weights=edge_weights,
    )
    if not Inv(candidate):
        return model_state
    return candidate


def drift_magnitude(
    model_state: State,
    observation: Observation,
    node_ids: tuple[str, ...],
    edge_ids: tuple[Edge, ...],
) -> float:
    """L2 norm of (model − observation) over non-missing dimensions."""

    expected_dim = len(node_ids) + len(edge_ids)
    if len(observation.measurements) != expected_dim:
        raise ValueError(
            f"observation has {len(observation.measurements)} dimensions; "
            f"expected {expected_dim}"
        )
    model_vec = _state_to_vector(model_state, node_ids, edge_ids)
    obs_vec = np.asarray(observation.measurements, dtype=np.float64)
    mask = np.asarray(observation.missing_mask, dtype=bool)
    diff = (model_vec - obs_vec)[~mask]
    if diff.size == 0:
        return 0.0
    return float(np.linalg.norm(diff))


__all__ = ["DEFAULT_MAX_CORRECTION", "drift_magnitude", "sync_model"]
