"""Probe policies for the MVRI loop.

The loop accepts *any* ``Callable[[State], Action]``; this module supplies
a reference probe policy intended for system probing. It is labelled
``PROBE_POLICY`` and is kept deliberately minimal.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from qagents.mvri.action_space import EPSILON, Action
from qagents.mvri.state import State, format_edge

#: Identifier so any loop / report can confirm which policy was in use.
PROBE_POLICY: str = "PROBE_POLICY"


def probe_policy(
    state: State,
    rng: np.random.Generator,
    *,
    epsilon: float = EPSILON,
) -> Action:
    """Reference probe policy.

    Picks uniformly between the three action types (where available),
    chooses an existing node or edge target, and draws a magnitude
    uniformly from ``[-epsilon, +epsilon]``. The RNG is passed in
    explicitly — this function never reads global random state.
    """

    if not isinstance(rng, np.random.Generator):
        raise TypeError("probe_policy requires an explicit numpy Generator")

    available_types: list[str] = ["node_activation_shift", "noise_injection"]
    if state.edges:
        available_types.append("edge_weight_adjust")

    a_type = available_types[int(rng.integers(0, len(available_types)))]
    magnitude = float(rng.uniform(-epsilon, epsilon))

    if a_type == "edge_weight_adjust":
        edges_sorted = sorted(state.edges)
        edge = edges_sorted[int(rng.integers(0, len(edges_sorted)))]
        return Action(
            type="edge_weight_adjust",
            target=format_edge(edge),
            magnitude=magnitude,
        )

    nodes_sorted = list(state.nodes)
    node = nodes_sorted[int(rng.integers(0, len(nodes_sorted)))]

    if a_type == "noise_injection":
        seed = int(rng.integers(0, 2**31 - 1))
        return Action(
            type="noise_injection",
            target=node,
            magnitude=magnitude,
            seed=seed,
        )

    return Action(
        type="node_activation_shift",
        target=node,
        magnitude=magnitude,
    )


def make_probe_policy(
    rng: np.random.Generator,
    *,
    epsilon: float = EPSILON,
) -> Callable[[State], Action]:
    """Bind ``rng`` to produce a stateful (but RNG-explicit) policy callable."""

    def _policy(state: State) -> Action:
        return probe_policy(state, rng, epsilon=epsilon)

    _policy.__name__ = "probe_policy_bound"
    _policy.policy_id = PROBE_POLICY  # type: ignore[attr-defined]
    return _policy


__all__ = ["PROBE_POLICY", "make_probe_policy", "probe_policy"]
