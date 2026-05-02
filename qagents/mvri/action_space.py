"""Action space ``A`` for the MVRI.

An action is a *bounded* perturbation specification. The forward model is
total — it accepts any well-formed ``Action`` — but ``validate_action`` is
used as a precondition gate inside the loop so that out-of-bound or
ill-targeted probes are rejected before they reach ``F``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from qagents.mvri.state import State, parse_edge

#: Default bound for ``|magnitude|``.
EPSILON: float = 0.05

ActionType = Literal[
    "edge_weight_adjust",
    "node_activation_shift",
    "noise_injection",
]

_VALID_TYPES: frozenset[str] = frozenset(
    {"edge_weight_adjust", "node_activation_shift", "noise_injection"}
)


@dataclass(frozen=True)
class Action:
    """Bounded perturbation specification.

    Parameters
    ----------
    type:
        One of ``edge_weight_adjust``, ``node_activation_shift``,
        ``noise_injection``.
    target:
        Node id (e.g. ``"n1"``) or directed edge id ``"src->dst"``.
    magnitude:
        Real-valued perturbation; must satisfy ``|magnitude| <= epsilon``.
    seed:
        Required (and only meaningful) for ``noise_injection``. Used to
        construct a deterministic RNG inside the forward model. If
        ``None`` for ``noise_injection``, validation fails.
    """

    type: ActionType
    target: str
    magnitude: float
    seed: int | None = None


def validate_action(a: Action, s: State, epsilon: float = EPSILON) -> bool:
    """Pure precondition check.

    Returns ``True`` iff the action is well-formed *and* applicable to ``s``.
    Specifically:

    - ``type`` is one of the three allowed strings.
    - ``magnitude`` is finite and ``|magnitude| <= epsilon``.
    - ``target`` resolves to an existing node or edge in ``s``.
    - ``noise_injection`` carries an explicit integer ``seed``.
    - The action does *not* introduce a new node or edge.
    """

    if a.type not in _VALID_TYPES:
        return False
    try:
        m = float(a.magnitude)
    except (TypeError, ValueError):
        return False
    # Reject NaN / inf
    if m != m or m in (float("inf"), float("-inf")):
        return False
    if abs(m) > float(epsilon):
        return False

    if a.type == "edge_weight_adjust":
        if "->" not in a.target:
            return False
        try:
            edge = parse_edge(a.target)
        except ValueError:
            return False
        return edge in s.edges

    if a.type == "node_activation_shift":
        if "->" in a.target:
            return False
        return a.target in s.nodes

    if a.type == "noise_injection":
        if a.seed is None or not isinstance(a.seed, int):
            return False
        if "->" in a.target:
            return False
        return a.target in s.nodes

    return False  # pragma: no cover - defensive


__all__ = ["Action", "ActionType", "EPSILON", "validate_action"]
