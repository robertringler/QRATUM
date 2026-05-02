"""Joint constraint predicate ``Φ(s_prev, s_new)`` for the MVRI.

Each constraint is a named, independently callable, pure predicate:

==========  ========================  =========================================
ID          Name                      Predicate
==========  ========================  =========================================
ER          Entropy Reduction         ``H(s_new) <= H(s_prev) + tol``
CS          Causal Structure          ``edge_set(s_new) ⊆ edge_set(s_prev)``
SS          Structural Stability      ``is_stable(s_new) is True``
IC          Internal Consistency      ``no_invalid_states(s_new) is True``
AUD         Reproducibility           Forward model is deterministic for
                                      fixed ``a`` (re-run check)
==========  ========================  =========================================

``constraint_gate`` evaluates them in the order ``ER → CS → SS → IC → AUD``
and returns ``(True, [])`` only when *every* predicate passes; otherwise
returns ``(False, [failed_ids])`` listing the failures (may include more
than one).
"""

from __future__ import annotations

from qagents.mvri.action_space import Action
from qagents.mvri.forward_model import forward_model
from qagents.mvri.state import H, State, edge_set, is_stable, no_invalid_states

#: Numerical tolerance for entropy comparison.
ER_TOL: float = 1e-9


# --------------------------------------------------------------------------- #
# Individually-callable predicates
# --------------------------------------------------------------------------- #


def entropy_reduction(s_prev: State, s_new: State, tol: float = ER_TOL) -> bool:
    """ER: ``H(s_new) <= H(s_prev) + tol``.

    The tolerance allows numerically equal entropy (no inflation) to pass.
    """

    return H(s_new) <= H(s_prev) + tol


def causal_structure(s_prev: State, s_new: State) -> bool:
    """CS: edge set may only shrink (no new edges)."""

    return edge_set(s_new).issubset(edge_set(s_prev))


def structural_stability(s_new: State) -> bool:
    """SS: bounded activations and finite weights."""

    return is_stable(s_new)


def internal_consistency(s_new: State) -> bool:
    """IC: schema and finiteness checks pass."""

    return no_invalid_states(s_new)


def reproducibility(
    s_prev: State, action: Action | None = None
) -> bool:
    """AUD: ``F(s_prev, action)`` is deterministic for fixed ``action``.

    We verify this *structurally* by re-running the pure forward model
    twice with identical inputs and asserting equality. Because ``F``
    consults no global random state and ``noise_injection`` requires an
    explicit seed, this check passes by construction; it is included
    here so that AUD is *individually callable* and would catch any
    regression that introduced hidden non-determinism.

    If ``action`` is ``None`` the check is treated as vacuous (returns
    ``True``); the structural guarantee is preserved since ``F`` is
    closed over deterministic operations.
    """

    if action is None:
        return True
    a = forward_model(s_prev, action)
    b = forward_model(s_prev, action)
    return a == b


# --------------------------------------------------------------------------- #
# Joint gate
# --------------------------------------------------------------------------- #


def constraint_gate(
    s_prev: State,
    s_new: State,
    action: Action | None = None,
) -> tuple[bool, list[str]]:
    """Evaluate Φ(s_prev, s_new). Returns ``(passed, failed_ids)``.

    Constraints are checked in the canonical order ``ER → CS → SS → IC → AUD``.
    All failures are collected; the function never short-circuits silently.
    """

    failed: list[str] = []

    if not entropy_reduction(s_prev, s_new):
        failed.append("ER")
    if not causal_structure(s_prev, s_new):
        failed.append("CS")
    if not structural_stability(s_new):
        failed.append("SS")
    if not internal_consistency(s_new):
        failed.append("IC")
    if not reproducibility(s_prev, action):
        failed.append("AUD")

    return (not failed, failed)


__all__ = [
    "ER_TOL",
    "causal_structure",
    "constraint_gate",
    "entropy_reduction",
    "internal_consistency",
    "reproducibility",
    "structural_stability",
]
