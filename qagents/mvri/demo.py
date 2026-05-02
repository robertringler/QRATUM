"""MVRI demonstration: Cases A, B, C, D plus structured analysis report.

Run with::

    PYTHONPATH=. python -m qagents.mvri.demo

The demo is fully deterministic. Every call into the forward model is
seeded; no global random state is consulted.
"""

from __future__ import annotations

import numpy as np

from qagents.mvri.action_space import EPSILON, Action, validate_action
from qagents.mvri.constraint_gate import constraint_gate
from qagents.mvri.forward_model import forward_model
from qagents.mvri.injector import inject
from qagents.mvri.loop import MVRITraceEntry, identity_pipeline, run_mvri_loop
from qagents.mvri.metrics import (
    delta_entropy,
    delta_mutual_information,
    delta_transfer_entropy,
)
from qagents.mvri.policy import PROBE_POLICY, make_probe_policy
from qagents.mvri.state import (
    H,
    Inv,
    State,
    make_state,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def build_initial_state() -> State:
    """A small stable graph well inside its bounds."""

    return make_state(
        activations={"n1": 0.6, "n2": 0.3, "n3": 0.1},
        edge_weights={
            ("n1", "n2"): 0.4,
            ("n2", "n3"): 0.2,
            ("n1", "n3"): 0.1,
        },
        bounds=(-1.0, 1.0),
    )


def build_invalid_initial_state() -> State:
    """Construct a state that *passes* schema but *fails* ``Inv``.

    We do this by claiming ``initial_edges`` that do not include all
    current edges, which causes ``edge_set(s) ⊆ initial_edges`` to fail.
    """

    s = build_initial_state()
    # Drop one edge from initial_edges so the current edge set is no longer
    # a subset; this is the cleanest Inv violation we can produce without
    # corrupting the schema.
    forced_initial = frozenset(s.initial_edges - {("n1", "n3")})
    return State(
        nodes=s.nodes,
        activations=s.activations,
        edges=s.edges,
        edge_weights=s.edge_weights,
        bounds=s.bounds,
        initial_edges=forced_initial,
        step=s.step,
        metadata=s.metadata,
    )


# --------------------------------------------------------------------------- #
# Pretty printing
# --------------------------------------------------------------------------- #


def _fmt_state(s: State) -> str:
    return (
        f"State(step={s.step}, "
        f"acts={dict(s.activations)}, "
        f"edges={sorted(s.edges)}, "
        f"weights={dict(s.edge_weights)}, "
        f"H={H(s):.4f})"
    )


def _print_entry(name: str, entry: MVRITraceEntry) -> None:
    print(f"\n--- {name} ---")
    print(f"step              : {entry.step}")
    print(f"action            : {entry.action}")
    print(f"action_valid      : {entry.action_valid}")
    print(f"outcome           : {entry.outcome}")
    print(f"constraints_passed: {list(entry.constraints_passed)}")
    print(f"constraints_failed: {list(entry.constraints_failed)}")
    print(f"state_t           : {_fmt_state(entry.state_t)}")
    if entry.candidate_state is not None:
        print(f"candidate_state   : {_fmt_state(entry.candidate_state)}")
    else:
        print("candidate_state   : None")
    print(f"state_t1          : {_fmt_state(entry.state_t1)}")
    print(f"metrics           : {entry.metrics}")


# --------------------------------------------------------------------------- #
# Cases
# --------------------------------------------------------------------------- #


def _trace_case(case_name: str, state: State, action: Action) -> MVRITraceEntry:
    """Run a single (state, action) through the MVRI and synthesise the
    canonical ``MVRITraceEntry`` for printing.

    This intentionally mirrors the loop's per-step logic so that each
    printed entry has the full schema, even when we exercise a single
    step outside the closed loop.
    """

    if not Inv(state):
        entry = MVRITraceEntry(
            step=0,
            state_t=state,
            action=action,
            action_valid=False,
            candidate_state=None,
            constraints_passed=(),
            constraints_failed=("INVARIANT_HALT",),
            outcome="invariant_halt",
            state_t1=state,
            metrics={"delta_H": 0.0, "delta_MI": 0.0, "delta_TE": 0.0},
        )
        return entry

    valid = validate_action(action, state, EPSILON)
    if not valid:
        return MVRITraceEntry(
            step=0,
            state_t=state,
            action=action,
            action_valid=False,
            candidate_state=None,
            constraints_passed=(),
            constraints_failed=("INVALID_ACTION",),
            outcome="invalid_action",
            state_t1=state,
            metrics={"delta_H": 0.0, "delta_MI": 0.0, "delta_TE": 0.0},
        )

    candidate = forward_model(state, action)
    passed, failed = constraint_gate(state, candidate, action)
    if passed:
        s_t1 = candidate
        outcome = "accepted"
        passed_ids: tuple[str, ...] = ("ER", "CS", "SS", "IC", "AUD")
        failed_ids: tuple[str, ...] = ()
    else:
        s_t1 = state
        outcome = "rejected"
        passed_ids = tuple(
            c for c in ("ER", "CS", "SS", "IC", "AUD") if c not in failed
        )
        failed_ids = tuple(failed)

    metrics = {
        "delta_H": delta_entropy(state, s_t1),
        "delta_MI": delta_mutual_information(state, s_t1),
        "delta_TE": delta_transfer_entropy(state, s_t1),
    }
    return MVRITraceEntry(
        step=0,
        state_t=state,
        action=action,
        action_valid=True,
        candidate_state=candidate,
        constraints_passed=passed_ids,
        constraints_failed=failed_ids,
        outcome=outcome,
        state_t1=s_t1,
        metrics=metrics,
    )


def case_a() -> MVRITraceEntry:
    """Action within ε, all constraints satisfied → accepted."""

    state = build_initial_state()
    # Reduce a high-mass activation slightly: this *concentrates* mass
    # further (still high) — entropy goes down or stays equal, ER passes.
    action = Action(
        type="node_activation_shift", target="n2", magnitude=-0.04
    )
    return _trace_case("CASE A: within-bound, constraint-satisfying", state, action)


def case_b() -> MVRITraceEntry:
    """Magnitude exceeds ε → rejected at validate_action."""

    state = build_initial_state()
    action = Action(
        type="node_activation_shift", target="n1", magnitude=0.5  # > ε=0.05
    )
    return _trace_case("CASE B: |magnitude| > ε", state, action)


def case_c() -> MVRITraceEntry:
    """Action would increase entropy → rejected at constraint_gate (ER)."""

    state = build_initial_state()
    # Pushing the *smallest* mass upward equalises the distribution and
    # therefore raises Shannon entropy. Magnitude is within ε so this
    # passes validate_action and reaches the gate.
    action = Action(
        type="node_activation_shift", target="n3", magnitude=+0.05
    )
    return _trace_case("CASE C: ER violation", state, action)


def case_d() -> MVRITraceEntry:
    """Initial state violates Inv → Type IV halt before any action."""

    state = build_invalid_initial_state()
    action = Action(
        type="node_activation_shift", target="n1", magnitude=0.0
    )
    return _trace_case("CASE D: Inv violated at S₀", state, action)


# --------------------------------------------------------------------------- #
# Determinism check
# --------------------------------------------------------------------------- #


def determinism_check() -> bool:
    """Same (state, action, seed) twice ⇒ identical outcome."""

    s = build_initial_state()
    a = Action(
        type="noise_injection", target="n2", magnitude=0.03, seed=12345
    )
    s1, st1 = inject(s, a)
    s2, st2 = inject(s, a)
    return s1 == s2 and st1.outcome == st2.outcome


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    print("=" * 72)
    print("MVRI DEMONSTRATION")
    print("=" * 72)
    print(f"epsilon                  : {EPSILON}")
    print(f"PROBE policy identifier  : {PROBE_POLICY}")

    a = case_a()
    b = case_b()
    c = case_c()
    d = case_d()

    _print_entry("CASE A", a)
    _print_entry("CASE B", b)
    _print_entry("CASE C", c)
    _print_entry("CASE D", d)

    # ------------------------------------------------------------------ #
    # Closed-loop probe run for aggregated metrics
    # ------------------------------------------------------------------ #

    rng = np.random.default_rng(2026)
    policy = make_probe_policy(rng)
    result = run_mvri_loop(
        initial_state=build_initial_state(),
        policy=policy,
        pipeline=identity_pipeline,
        n_steps=20,
        epsilon=EPSILON,
    )

    accepted_metrics = [
        e.metrics for e in result.trace if e.outcome == "accepted"
    ]
    if accepted_metrics:
        avg_dh = sum(m["delta_H"] for m in accepted_metrics) / len(
            accepted_metrics
        )
        avg_dmi = sum(m["delta_MI"] for m in accepted_metrics) / len(
            accepted_metrics
        )
        avg_dte = sum(m["delta_TE"] for m in accepted_metrics) / len(
            accepted_metrics
        )
    else:
        avg_dh = avg_dmi = avg_dte = 0.0

    print("\n" + "=" * 72)
    print("ANALYSIS REPORT")
    print("=" * 72)

    det_ok = determinism_check()
    print("\n1. Determinism")
    print(
        f"   identical (state, action, seed) → identical outcome: {det_ok}"
    )

    print("\n2. Constraint Enforcement")
    print(f"   CASE A failed constraints: {list(a.constraints_failed)}")
    print(f"   CASE B failed constraints: {list(b.constraints_failed)}")
    print(f"   CASE C failed constraints: {list(c.constraints_failed)}")
    print(f"   CASE D failed constraints: {list(d.constraints_failed)}")

    print("\n3. Invariant Preservation")
    inv_ok = all(
        Inv(e.state_t1) for e in (a, b, c, d) if e.outcome == "accepted"
    )
    inv_ok = inv_ok and all(
        Inv(e.state_t1)
        for e in result.trace
        if e.outcome == "accepted"
    )
    print(f"   all accepted states satisfy Inv: {inv_ok}")

    print("\n4. Rejection Semantics")
    rej_unchanged = (
        b.state_t == b.state_t1
        and c.state_t == c.state_t1
        and d.state_t == d.state_t1
    )
    print(f"   rejected/halt steps left state unchanged: {rej_unchanged}")

    print("\n5. Metric Behavior")
    print(f"   loop n_steps          : {result.n_steps}")
    print(f"   acceptance_rate       : {result.acceptance_rate():.4f}")
    print(f"   violation_count       : {result.violation_count()}")
    print(f"   <ΔH>  (accepted)      : {avg_dh:+.4e}")
    print(f"   <ΔMI> (accepted)      : {avg_dmi:+.4e}")
    print(f"   <ΔTE> (accepted)      : {avg_dte:+.4e}")
    if abs(avg_dh) < 1e-6:
        print(
            "   interpretation        : ΔH ≈ 0 → system is "
            f"*stiff* under PROBE_POLICY at ε={EPSILON:.3f}"
        )

    print("\n6. Policy Independence")
    print(
        "   loop signature accepts arbitrary "
        "Callable[[State], Action]; PROBE_POLICY is one of many."
    )

    print("\n7. Weaknesses")
    print(
        "   - AUD is verified structurally (re-run F twice). It cannot\n"
        "     detect non-determinism that escapes via hidden global state\n"
        "     introduced by future refactors; CI must continue to enforce\n"
        "     'no global RNG' lint.\n"
        "   - CS is trivially satisfied for the three current action types\n"
        "     because none of them add edges. CS becomes load-bearing only\n"
        "     when the action space is extended.\n"
        "   - Probe policies that systematically push entropy upward will\n"
        "     produce pathologically high rejection rates by construction;\n"
        "     this is a feature of MVRI, not a bug, but operators should\n"
        "     monitor acceptance_rate as a stiffness signal."
    )

    print("\n8. Verdict")
    if det_ok and inv_ok and rej_unchanged:
        print(
            "   MVRI is VALID: constraint-preserving, deterministic, "
            "and rejection-complete."
        )
    else:
        reasons = []
        if not det_ok:
            reasons.append("non-deterministic forward path")
        if not inv_ok:
            reasons.append("accepted state violated Inv")
        if not rej_unchanged:
            reasons.append("rejection mutated state")
        print(f"   MVRI is NOT VALID due to: {reasons}")


if __name__ == "__main__":
    main()
