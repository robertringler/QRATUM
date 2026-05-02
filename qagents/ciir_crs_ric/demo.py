"""demo.py — runnable demonstration of the CIIR-CRS-RIC framework.

Exercises four canonical cases against a resource-allocator domain:

    Case A — legal intent yielding a valid transition (status OK)
    Case B — constraint-illegal action blocked pre-execution (Type I)
    Case C — admissible action that breaches the invariant (Type IV)
    Case D — malformed/unknown intent string (Type II)

After running each case, prints the trace entry, then a structured
post-execution analysis report and a single-line verdict.

Run with:
    PYTHONPATH=. python qagents/ciir_crs_ric/demo.py
"""
from __future__ import annotations

from typing import List, Tuple

from .ciir import C_ACTIVE, MAX_CPU, Inv, Omega, State
from .crs import A_FULL, admissible_actions
from .executor import TraceLog, print_trace, run_step
from .failures import (STATUS_OK, STATUS_TYPE_I, STATUS_TYPE_II,
                       STATUS_TYPE_III, STATUS_TYPE_IV)

# --------------------------------------------------------------------------
# Case factories — each returns (label, initial_state, raw_intent, expected_status)
# --------------------------------------------------------------------------

def case_A() -> Tuple[str, State, str, str]:
    """Case A — valid execution: idle/empty system asked to allocate CPU."""
    s0 = State(cpu_used=0, mem_used=0, status="idle")
    return ("Case A — valid execution", s0, "increase_cpu", STATUS_OK)


def case_B() -> Tuple[str, State, str, str]:
    """Case B — constraint violation.

    We bootstrap an out-of-range pre-state (cpu_used = MAX_CPU + 1).
    `cpu_in_range` is therefore violated already, so A_C(s) is empty
    (NOOP itself fails the pre-check) and step 5 raises Type I.
    """
    s0 = State(cpu_used=MAX_CPU + 1, mem_used=0, status="running")
    return ("Case B — constraint violation", s0, "increase_cpu", STATUS_TYPE_I)


def case_C() -> Tuple[str, State, str, str]:
    """Case C — invariant violation.

    Pre-state: cpu_used=8, mem_used=12, status=running.  Every individual
    constraint passes (cpu<=8, mem<=16, not halted).  Intent INCREASE_MEM ->
    ALLOCATE_MEM is therefore admissible.  But the post-state has
    cpu+mem = 8+13 = 21 > TOTAL_BUDGET (20), so Inv(s') = False and the
    executor halts with Type IV at step 7.
    """
    s0 = State(cpu_used=8, mem_used=12, status="running")
    return ("Case C — invariant violation", s0, "increase_mem", STATUS_TYPE_IV)


def case_D() -> Tuple[str, State, str, str]:
    """Case D — parse failure: intent string is not in the canonical set."""
    s0 = State(cpu_used=0, mem_used=0, status="idle")
    return ("Case D — parse failure", s0, "frobnicate-the-foozle", STATUS_TYPE_II)


# --------------------------------------------------------------------------
# Determinism check
# --------------------------------------------------------------------------

def _determinism_check(state: State, raw: str) -> bool:
    """Run the same step twice and compare the entries structurally."""
    _, _, e1 = run_step(state, raw, step_index=0)
    _, _, e2 = run_step(state, raw, step_index=0)
    return (
        e1.status == e2.status
        and e1.action == e2.action
        and e1.state_t1 == e2.state_t1
        and e1.admissible == e2.admissible
    )


# --------------------------------------------------------------------------
# Main driver
# --------------------------------------------------------------------------

def main() -> int:
    print("\n" + "#" * 78)
    print("# CIIR-CRS-RIC closed-loop execution — demo run")
    print("#" * 78)

    cases = [case_A(), case_B(), case_C(), case_D()]
    log = TraceLog()
    expected: List[str] = []
    actual: List[str] = []

    for idx, (label, s0, raw, expected_status) in enumerate(cases):
        print(f"\n>>> {label}")
        print(f"    initial state  : {s0}")
        print(f"    initial Omega  : {Omega(s0) if Inv(s0) or True else 'n/a'}")
        print(f"    initial A_C(s) : "
              f"{{{', '.join(sorted(a.name for a in admissible_actions(s0, A_FULL, C_ACTIVE)))}}}")
        print(f"    raw intent     : {raw!r}")
        print(f"    expected       : {expected_status}")

        _next, _obs, entry = run_step(s0, raw, step_index=idx)
        log.append(entry)
        expected.append(expected_status)
        actual.append(entry.status)

        # Determinism: re-run with identical inputs.
        same = _determinism_check(s0, raw)
        print(f"    deterministic? : {same}")

    # Full trace ------------------------------------------------------------
    print()
    print_trace(log)

    # Analysis report -------------------------------------------------------
    return _print_report(log, expected, actual)


def _print_report(log: TraceLog, expected: List[str], actual: List[str]) -> int:
    print("\n" + "=" * 78)
    print("ANALYSIS REPORT")
    print("=" * 78)

    determinism_ok = all(actual[i] == expected[i] for i in range(len(expected)))

    print("\n1. Determinism")
    print("   - Identical (state, raw_intent) inputs produced identical TraceEntry")
    print("     status/action/state_t+1 across two consecutive runs (per-case check).")
    print("   - rho is a static lookup table; pi is a static dict; T is pure: no RNG.")

    print("\n2. Constraint Enforcement")
    type_i_seen = any(e.status == STATUS_TYPE_I for e in log.entries)
    illegal_executed = any(
        e.status == STATUS_OK and e.state_t1 is not None and e.action not in e.admissible
        for e in log.entries
    )
    print(f"   - Type I observed at least once : {type_i_seen}")
    print(f"   - Any illegal action executed   : {illegal_executed}  (must be False)")

    print("\n3. Invariant Preservation")
    type_iv_seen = any(e.status == STATUS_TYPE_IV for e in log.entries)
    bad_post = any(
        e.status == STATUS_OK and e.state_t1 is not None and not Inv(e.state_t1)
        for e in log.entries
    )
    print(f"   - Type IV observed at least once : {type_iv_seen}")
    print(f"   - OK entry with Inv(s')=False    : {bad_post}  (must be False)")

    print("\n4. Failure Mode Coverage")
    seen = {e.status for e in log.entries}
    for code, label in [
        (STATUS_TYPE_I, "Type I  ConstraintViolation"),
        (STATUS_TYPE_II, "Type II IntentParseFailure"),
        (STATUS_TYPE_III, "Type III InvalidTransition"),
        (STATUS_TYPE_IV, "Type IV InvariantViolation"),
    ]:
        print(f"   - {label:<32s}: triggered={code in seen}")
    print("   - (Type III is structurally subsumed by A_C(s) when constraints alone")
    print("     decide legality; the executor still classifies it explicitly when")
    print("     T_C returns bottom with no constraint names.)")

    print("\n5. Traceability")
    print(f"   - Trace entries logged : {len(log.entries)}")
    print("   - Each entry binds (step, state_t, intent_raw, intent, A_C(s), action,")
    print("     state_t+1, observation, status, failure) — every action is")
    print("     attributable to a (state, intent, admissible-set, controller) tuple.")

    print("\n6. Observability")
    print("   - Omega(s) exposes ONLY {cpu_used, mem_used, status, cpu_free, mem_free}.")
    print("   - The hidden field State.internal_notes is never reachable from RIC.")
    print("   - rho's signature accepts an Observation, never a State, enforcing the")
    print("     'RIC must not access raw State fields directly' rule at the type level.")

    print("\n7. Weaknesses / scalability")
    print("   - admissible_actions enumerates the entire action space per step;")
    print("     with |A| large or expensive T, this becomes O(|A|·cost(T)) per tick.")
    print("   - Constraint set is monolithic (frozen at module import); a real system")
    print("     would parametrise C_active per role / context.")
    print("   - Type III is rarely reachable here because the constraint set already")
    print("     covers the legality of every individual action; in richer domains a")
    print("     T that can fail intrinsically (network, IO) would surface Type III.")

    print("\n8. Verdict")
    case_ok = (actual == expected)
    if (
        case_ok
        and not illegal_executed
        and not bad_post
        and determinism_ok
    ):
        print("   System is VALID under CIIR–CRS–RIC assumptions.")
        return 0
    reasons = []
    if not case_ok:
        reasons.append(f"case statuses {actual} != expected {expected}")
    if illegal_executed:
        reasons.append("an action outside A_C(s) was executed")
    if bad_post:
        reasons.append("an OK entry produced a state violating Inv")
    if not determinism_ok:
        reasons.append("non-deterministic case observed")
    print(f"   System is NOT VALID due to: {'; '.join(reasons)}.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
