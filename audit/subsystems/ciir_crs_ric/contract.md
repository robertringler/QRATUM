# Subsystem contract: `ciir_crs_ric`
_Commit_: `8fc58a9107334a3b53b69a47580df64b185d3317`  _Generated_: 2026-04-29T22:29:11Z

**Path**: `qagents/ciir_crs_ric/`

## Files
- __init__.py
- ciir.py
- crs.py
- ric.py
- executor.py
- failures.py
- demo.py

## Public types / API surface
- ciir.State, ciir.Constraint, ciir.phi, ciir.Inv, ciir.Omega
- crs.Action (enum), crs.T (pure), crs.T_C (pre+post phi), crs.admissible_actions
- ric.parse_intent, ric.select_action (receives Observation, not State)
- executor.run_step, executor.TraceEntry
- failures: Type I (ConstraintViolation), Type II (InvariantBroken), Type III (ObserverFailure), Type IV (ActionUndefined)

## Invariants
- Inv(phi(s)) holds for every accepted action
- Omega hides internal_notes (no leak of internal state)
- executor.run_step never raises — surfaces TraceEntry errors only
