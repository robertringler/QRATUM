# Subsystem contract: `framework`
_Commit_: `8fc58a9107334a3b53b69a47580df64b185d3317`  _Generated_: 2026-04-29T22:29:11Z

**Path**: `qagents/framework/`

## Files
- ciir.py (Constraint, ConstraintAlgebra, ObserverMap)
- crs.py (CRS, Action, FailureMode I-IV, T_C)
- ric.py (RIC, Intent, TraceEntry, 9-step loop)

## Public types / API surface
- ConstraintAlgebra(meet, join, comp, leq)
- FailureMode (I-IV)
- RIC.run_step()

## Invariants
- 9-step loop terminates
- TraceEntry is append-only
