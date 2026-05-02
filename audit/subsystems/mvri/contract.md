# Subsystem contract: `mvri`
_Commit_: `8fc58a9107334a3b53b69a47580df64b185d3317`  _Generated_: 2026-04-29T22:29:11Z

**Path**: `qagents/mvri/`

## Files
- state.py
- action_space.py
- forward_model.py
- constraint_gate.py
- injector.py
- loop.py
- metrics.py
- policy.py
- demo.py

## Public types / API surface
- State (typed immutable, H/Inv/is_stable/no_invalid_states)
- Action + validate_action (EPSILON=0.05)
- InjectionStatus
- MVRITraceEntry
- LoopResult

## Invariants
- Bounded EMA correction (DEFAULT_MAX_CORRECTION=0.1)
- Δentropy / ΔMI / ΔTE computed deterministically
