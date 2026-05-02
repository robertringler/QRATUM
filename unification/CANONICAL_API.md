# QRATUM Canonical API
_Phase_: U1 Canonical Architecture  
_Generated_: 2026-04-30

## Canonical Type System

### Core Control Types (qagents/reality_interface.py)

| Type | Fields | Invariants |
|------|--------|-----------|
| `IntentInterpretation` | goal, constraints, priority | frozen dataclass |
| `SelectedAction` | type ∈ {control,adjust,hold,abort}, magnitude ∈ [0,1], confidence ∈ [0,1] | frozen |
| `PredictedOutcome` | expected_state, risk ∈ [0,1], stability_score ∈ [0,1] | frozen |
| `FallbackPlan` | trigger_condition, action | frozen |
| `ControlDecision` | intent_interpretation, selected_action, predicted_outcome, fallback_plan | frozen |
| `RICDecision` | intent_interpretation, selected_action, predicted_outcome, fallback_plan | frozen, strict 4-key |

### CIIR–CRS–RIC Types (qagents/ciir_crs_ric/)

| Type | Module | Invariants |
|------|--------|-----------|
| `State` | ciir.py | Immutable, typed fields |
| `Constraint` | ciir.py | phi: State → bool |
| `Action` | crs.py | Enum {INCREASE, DECREASE, HOLD, RESET} |
| `Observation` | ric.py | Received by RIC, not State |
| `TraceEntry` | executor.py | 9-step pipeline record |

### MVRI Types (qagents/mvri/)

| Type | Module | Fields |
|------|--------|-------|
| `State` | state.py | H, Inv, is_stable |
| `Action` | action_space.py | validate_action, EPSILON=0.05 |
| `InjectionStatus` | injector.py | ACCEPTED/REJECTED/CONSTRAINED |
| `MVRITraceEntry` | loop.py | per-step trace |

## Failure Modes

| Code | Name | When |
|------|------|------|
| Type I | ConstraintViolation | action violates a hard constraint |
| Type II | FeasibilityFailure | world state infeasible |
| Type III | SafetyDominance | risk > threshold → abort |
| Type IV | StabilityCollapse | stability < sigma_min |
| F1-F5 | Plasma failures | CIIR plasma reconnection specific |

## Run-Result Schema

```python
{
    "intent_interpretation": {"goal": str, "constraints": dict, "priority": float},
    "selected_action": {"type": str, "magnitude": float, "confidence": float},
    "predicted_outcome": {"expected_state": dict, "risk": float, "stability_score": float},
    "fallback_plan": {"trigger_condition": str, "action": str}
}
```

## Simulator / Proposer Protocols

```python
Simulator = Callable[[str, float, Mapping, Mapping], PredictedOutcome]
Proposer  = Callable[[Mapping, Mapping, Mapping], tuple[str, float]]
```

## Public API Invariants

1. `RealityInterfaceController.step(payload)` is **deterministic** for fixed (intent, world_state, system_limits)
2. `selected_action.type` ∈ `{"control", "adjust", "hold", "abort"}`
3. `selected_action.magnitude` ∈ `[0.0, 1.0]`
4. `selected_action.confidence` ∈ `[0.0, 1.0]`
5. `predicted_outcome.risk` ∈ `[0.0, 1.0]`
6. `predicted_outcome.stability_score` ∈ `[0.0, 1.0]`
7. `fallback_plan` is **never omitted** (always present)
8. CIIR 9-step loop **never raises** (executor catches all exceptions into trace)
9. MVRI gate: ER ∧ CS ∧ SS ∧ IC ∧ AUD — all must pass for injection
10. Safety gate: MAG → TGT → ROC → MVR — failures collected, not short-circuited
