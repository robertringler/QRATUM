# Subsystem contract: `trajectory_controller`

_Commit_: `8fc58a9107334a3b53b69a47580df64b185d3317`  _Generated_: 2026-04-29T22:29:11Z

**Path**: `qagents/trajectory_controller/`

## Files

- types.py
- planner.py
- predictor.py
- validator.py
- executor.py
- controller.py
- demo.py

## Public types / API surface

- Numeric action vectors [type_code, target_index, magnitude, seed]
- planner / predictor / validator / executor

## Invariants

- Deterministic per-seed
- Operates over RealWorldBridge + MVRI
