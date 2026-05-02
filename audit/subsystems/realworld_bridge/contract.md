# Subsystem contract: `realworld_bridge`
_Commit_: `8fc58a9107334a3b53b69a47580df64b185d3317`  _Generated_: 2026-04-29T22:29:11Z

**Path**: `qagents/realworld_bridge/`

## Files
- types.py
- sensors.py
- observer.py
- actuators.py
- world_model_sync.py
- safety_gate.py
- control_loop.py
- demo.py

## Public types / API surface
- Observation, ActionResult, ActionValidationResult, LoopTrace, ControlRunResult
- Sensor ABC + MockSensor[SensorExhausted] + RealSensorAdapter[SensorReadError]
- StateObserver (EMA + canonical bijection nodes→edges)
- SafetyConfig
- ControlLoop (strict 9-step)

## Invariants
- model_state advances only on accepted injection
- SensorExhausted ends run early
- validate_action collects all of MAG → TGT → ROC → MVR failures (no early exit)
