# Missing property/fuzz tests
_Commit_: `8fc58a9107334a3b53b69a47580df64b185d3317`  _Generated_: 2026-04-29T22:27:58Z

Modules where invariants are claimed but no Hypothesis / fuzz / property test was found:

| Module | Invariant | Existing test | Property-test? |
|---|---|---|---|
| `qagents/ciir_crs_ric/ciir.py` | `phi(s) ∈ Inv` for all admissible s | `tests/test_ciir_crs_ric_strict.py` | ❌ no Hypothesis decorator detected |
| `qagents/realworld_bridge/safety_gate.py` | MAG/TGT/ROC/MVR closed under composition | `tests/test_realworld_bridge.py` | ❌ |
| `qagents/control_geometry/reachability.py` | reachability_cost ≥ 0, monotone in step | `tests/test_control_geometry.py` | ❌ |
| `qagents/mvri/constraint_gate.py` | ER ∧ CS ∧ SS ∧ IC ∧ AUD reject ⇒ injection rejected | `tests/test_mvri.py` | ❌ |
| `qagents/trajectory_controller/planner.py` | deterministic per-seed | `tests/test_trajectory_controller.py` | ❌ no byte-identical re-run assertion |
