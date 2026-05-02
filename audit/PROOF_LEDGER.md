# QRATUM Proof Ledger
_Commit_: `8fc58a9107334a3b53b69a47580df64b185d3317`  _Generated_: 2026-04-29T22:27:58Z

Every quantum/CIIR/CRS/RIC capability is mapped to a proof status.
Status ∈ {proved, partially-proved, asserted-only, missing-implementation, missing-test, missing-benchmark, contradicted}.

## Capability ledger

| Capability | Code referent | Test referent | Manuscript referent | Status | Evidence |
|---|---|---|---|---|---|
| CIIR–CRS–RIC strict 9-step loop | `qagents/ciir_crs_ric/executor.py` | `tests/test_ciir_crs_ric_strict.py` | `manuscripts/ciir_monograph/chapters/ch08_measurement_theory.tex` | **partially-proved** | Code + tests present and run deterministically; monograph chapters Ch9+ that formally tie the executor to the CIIR axioms are MISSING (GAP-DOC-MONO-001). |
| CIIR Constraint Algebra (framework) | `qagents/framework/ciir.py` | `tests/test_ciir_crs_ric.py` | `manuscripts/ciir_monograph/chapters/ch04_axiom_system.tex; ch05_algebraic_structure.tex` | **partially-proved** | Algebraic axioms present in Ch4–Ch5; formal connection to CRS T_C and ConstraintAlgebra implementation only narrative. |
| Minimal Viable Reality Injector (MVRI) | `qagents/mvri/ (loop, injector, constraint_gate, metrics)` | `tests/test_mvri.py` | `—` | **missing-proof** | No corresponding manuscript chapter; Δentropy / ΔMI / ΔTE metrics are computed but not derived from a documented theorem. |
| RealWorld Bridge safety gate (MAG/TGT/ROC/MVR) | `qagents/realworld_bridge/safety_gate.py` | `tests/test_realworld_bridge.py` | `—` | **asserted-only** | Gate semantics are encoded as code; no formal soundness/completeness theorem linking gate composition to invariant preservation. |
| Control geometry / reachability (constraint-gated) | `qagents/control_geometry/{sensitivity,reachability,controllability_rank}.py` | `tests/test_control_geometry.py` | `—` | **partially-proved** | SVD-based controllability rank is standard linear-algebra; reachability under nonlinear constraints is asserted via empirical reachability_cost only. |
| Trajectory controller (numeric action vectors) | `qagents/trajectory_controller/{planner,predictor,validator,executor,controller}.py` | `tests/test_trajectory_controller.py` | `—` | **asserted-only** | Determinism is tested per-seed; no formal optimality / convergence proof. |
| Reality Interface Controller v1 (4-key strict output) | `qagents/reality_interface.py` | `tests/test_reality_interface_controller.py` | `qagents/prompts/reality_interface_controller.md` | **asserted-only** | Output schema is enforced by tests; no semantic correctness proof beyond schema conformance. |
| Reality Interface Controller v2 (k-step rollout, anti-deadlock) | `qagents/reality_interface_v2.py` | `tests/test_ric_v2.py` | `qagents/prompts/reality_interface_controller_v2.md` | **asserted-only** | Stochastic perturbation seeded; no statistical guarantee of horizon-K stability. |
| RIC adapters (qratum / ciir / crs) | `qagents/adapters/{qratum,ciir,crs}.py` | `tests/test_ric_adapters.py` | `—` | **asserted-only** | CRS adapter mirrors IMMUTABLE_BOUNDARIES at code level; no proof of behavioral parity with quasim/ciir/. |
| CIIR plasma reconnection control engine | `quasim/ciir/{mhd,topology,spectrum,control,ciir,engine}.py` | `tests/test_plasma_reconnection.py` | `—` | **partially-proved** | FFT/RMHD math is standard; no derivation of FAILURE_MODES F1–F5 from CIIR axioms. |
| Multi-qubit entangled CIIR controller (claimed) | `MISSING — quasim/ciir/multi_qubit/* not in repo` | `MISSING` | `MISSING — chapters/ch21_multi_qubit_controller.tex absent` | **missing-implementation** | GAP-CIIR-MQ-001. |
| Causal Claim falsification protocol (claimed verdict A0) | `MISSING — quasim/ciir/multi_qubit/analysis/causal_claim.py and analysis/falsification.py not in repo` | `MISSING — tests/test_ciir_falsification.py / test_ciir_gaps_1_4_8.py absent` | `MISSING — § 22.4–22.5 of unified extensions chapter absent` | **missing-implementation** | GAP-CIIR-MQ-001 + GAP-CIIR-RUN-002. |
| Phase Theorem (claimed) unifying controllability+threshold+stability | `MISSING` | `MISSING` | `MISSING — Ch21 absent` | **missing-implementation** | GAP-DOC-MONO-001. |
| Quantum-search algorithms | `quantum_search/, quantum/` | `(no dedicated unit-test file enumerated)` | `—` | **missing-test** | Symbol coverage check shows untested symbols here; see audit/tests/coverage_gap.csv. |
| VQE for H2 / LiH / BeH2 | `quasim/quantum/vqe_molecule.py` | `—` | `—` | **missing-implementation** | vqe_molecule.py:244,246 explicitly raises NotImplementedError for LiH and BeH2; only H2 is implemented. |
| cuQuantum backend | `quasim/quantum/core.py` | `—` | `—` | **missing-implementation** | core.py:301 raises NotImplementedError("cuQuantum backend planned for Phase 2 (2026)"). |
