# QRATUM Proof Ledger — Final

_Phase_: U6 Quantum Proof Closure  
_Generated_: 2026-04-30  
_Input_: `audit/PROOF_LEDGER.md`

Status ∈ {proved, partial, retracted}

## Capability Proof Status

| Capability | Code | Tests | Manuscript | Final Status | Evidence |
|-----------|------|-------|------------|-------------|---------|
| CIIR–CRS–RIC strict 9-step loop | `qagents/ciir_crs_ric/executor.py` | `test_ciir_crs_ric_strict.py` (33 tests) | `ch08_measurement_theory.tex` (stub) | **partial** | Code + tests deterministic; formal axiom-to-executor proof pending ch09+ completion |
| CIIR Constraint Algebra (framework) | `qagents/framework/ciir.py` | `test_ciir_crs_ric.py` (48 tests) | `ch04_axiom_system.tex`, `ch05_algebraic_structure.tex` | **partial** | Axioms in ch4–5; formal CRS T_C connection narrative only |
| MVRI (ER/CS/SS/IC/AUD gates) | `qagents/mvri/` | `test_mvri.py` (38 tests) | — | **partial** | Gates tested; no formal soundness theorem |
| RealWorld Bridge safety gate | `qagents/realworld_bridge/safety_gate.py` | `test_realworld_bridge.py` (45 tests) | — | **partial** | Gate semantics tested; no formal completeness proof |
| Control geometry / reachability | `qagents/control_geometry/` | `test_control_geometry.py` (40 tests) | — | **partial** | SVD-based rank is standard; reachability asserted empirically |
| Trajectory controller | `qagents/trajectory_controller/` | `test_trajectory_controller.py` | — | **partial** | Determinism tested at fixed seed; no convergence proof |
| RIC v1 (4-key strict output) | `qagents/reality_interface.py` | `test_reality_interface_controller.py` (24 tests) | `prompts/reality_interface_controller.md` | **partial** | Schema enforced; no semantic correctness proof |
| RIC v2 (k-step rollout, anti-deadlock) | `qagents/reality_interface_v2.py` | `test_ric_v2.py` (18 tests) | `prompts/reality_interface_controller_v2.md` | **partial** | Stochastic perturbation seeded; no K-stability guarantee |
| RIC adapters (qratum/ciir/crs) | `qagents/adapters/` | `test_ric_adapters.py` (23 tests) | — | **partial** | Round-trips tested; no behavioral parity proof |
| CIIR plasma reconnection | `quasim/ciir/plasma/` | `test_plasma_reconnection.py` (55 tests) | — | **partial** | RMHD math standard; F1–F5 from CIIR axioms not derived |
| Multi-qubit CIIR controller | `quasim/ciir/multi_qubit/controller.py` | stub | `ch21_multi_qubit_controller.tex` (stub) | **partial** | Minimal implementation created (GAP-CIIR-MQ-001); full proof pending |
| Causal claim falsification (Verdict A0) | `quasim/ciir/multi_qubit/analysis/falsification.py` | stub | `ch22_unified_extensions.tex` (stub) | **partial** | Protocol stub created; numerical verification pending |
| Phase Theorem (controllability+threshold+stability) | — | — | `ch21_multi_qubit_controller.tex` (stub) | **partial** | Claimed in monograph stub; proof pending |
| Quantum-search algorithms | `quantum_search/`, `quantum/` | (no dedicated tests) | — | **partial** | Code present; no unit tests |
| VQE for H2 | `quasim/quantum/vqe_molecule.py` | — | — | **partial** | H2 implemented; LiH/BeH2 raise ValueError (GAP-STUB-023/024) |
| VQE for LiH/BeH2 | `quasim/quantum/vqe_molecule.py` | — | — | **retracted** | Raises ValueError: not implemented; removed from capability claims |
| cuQuantum backend | `quasim/quantum/core.py` | — | — | **retracted** | Raises ValueError: Phase 2 (2026); removed from Phase 1 capability claims |
| CIIR monograph (Ch1-22) | — | — | `manuscripts/ciir_monograph/chapters/ch01-ch22.tex` | **partial** | Chapter stubs created; content pending |

## Retracted Capabilities

Per Rule §U6 item 4 (Formal retraction), the following are explicitly retracted from shipped APIs:

1. **LiH/BeH2 VQE** (`quasim/quantum/vqe_molecule.py`) — raises `ValueError` with explicit message
2. **cuQuantum backend** (`quasim/quantum/core.py`) — raises `ValueError` with Phase 2 timeline

## Remaining Proof Work

All 397 formal claims in `audit/theory/claims_matrix.csv` (GAP-THEORY-001) require individual proof closure. This is beyond the scope of this remediation phase and requires dedicated theoretical work per claim.

**Priority**: CIIR 9-step loop formal proof (connects most other proofs).
