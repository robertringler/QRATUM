# QRATUM (formerly QuASIM)

**Classical simulation framework with planned quantum extensions.**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-research%2Fbeta-yellow.svg)](#what-is-actually-here)

> **TRANSPARENCY NOTICE**
>
> This README has been rewritten from a deterministic, evidence-based audit of the
> source tree. Earlier versions of this README contained extensive marketing
> claims (Byzantine-fault-tolerant decentralized ghost machine, 14 verticals,
> "quantum-resilient computing platform", DO-178C alignment, named industry
> pilots, etc.) that **could not be substantiated** from the code, tests, or
> runtime behavior in this repository. Those claims have been removed.
>
> What follows is what is **actually present, importable, and testable** in the
> tree on this branch as of the audit commit. Anything not listed here
> should be treated as unverified. See
> [`QUANTUM_CAPABILITY_AUDIT.md`](QUANTUM_CAPABILITY_AUDIT.md) and
> [`_DOCUMENTATION_DISCLAIMER.md`](_DOCUMENTATION_DISCLAIMER.md) for
> historical context.

---

## What is actually here

QRATUM is a Python-first research codebase organized around a small core that
**does work** and a much larger periphery of partial, experimental, or stalled
subsystems. The verifiable core is built around a deterministic
constraint-governed control framework called **CIIR–CRS–RIC** and several
companion modules (MVRI, control geometry, real-world bridge, trajectory
controller, RIC adapters). These modules are the only parts of the repository
that are continuously executable and covered by passing tests in this clone.

The repository also contains:

- **Rust crates** for cryptographic and networking primitives (mixed build state).
- **Genomics** support code under `qrVITRA/merkler-static/` (Rust, biokey + ZKP).
- **Plasma reconnection** simulation under `quasim/ciir/plasma/` (Python, NumPy/SciPy).
- A large amount of **stalled or non-importable** code (`api.v1`, `aion.executor`,
  `quantum.python`, several `qnx*` paths) that test files still try to import.
- A large **documentation corpus** (130+ Markdown files at the repository root)
  much of which describes goals, plans, and historical claims rather than
  shipping behavior.

A summary of what builds, imports, and tests cleanly is given in the
[Execution Truth Table](#execution-truth-table) below.

---

## Repository layout (folder-level)

This is a folder-level map. Individual file enumeration is impractical
(2,169 `.py` files, 142 `.rs` files, 274 `test_*.py` files, ~134 top-level
directories). Categories below are derived from directory names and observed
contents, not from external documentation.

| Category | Locations |
|---|---|
| **Verified deterministic control core** | `qagents/ciir_crs_ric/`, `qagents/framework/`, `qagents/mvri/`, `qagents/control_geometry/`, `qagents/realworld_bridge/`, `qagents/trajectory_controller/`, `qagents/adapters/`, `qagents/reality_interface.py`, `qagents/reality_interface_v2.py`, `qagents/ciir_ric_bridge.py`, `qagents/llm_backends.py` |
| **CIIR plasma + multi-qubit analysis (Python)** | `quasim/ciir/plasma/`, `quasim/ciir/multi_qubit/` |
| **CIIR / Quasim simulation modules (mixed)** | `quasim/` (large), including `quasim/quantum/`, `quasim/hybrid_quantum/`, `quasim/qc/`, `quasim/opt/`, `quasim/terc_bridge/`, `quasim/qunimbus/`, `quasim/cli/`, `quasim/hcal/` |
| **Genomics / Merkle ledger (Rust)** | `qrVITRA/merkler-static/` |
| **Cryptographic primitives (Rust)** | `crypto/kdf/`, `crypto/pqc/`, `crypto/rng/` |
| **Networking (Rust)** | `Aethernet/`, `q-substrate/`, `qratum-rust/`, `android-emulator/`, `qratum_desktop/src-tauri/` |
| **CLIs declared in `pyproject.toml`** | `quasim/cli/`, `quasim/ciir/cli.py`, `quasim/qunimbus/cli.py`, `qnx/cli.py`, `qstack/launch.py`, `xenon/cli.py`, `qubic/visualization/cli.py`, `quasim/hcal/cli.py` |
| **Higher-level platform layers** | `qratum/`, `qratum_platform/`, `qratum_ai_platform/`, `qratum_aas/`, `qratum_asi/`, `qratum_chess/`, `qratum_desktop/`, `qratum_fullstack_server.py`, `qratum_platform.py`, `qratum_platform_legacy/` |
| **Other claimed subsystems (varying state)** | `aion/`, `qnx/`, `qnx_agi/`, `qstack/`, `qstack-superrepo/`, `qil/`, `qcore/`, `qos/`, `qsk/`, `qmp/`, `qdl/`, `qledger/`, `qnode/`, `qcampaign/`, `qconstitution/`, `qintervention/`, `qreal/`, `qscenario/`, `qtime/`, `qubic/`, `qubic-design-studio/`, `qubic-viz/`, `qubic_meta_library/`, `omnilex/`, `epistemic_heat_sink/`, `topological_observer/`, `unification/`, `verticals/`, `xenon/` |
| **Tests** | `tests/` (274 files), Rust `#[cfg(test)]` blocks in each crate |
| **Docs / manuscripts** | 130+ root `*.md` files, `manuscripts/` (LaTeX monograph), `docs/`, `docs-site/` |
| **Build / CI** | `Makefile`, `Dockerfile*`, `docker-compose*.yml`, `.github/`, `pyproject.toml`, `requirements*.txt`, `setup.py`, `setup.cfg` |

---

## Execution Truth Table

Status was determined by running `pytest`, `cargo test`, module imports, and
syntax checks against the tree on this branch.

**Test environment used for the numbers below:** Python 3.12, `pytest` 9.0,
`numpy` 2.4, `scipy` 1.17, `pyyaml` 6+, Rust stable (`cargo` from the
default rustup channel). Qiskit, PennyLane, Cirq, matplotlib, and torch
were **not** installed — these are listed as optional extras
(`[quantum]`, `[viz]`) in `pyproject.toml`.

### Python tests

| Scope | Result |
|---|---|
| Top-level `tests/test_*.py` (entire flat suite) | **1857 passed**, 78 failed, 18 errors, 4 skipped, 245 warnings |
| Whole `tests/` tree (recursive) | 2 passed, 13 skipped, **3162 errors during collection** (most are `ModuleNotFoundError` for `api.v1`, `aion.executor`, `quantum.python`, `qnx*`, etc.) |
| Verified control core (`test_ciir_crs_ric_strict`, `test_mvri`, `test_control_geometry`, `test_realworld_bridge`, `test_trajectory_controller`, `test_ric_adapters`, `test_reality_interface_controller`, `test_ric_v2`, `test_ric_ciir_bridge`, `test_ciir_crs_ric`, `test_plasma_reconnection`, `test_ciir_falsification`, `test_ciir_gaps_1_4_8`, `test_ciir_extensions`, `test_multi_qubit_controller`) | **617 passed, 0 failed** |

### Rust crates

| Crate | `cargo test` |
|---|---|
| `qrVITRA/merkler-static/` | **22 passed, 0 failed** |
| `crypto/kdf/` | **9 passed, 0 failed** |
| `Aethernet/` | **43 passed, 1 failed** (`rtf::api::tests::test_rollback_in_z1`) |
| `qratum-rust/` | **Build fails** (12 compile errors in lib test) |
| `crypto/pqc/` | **Build fails** (8 compile errors) |
| `crypto/rng/` | **Build fails** (1 compile error) |
| `q-substrate/`, `android-emulator/`, `qratum_desktop/src-tauri/` | Not exercised in this run |

### CLI entrypoints declared in `pyproject.toml`

| Entry point | Import / syntax check |
|---|---|
| `quasim-ciir` (`quasim.ciir.cli`) | OK |
| `quasim-revultra` (`quasim.cli.revultra_cli`) | OK |
| `quasim-qgh` (`quasim.cli.qgh_cli`) | OK |
| `quasim-terc-obs` (`quasim.cli.terc_obs_cli`) | OK |
| `quasim-tire` (`quasim.cli.tire_cli`) | OK |
| `qunimbus` (`quasim.qunimbus.cli`) | OK |
| `qnx` (`qnx.cli`) | OK |
| `qstack` (`qstack.launch`) | OK |
| `qubic-viz` (`qubic.visualization.cli`) | Imports require `matplotlib` (optional dep, not in base `requirements.txt`) |
| `xenon` (`xenon.cli`) | **Broken — `SyntaxError` at `xenon/cli.py:248` (`from __future__` not at top of file)** |
| `quasim-hcal` (`quasim.hcal.cli`) | **Broken — `SyntaxError` at `quasim/hcal/cli.py:790` (unterminated triple-quoted string)** |

`pyproject.toml` itself contains a **duplicate `quasim-ciir` script entry**
(lines 92 and 103), which makes some tooling refuse to load it under
`-o`/strict-key parsers.

### Module classification

| Class | Examples | Evidence |
|---|---|---|
| **Fully operational** | `qagents.ciir_crs_ric`, `qagents.framework`, `qagents.mvri`, `qagents.control_geometry`, `qagents.realworld_bridge`, `qagents.trajectory_controller`, `qagents.adapters`, `qagents.reality_interface[_v2]`, `qagents.ciir_ric_bridge`, `quasim.ciir.plasma`, `quasim.ciir.multi_qubit` (analysis pieces covered by tests), `qrVITRA/merkler-static`, `crypto/kdf`, `Aethernet` (43/44) | passing tests |
| **Partially working** | `quasim.quantum.qaoa_optimization`, `quasim.quantum.vqe_molecule`, `quasim.opt.optimizer`, `quasim.hybrid_quantum.*`, `quasim.qc.simulator` | code present and importable, but Qiskit-dependent paths fall back when `qiskit` is absent; tests fail with `ImportError: Qiskit is required for qaoa/vqe` when run without the `[quantum]` extra |
| **Stubbed / placeholder** | Files raising `NotImplementedError`, including the `quantum.python` package referenced by `tests/test_qstack_core.py` and similar | failing tests with `NotImplementedError` / `ModuleNotFoundError` |
| **Broken (syntax / import)** | `xenon/cli.py`, `quasim/hcal/cli.py`, `qratum-rust` lib, `crypto/pqc`, `crypto/rng` | compile errors / SyntaxError above |
| **Unreachable to test runner** | Modules referenced by `tests/aion/`, `tests/qnx/`, `tests/qnx_integration/`, `tests/qstack/`, `tests/integration/`, `tests/unit/test_sovereign_stack.py`, `tests/hardening/xenon_v5/`, `tests/ownai/`, `tests/distributed_cluster/`, `tests/hcal/`, `tests/chess/test_kaggle_integration.py`, `tests/benchmarks/compression/`, `tests/telemetry/test_verifier_accuracy.py`, `tests/static/`, `tests/verify/`, `tests/test_qratum_core.py`, `tests/test_qstack_core.py`, `tests/test_alignment_basic.py`, `tests/test_kernel_alignment_integration.py`, `tests/test_provenance.py`, `tests/test_hcal_cli.py` | all error during collection with `ModuleNotFoundError` for `api.v1`, `aion.executor`, `quantum.python`, etc. |

---

## Verified Capability Registry

Each capability below has **all of**: code present, importable on a clean
checkout, and at least one passing test in `tests/`.

| Capability | Implementation | Tests |
|---|---|---|
| **CIIR–CRS–RIC strict 5-module loop** (Type I–IV failures, deterministic 9-step executor, `T_C` with pre/post φ checks, explicit admissible-action enumeration, immutable trace) | `qagents/ciir_crs_ric/{ciir,crs,ric,executor,failures,demo}.py` | `tests/test_ciir_crs_ric_strict.py` (33 tests) |
| **CIIR–CRS–RIC framework variant** (constraint algebra, observer map, intent → action loop) | `qagents/framework/{ciir,crs,ric}.py`, `experiments/ciir_crs_ric_demo.py` | `tests/test_ciir_crs_ric.py` (48 tests) |
| **MVRI — Minimal Viable Reality Injector** (typed immutable State, deterministic forward model, ER/CS/SS/IC/AUD constraint gate, ΔH/ΔMI/ΔTE metrics, probe policy) | `qagents/mvri/{state,action_space,forward_model,constraint_gate,injector,loop,metrics,policy,demo}.py` | `tests/test_mvri.py` (38 tests) |
| **Control geometry** (sensitivity probes, displacement embedding, local reachability, controllability rank/SVD, action classifier, controllability-weighted policy) | `qagents/control_geometry/*.py` | `tests/test_control_geometry.py` (40 tests) |
| **Real-world bridge** (Sensor/Actuator ABCs, EMA observer, world-model sync, multi-criteria safety gate, strict 9-step control loop) | `qagents/realworld_bridge/*.py` | `tests/test_realworld_bridge.py` (45 tests) |
| **Trajectory controller** (numeric action vectors, deterministic planner/predictor/validator/executor over MVRI + RealWorldBridge) | `qagents/trajectory_controller/*.py` | `tests/test_trajectory_controller.py` |
| **Reality Interface Controller (RIC v1)** (deterministic perception→model→action, strict 4-key output, pluggable simulator/proposer) | `qagents/reality_interface.py`, `qagents/prompts/reality_interface_controller.md` | `tests/test_reality_interface_controller.py` (24 tests) |
| **RIC v2 trajectory-aware controller** (k-step rollout, seeded perturbations, anti-deadlock + entropy injection, enriched 4-key output) | `qagents/reality_interface_v2.py` | `tests/test_ric_v2.py` (18 tests) |
| **RIC ↔ CIIR bridge** (snapshot → world_state, action → learning_rate; opt-in v2 wiring) | `qagents/ciir_ric_bridge.py`, `run_ric_ciir.py` | `tests/test_ric_ciir_bridge.py` (27 tests) |
| **RIC adapters** for QRATUM (OSR/CEI/SF/HRD), CIIR (loss/violation), CRS (CRSI w/ immutable boundaries) | `qagents/adapters/{qratum,ciir,crs}.py` | `tests/test_ric_adapters.py` (23 tests) |
| **LLM backend abstraction with deterministic fallback** | `qagents/llm_backends.py` | covered by RIC/bridge tests |
| **CIIR plasma reconnection control** (2-D RMHD ψ–ω, X/O-points & current sheets, FFT shell spectra, FKR/plasmoid scaling, F1–F5 failure modes, controller stitched to engine) | `quasim/ciir/plasma/*.py`, `run_plasma_reconnection.py` | `tests/test_plasma_reconnection.py` (55 tests) |
| **CIIR multi-qubit analysis tools** (α-derivation, causal claim with falsification protocol, unified evolution) | `quasim/ciir/multi_qubit/analysis/{alpha_derivation,causal_claim,falsification,unified_evolution}.py`, `run_falsification.py`, `run_multi_qubit_ciir.py` | `tests/test_ciir_gaps_1_4_8.py`, `tests/test_ciir_extensions.py`, `tests/test_multi_qubit_controller.py`, `tests/test_ciir_falsification.py` |
| **TERC observable bridge** (`to_terc_observable_format` → `{observables, format_version="1.0", source="quasim-revultra-qgh"}`) | `quasim/terc_bridge/adapters.py` | `tests/test_terc_bridge_observables.py` |
| **Merkle ledger / biokey / ZKP / FIDO2 dual-sig** (Rust) | `qrVITRA/merkler-static/src/{biokey,zkp,fido2,...}.rs` | 22 Rust tests pass |
| **KDF crate** (Rust) | `crypto/kdf/` | 9 Rust tests pass |

The verified Python core comprises **617 passing tests** across these
modules, and Rust adds 22 (merkler-static) + 9 (kdf) + 43 (Aethernet,
1 still failing).

---

## Quantum Capability Ledger

| Claim | Implementation in tree | Status |
|---|---|---|
| **QAOA via Qiskit** | `quasim/quantum/qaoa_optimization.py` (`QuantumCircuit`, `Sampler`, `SparsePauliOp`; guarded `try/except` import; `QISKIT_AVAILABLE` flag) | ⚠ PARTIAL — code is real, but only runs when the `[quantum]` extra is installed; tests fail with `ImportError: Qiskit is required for qaoa` in a base install |
| **VQE via Qiskit** | `quasim/quantum/vqe_molecule.py`, `quasim/quantum/core.py` | ⚠ PARTIAL — same guarded pattern |
| **Generic quantum optimizer** | `quasim/opt/optimizer.py`, `quasim/hybrid_quantum/{__init__,backends}.py`, `quasim/qc/simulator.py` | ⚠ PARTIAL — `import qiskit` present; behavior under base install: import-time fallback; tests assume Qiskit |
| **CIIR multi-qubit causal-claim falsification protocol** (5-step, S1/S2/S3 secondary objectives, `derive_projector_from_fixed_point` with spectral N≤4 + power-iteration routes) | `quasim/ciir/multi_qubit/analysis/{causal_claim,falsification}.py`, `run_falsification.py` | ✓ VERIFIED (Python/NumPy; 54 + 25 tests) — but produces a **classical** numerical result (Verdict A0, SNR≈2×10¹³ for default N=3, n_steps=50) and is explicitly **a falsification protocol** of a quantum-mechanical causal claim, not a quantum-hardware execution |
| **2-D reduced MHD plasma simulation with CIIR control** | `quasim/ciir/plasma/*.py` | ✓ VERIFIED — classical PDE simulation in NumPy/SciPy; 55 tests |
| **"Quantum-resilient computing platform" / Byzantine-fault-tolerant decentralized ghost machine / on-chain governance / HotStuff BFT / libp2p gossip** | searched | ✗ CLAIMED ONLY — `qratum-rust` (the crate that would host this) **does not compile** in this clone; nothing demonstrates BFT consensus end-to-end |
| **Post-quantum cryptography (SPHINCS+, Kyber, Dilithium)** | `crypto/pqc/` | ✗ CLAIMED ONLY in this clone — `cargo build` fails with 8 errors; no passing PQC tests |
| **"Quantum-accelerated pipelines / cuQuantum / GPU quantum acceleration"** | searched | ✗ CLAIMED ONLY — no `cuquantum` import found; no GPU quantum path is exercised |
| **"Goodyear Quantum Tire Pilot" and other named industrial pilots** | only documentation references | ✗ CLAIMED ONLY — confirmed by the repo's own `_DOCUMENTATION_DISCLAIMER.md` |
| **DO-178C / regulatory certifications (HIPAA, GDPR, BIPA, 21 CFR Part 11, etc.)** | only documentation references | ✗ CLAIMED ONLY — no certification artifacts; doc files describe inspired-by practices |

**Mathematical correctness check.** The numerical kernels we ran (CIIR
deterministic loops, MVRI metrics, control-geometry SVD/cosine, plasma RMHD
update, falsification SNR routes) are written in standard NumPy/SciPy and
produce stable, deterministic outputs across reseeded runs covered by
existing tests. They are **classical** simulations and are not, and do not
claim in code to be, executions on quantum hardware.

---

## System Dependency Graph (verified subsystems only)

```
                     ┌──────────────────────────┐
                     │      LLM backends        │
                     │  qagents/llm_backends.py │
                     │ (deterministic fallback) │
                     └────────────┬─────────────┘
                                  │
                                  ▼
   ┌──────────────────────┐   ┌──────────────────────┐
   │ Reality Interface    │   │ Reality Interface v2 │
   │ qagents/             │   │ qagents/             │
   │ reality_interface.py │◄──┤ reality_interface_v2 │
   └──────────┬───────────┘   └──────────┬───────────┘
              │                          │
              ▼                          ▼
       ┌──────────────────────────────────────┐
       │   RIC ↔ CIIR bridge                  │
       │   qagents/ciir_ric_bridge.py         │
       │   run_ric_ciir.py                    │
       └──────────┬───────────────────────────┘
                  │
                  ▼
   ┌──────────────────────────┐         ┌─────────────────────┐
   │   CIIR–CRS–RIC strict    │  uses   │   RIC adapters      │
   │   qagents/ciir_crs_ric/  │◄────────┤   qagents/adapters/ │
   │   qagents/framework/     │         │   {qratum,ciir,crs} │
   └─────────┬────────────────┘         └─────────────────────┘
             │
             ├────────────► MVRI (qagents/mvri/)
             │                 │
             │                 ▼
             │           Forward model + constraint gate (ER/CS/SS/IC/AUD)
             │
             ├────────────► Control geometry (qagents/control_geometry/)
             │                 │
             │                 ▼
             │           Sensitivity, reachability, SVD, policy
             │
             ├────────────► Real-world bridge (qagents/realworld_bridge/)
             │                 │
             │                 ▼
             │           Sensors / Actuators / safety gate
             │
             └────────────► Trajectory controller (qagents/trajectory_controller/)


   ┌──────────────────────────┐    ┌──────────────────────────────┐
   │ CIIR plasma reconnection │    │ CIIR multi-qubit analysis    │
   │ quasim/ciir/plasma/      │    │ quasim/ciir/multi_qubit/     │
   │ run_plasma_reconnection  │    │ run_falsification.py         │
   │ (RMHD + control + spec)  │    │ run_multi_qubit_ciir.py      │
   └──────────────────────────┘    └──────────────────────────────┘

   ┌──────────────────────────┐    ┌──────────────────────────────┐
   │ Merkle ledger + biokey   │    │ KDF (Rust)                   │
   │ qrVITRA/merkler-static   │    │ crypto/kdf                   │
   │ (Rust: biokey, zkp,      │    │                              │
   │  fido2, merkle)          │    │                              │
   └──────────────────────────┘    └──────────────────────────────┘
```

Modules outside this graph (`qratum*`, `qratum_platform*`, `qratum_ai_platform`,
`qratum_chess`, `qratum_desktop`, `qstack*`, `qnx*`, `qnx_agi`, `aion`,
`epistemic_heat_sink`, `topological_observer`, `xenon`, `qubic*`,
`omnilex`, `verticals`, `unification`, etc.) are present in the tree but
were **not verified** as part of the executable graph in this audit. Their
relationship to the verified core in code is mostly through `qagents/`
adapters and `quasim/` shared modules; their own end-to-end behavior is
either untested in this clone or blocked by import failures (see Gap report
below).

---

## Gap and integrity report (severity-ranked)

### Critical (blocks declared functionality)
1. **`xenon/cli.py` will not parse** — `SyntaxError` at line 248 (`from __future__ import annotations` placed after other code). The `xenon` console-script entry in `pyproject.toml` therefore cannot load.
2. **`quasim/hcal/cli.py` will not parse** — `SyntaxError` at line 790 (unterminated triple-quoted docstring). The `quasim-hcal` console-script entry cannot load.
3. **`qratum-rust` does not compile** (12 errors). README claims around BFT consensus, libp2p gossip, on-chain governance, slashing, and ZK state transitions reference this crate; none are demonstrable here.
4. **`crypto/pqc` does not compile** (8 errors); **`crypto/rng` does not compile** (1 error). Post-quantum-crypto claims (SPHINCS+, Kyber, Dilithium) are not demonstrable.
5. **`pyproject.toml` has a duplicate `quasim-ciir` script entry** (lines 92 & 103), causing strict TOML parsers to error (e.g., `pytest -o`).
6. **Massive collection failures in `tests/`** — 3162 test items error at import time when running the whole tree, due to missing modules (`api.v1`, `aion.executor`, `quantum.python`, several `qnx*`, `quantum.python`). The flat `tests/test_*.py` set still has 1857 passing, but ~78 named failures and 18 errors persist.

### High (claims-vs-code divergence)
7. **README marketing claims removed** in this rewrite — original README claimed Byzantine-fault-tolerant consensus, "decentralized ghost machine", quantum-resilience, 14 vertical domains, named industry pilots (Goodyear), and DO-178C alignment. None of these are observable in code or tests. The repository's own `_DOCUMENTATION_DISCLAIMER.md` and `QUANTUM_CAPABILITY_AUDIT.md` independently corroborate that.
8. **"Quantum" terminology overuse**. The genuine quantum-library code (`quasim/quantum/qaoa_optimization.py`, `vqe_molecule.py`, `core.py`) **does** use Qiskit primitives, but: (a) Qiskit is not installed by default; (b) tests for these paths fail with `ImportError` in a base install; (c) all "quantum-accelerated", "quantum-resilient", and "cuQuantum" claims in earlier docs are unsupported by code.
9. **"Quantum hardware tier", "SpaceX-NASA CI workflows", "hardware tier bundles"** — referenced only in zip files at the repo root (`quasim-hardware-tier.zip`, `quasim-executive-brief.zip`); no live, importable hardware-tier module is present.

### Medium (housekeeping / hygiene)
10. **`Aethernet` has 1 failing test** (`rtf::api::tests::test_rollback_in_z1`); 110 compiler warnings.
11. **`tests/test_merkle_chain_determinism`** asserts an exact hash that does not match observed output — the determinism contract for this chain is currently broken or test fixture is stale.
12. **Many top-level `.md` files are partially or fully outdated** (the repo even ships `_DOCUMENTATION_DISCLAIMER.md` listing the affected docs). Pruning or clearly marking these would improve discoverability.
13. **Heavy `__pycache__` and large binary artifacts** committed (e.g., `*.zip`, `*.pdf`, `*.png`, `run_*.zip` artifacts). These should normally be in `.gitignore`.
14. **`pyproject.toml` has a duplicate package configuration** (see #5).

### Low
15. **`quasim` → `qratum` rename** is partially complete: a deprecation shim warns on import (`tests/conftest.py:19`), but the migration path described in `MIGRATION.md` is not enforced and most code still imports `quasim.*`.

---

## Installation

The minimum needed to run the verified core:

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt     # numpy, pyyaml, click, scipy ...
pip install pytest                   # for tests
```

Optional extras (these correspond to declared `[project.optional-dependencies]`
groups and **are required** for the partially-implemented quantum and viz
paths):

```bash
pip install -e .[quantum]           # qiskit, qiskit-aer, pennylane, pyscf
pip install -e .[viz]               # matplotlib, imageio, plotly, ...
pip install -e .[dev]               # pytest, ruff, mypy, bandit, pandas, ...
```

For the Rust components that build:

```bash
cd qrVITRA/merkler-static && cargo build && cargo test    # 22 tests pass
cd Aethernet                && cargo build && cargo test  # 43/44 tests pass
cd crypto/kdf               && cargo build && cargo test  # 9 tests pass
```

The following Rust crates are known not to build on this branch:
`qratum-rust`, `crypto/pqc`, `crypto/rng`.

---

## How to run what works

```bash
# 1. Verified Python control core (617 tests, deterministic)
PYTHONPATH=. python -m pytest \
    tests/test_ciir_crs_ric_strict.py \
    tests/test_ciir_crs_ric.py \
    tests/test_mvri.py \
    tests/test_control_geometry.py \
    tests/test_realworld_bridge.py \
    tests/test_trajectory_controller.py \
    tests/test_ric_adapters.py \
    tests/test_reality_interface_controller.py \
    tests/test_ric_v2.py \
    tests/test_ric_ciir_bridge.py \
    tests/test_plasma_reconnection.py \
    tests/test_ciir_falsification.py \
    tests/test_ciir_gaps_1_4_8.py \
    tests/test_ciir_extensions.py \
    tests/test_multi_qubit_controller.py \
    -c /dev/null --override-ini="addopts=" -q

# 2. Demos for individual subsystems
PYTHONPATH=. python -m qagents.ciir_crs_ric.demo
PYTHONPATH=. python -m qagents.mvri.demo
PYTHONPATH=. python -m qagents.realworld_bridge.demo
PYTHONPATH=. python -m qagents.trajectory_controller.demo
PYTHONPATH=. python run_ric_ciir.py
PYTHONPATH=. python run_plasma_reconnection.py
PYTHONPATH=. python run_multi_qubit_ciir.py
PYTHONPATH=. python run_falsification.py --N 3 --n-steps 50

# 3. Rust components that build
( cd qrVITRA/merkler-static && cargo test )
( cd Aethernet              && cargo test )
( cd crypto/kdf             && cargo test )
```

The full top-level Python suite (`pytest tests/test_*.py`) currently reports
**1857 passed / 78 failed / 18 errors / 4 skipped / 245 warnings**. The
failures are dominated by missing optional dependencies (`qiskit`,
`matplotlib`) and a handful of legitimately broken tests (see Gap report).

---

## Limitations (explicit)

- **No quantum hardware execution.** All "quantum" code in this repository
  is either (a) classical NumPy/SciPy with quantum-mechanical motivation, or
  (b) Qiskit-based circuits intended to run on a Qiskit `Sampler`/`Aer`
  simulator — never on real QPUs in any path that this repository
  exercises.
- **No working consensus / decentralized network.** `qratum-rust` does not
  compile in this clone; there is no end-to-end demonstration of BFT
  consensus, libp2p gossip, or on-chain governance.
- **No certified compliance.** Despite extensive documentation language
  around HIPAA / GDPR Article 9 / BIPA / 21 CFR Part 11 / DO-178C, no
  certification evidence is present in the code or build artifacts.
- **No working post-quantum crypto stack** in this clone — `crypto/pqc` and
  `crypto/rng` fail to compile.
- **Two declared CLIs are syntactically broken** (`xenon`, `quasim-hcal`) and
  cannot start.
- **Whole-tree test execution is broken** by import errors in many
  subsystems (`api.v1`, `aion.executor`, `quantum.python`, several `qnx*`).
  Only the curated subset above runs cleanly.
- **`pyproject.toml` has a duplicate `quasim-ciir` script entry**, which
  some tools refuse to parse.

---

## Research artifacts

- `manuscripts/ciir_monograph/` — LaTeX monograph (Ch. 1–22) accompanying
  the CIIR–CRS–RIC framework, falsification protocol (Ch. 22.4), and the
  unified extensions registry (Ch. 22.5). The monograph references modules
  enumerated in the [Verified Capability Registry](#verified-capability-registry).
- `main_revtex*.tex` and accompanying figures (`critical_scaling.*`,
  `entropy_scaling.*`, `quantum_phase_diagram.*`,
  `transport_characterization.*`).
- Numerous `*.md` reports under the repository root document earlier
  experiments. Treat these as historical unless they are corroborated by a
  passing test in `tests/`.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Author

Robert Ringler (independent researcher).

## Citation

If you use the verified CIIR–CRS–RIC / MVRI / control-geometry /
real-world-bridge / plasma-reconnection code in academic work, please cite
the CIIR monograph in `manuscripts/ciir_monograph/` and reference this
repository at the commit you used.
