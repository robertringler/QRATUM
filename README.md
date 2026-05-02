# 1. QRATUM

**A modular, deterministic, traceable execution framework for constraint-governed multi-agent computation, with a verifiable Merkle-anchored execution ledger and a plugin-based subsystem architecture.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-research%2Fbeta-yellow.svg)](#12-roadmap)

---

## 2. System Overview

QRATUM is a Python-first computational framework that composes three properties most multi-agent systems treat as optional:

1. **Constraint-governed execution** — every state transition is gated by an explicit constraint algebra; transitions that would violate an invariant are rejected, not retried silently.
2. **Bit-exact determinism** — given identical inputs and configuration, every run produces byte-identical outputs and an identical execution trace. No wall-clock timestamps, no global RNG, no dictionary-order dependence in canonical encoders.
3. **Cryptographic traceability** — every accepted step is appended to a Merkle-anchored ledger keyed by canonical-JSON SHA-256 over the step payload, so a third party can replay the run and verify each entry independently.

It exists because typical agent stacks freely interleave LLM calls, stochastic samplers, and side-effecting tools without a verification layer. That is acceptable for exploratory work but unacceptable when the same pipeline must be (a) auditable after the fact, (b) reproducible across machines, and (c) safe to extend with new agents or new physics modules without breaking guarantees the rest of the system relies on. QRATUM is the substrate that makes those three things hold simultaneously.

The system targets a research-and-production hybrid envelope: rigorous enough that a falsification protocol can be applied to its claims, modular enough that experimental subsystems (plasma reconnection control, biokey-derived signing, trajectory controllers) live as plugins on the same spine.

---

## 3. Core Design Principles

| Invariant | What it means concretely | Where it is enforced |
|---|---|---|
| **Determinism** | No `time.time()`, no global RNG, no unordered iteration in any path that influences output. All randomness is seeded and explicit. | `qagents/control_geometry/`, `qagents/mvri/`, `qratum_framework/trace.py` (canonical-JSON, sorted keys, metadata excluded from hashes). |
| **Traceability** | Every accepted step yields a `TraceEntry` with the prior hash, payload hash, and verdict. The chain is appendable but tamper-evident. | `qratum_framework/trace.py` (`UnifiedTraceEntry`, `MerkleLedger`). |
| **Reproducibility** | Two runs with identical config and seed produce identical ledgers and identical hash chains. CI compares hashes, not floats. | Profiles in `qratum_framework/config.py` (`quick`, `medium`, `strong`, `full_fast`, `full_report`). |
| **Modular isolation** | Subsystems communicate through typed adapters; a misbehaving plugin cannot corrupt the spine, only its own trace entries. | `qagents/adapters/`, `OperatorBackend` Protocol in `qratum_framework/operator.py`. |
| **Execution safety** | Failure modes are named (Type I–IV in CIIR–CRS–RIC), so refusal to act is a first-class outcome, not an exception leak. The executor never raises into user code. | `qagents/ciir_crs_ric/failures.py`, `qagents/ciir_crs_ric/executor.py` (9-step `run_step`, never raises). |
| **Verifiable computation** | Claims about behavior are stated as falsifiable verdicts (A / A0 / B) and screened by a published protocol. | `qratum_framework/falsifier.py`, `quasim/ciir/multi_qubit/analysis/falsification.py`. |

These invariants are not aspirational. They are the gating conditions for the test suites under `tests/test_qratum_framework.py`, `tests/test_ciir_crs_ric_strict.py`, `tests/test_mvri.py`, `tests/test_realworld_bridge.py`, `tests/test_control_geometry.py`, and `tests/test_ric_*.py`.

---

## 4. Architecture Breakdown

```
                                qratum_framework/
                            ┌────────────────────────┐
                            │  Operator (spine)      │
                            │  ├─ OperatorBackend ◄──┼─── pluggable
                            │  ├─ MerkleLedger       │
                            │  ├─ Falsifier (A/A0/B) │
                            │  └─ Health / Config    │
                            └──────────┬─────────────┘
                                       │ run_step
                       ┌───────────────┼───────────────────────────┐
                       ▼               ▼                           ▼
              ┌────────────────┐  ┌────────────────┐      ┌────────────────────┐
              │  CIIR–CRS–RIC  │  │ Reality        │      │ Domain plugins     │
              │  control core  │  │ Interface      │      │ (plasma, VITRA-E0, │
              │  (qagents/...) │  │ Controller     │      │  trajectory, ...)  │
              └───────┬────────┘  └───────┬────────┘      └─────────┬──────────┘
                      │                   │                         │
                      ▼                   ▼                         ▼
              ┌────────────────┐  ┌────────────────┐      ┌────────────────────┐
              │ MVRI           │  │ RIC adapters   │      │ Real-world bridge  │
              │ (state/inj.)   │  │ (qratum/ciir/  │      │ (sensors/actuators)│
              │                │  │  crs)          │      │                    │
              └────────────────┘  └────────────────┘      └────────────────────┘
                                                                    │
                                                                    ▼
                                                           ┌──────────────────┐
                                                           │  Trace ledger    │
                                                           │  (Merkle chain)  │
                                                           └──────────────────┘
```

### 3.1 Control layer — CIIR–CRS–RIC

The deterministic execution core. Three coupled modules:

- **CIIR** (`qagents/ciir_crs_ric/ciir.py`, `qagents/framework/ciir.py`) — Constraint-governed Iterated Inference. Defines `State`, `Constraint`, the satisfaction predicate `phi`, the invariant predicate `Inv`, and the observer map `Omega` (which redacts internal-only fields).
- **CRS** (`qagents/ciir_crs_ric/crs.py`, `qagents/framework/crs.py`) — Constrained Reactive System. Pure transition function `T` plus its constrained form `T_C` with explicit pre- and post-`phi` checks. Action sets are enumerated, never inferred.
- **RIC** (`qagents/ciir_crs_ric/ric.py`, `qagents/reality_interface.py`, `qagents/reality_interface_v2.py`) — Reality Interface Controller. A perception → model → action loop with a strict 4-key output contract (`intent_interpretation`, `selected_action`, `predicted_outcome`, `fallback_plan`). RIC v2 adds k-step rollouts, seeded bounded perturbations, anti-deadlock heuristics.

The `Operator` (`qratum_framework/operator.py`) wraps these via the `OperatorBackend` Protocol; the default `StrictCIIRBackend` delegates to `qagents.ciir_crs_ric.executor.run_step`, a 9-step pipeline that never raises.

### 3.2 Orchestration layer — multi-agent coordination

- **RIC adapters** (`qagents/adapters/{qratum,ciir,crs}.py`) translate between the controller's abstract action space and three concrete domain encodings (OSR/CEI/SF/HRD; loss/violation; CRSI). Each adapter ships a `Simulator`, a `Proposer`, and a `make_<sys>_controller` builder.
- **CIIR↔RIC↔LLM bridge** (`qagents/ciir_ric_bridge.py`, `qagents/llm_backends.py`) maps controller snapshots to world-state inputs and action verdicts to learning-rate signals; LLM backends (OpenAI, Anthropic, Gemini, Local, plus a `DeterministicLLM`) fall back deterministically when no SDK or key is present.
- **Trajectory controller** (`qagents/trajectory_controller/`) plans, predicts, validates, and executes numeric action vectors `[type_code, target_index, magnitude, seed]` over the real-world bridge and MVRI.

### 3.3 State / ledger system

- **Trace** (`qratum_framework/trace.py`) — `UnifiedTraceEntry` + `MerkleLedger`. Hashing uses canonical JSON (sorted keys, no whitespace) over a payload that explicitly excludes the `metadata` field, so cosmetic edits never alter the chain. `iter_with_hashes` provides a public audit cursor.
- **MVRI** (`qagents/mvri/`) — Minimal Viable Reality Injector. Typed immutable `State`, entropy `H`, invariant `Inv`, the action gate `validate_action`, and the constraint gate (`ER/CS/SS/IC/AUD` predicates) feed every accepted injection into the loop trace.
- **Real-world bridge** (`qagents/realworld_bridge/`) — typed `Sensor`/`Actuator` ABCs, an EMA-based `StateObserver` with a canonical nodes↔edges bijection, a four-stage safety gate (`MAG → TGT → ROC → MVR`), and a strict 9-step `ControlLoop`.

### 3.4 Plugin / module system

A subsystem becomes a QRATUM plugin by satisfying three contracts:

1. Implement `OperatorBackend` (one method, `run_step`).
2. Emit `TraceEntry`-compatible dictionaries.
3. Register failure modes (subclasses of the named exceptions in `qagents/ciir_crs_ric/failures.py`).

Existing plugins demonstrate the pattern:

| Plugin | Path | Domain |
|---|---|---|
| Plasma reconnection control | `quasim/ciir/plasma/` | 2D RMHD ψ–ω, X/O-points, plasmoid scaling, controller |
| CIIR multi-qubit falsification | `quasim/ciir/multi_qubit/analysis/` | projector derivation, A/A0/B verdicts |
| VITRA-E0 sovereign genomics | `qrVITRA/merkler-static/` (Rust) | biokey, FIDO2 dual-signature, ZKP |
| Control geometry | `qagents/control_geometry/` | sensitivity, reachability, controllability rank |

### 3.5 CLI + API

- **CLI**: `qratum` (entry point declared in `pyproject.toml` as `qratum = qratum_framework.cli:main`). Subcommands: `run`, `simulate`, `ledger`, `falsify`, `verify`.
- **Programmatic API**: `from qratum_framework import Operator, MerkleLedger, FalsificationVerdict, PROFILES, check_health, check_readiness`. Adapter builders are re-exported from `qagents.adapters` and `qagents`.

---

## 5. Data / Execution Flow

```
INPUT (CLI args / API call / config profile)
   │
   ▼
[ 1 ] Profile resolution           qratum_framework/config.py → PROFILES[name]
[ 2 ] Health / readiness check     qratum_framework/health.py
[ 3 ] Operator construction        Operator(backend=StrictCIIRBackend(), ledger=MerkleLedger())
   │
   ▼
[ 4 ] Per-step loop (Operator.run_step):
        ├── 4a parse_intent          (RIC)
        ├── 4b build Observation     (RIC; not raw State — Omega-filtered)
        ├── 4c select_action         (RIC + adapter Proposer)
        ├── 4d enumerate admissible  (CRS)
        ├── 4e pre-phi check         (CIIR Constraint algebra)
        ├── 4f apply T_C             (CRS: pure transition under constraint)
        ├── 4g post-phi + Inv check  (CIIR)
        ├── 4h validate_action gate  (MVRI / safety_gate)
        └── 4i emit TraceEntry       (verdict ∈ {accept, reject(Type I–IV), hold})
   │
   ▼
[ 5 ] Ledger append                  MerkleLedger.append(entry)
        canonical_json(payload) → SHA-256 → linked to prev_hash
[ 6 ] Falsification screen (opt.)    Falsifier → {A, A0, B}
   │
   ▼
OUTPUT
   ├── ledger.jsonl  (append-only, replayable)
   ├── final state   (Omega-filtered)
   └── verdict       (FalsificationVerdict)
```

Inputs never bypass the constraint algebra. Failures short-circuit at the earliest gate that detects them, are typed (`Type I` precondition / `Type II` postcondition / `Type III` invariant / `Type IV` admissibility), and are recorded in the ledger with the same hashing discipline as accepted steps. A rejected step is part of the audit trail, not noise.

---

## 6. Key Features

- **Deterministic execution** — canonical-JSON hashing, seeded RNG, no global state; bit-exact reruns.
- **Multi-agent coordination** — RIC adapters bridge a single controller to multiple domain encodings (qratum / CIIR / CRS) without state aliasing.
- **Audit trail** — every step (accepted *and* rejected) is committed to a Merkle-linked ledger; tamper detection is O(1) per entry on replay.
- **Runtime validation** — four-stage safety gate (`MAG → TGT → ROC → MVR`), constraint algebra pre/post-checks, observer-state invariants.
- **Extensibility** — plugin contract is one Protocol method (`run_step`) plus typed trace entries; no central registry to fight.
- **Falsification protocol** — verdicts A / A0 / B with a published 5-step screen (baseline / null / signal / artifacts / verdict).
- **Fault isolation** — failures are typed, never silent; the operator catches and records them, the loop never raises into caller code.
- **Health and readiness probes** — `check_health()` and `check_readiness()` for orchestration layers.
- **CLI + programmatic parity** — every CLI subcommand is a thin shell over a public API call.
- **Profiles** — `quick`, `medium`, `strong`, `full_fast`, `full_report` cover the latency / coverage trade space without ad-hoc flags.

---

## 7. Installation

### Prerequisites

- Python **3.10+**
- `pip` ≥ 23
- (Optional) Rust **1.75+** with `cargo` for the `qrVITRA/merkler-static/` and `Aethernet/` crates.
- (Optional) NumPy / SciPy for the plasma reconnection plugin (`quasim/ciir/plasma/`).

### Install

```bash
git clone https://github.com/robertringler/QRATUM.git
cd QRATUM

python -m venv .venv
source .venv/bin/activate      # POSIX
# .venv\Scripts\activate       # Windows

pip install -e .
```

This exposes the `qratum` console script and the `qratum_framework` and `qagents` packages.

### Optional Rust components

```bash
# Genomics / biokey + ZKP + FIDO2 dual-signature
cd qrVITRA/merkler-static && ./build.sh && cargo test

# Networking primitives
cd Aethernet && cargo test
```

### Verifying the install

```bash
PYTHONPATH=. python -m pytest tests/test_qratum_framework.py --override-ini="addopts=" -q
```

A clean run reports the unified-spine test count and exits with code 0.

---

## 8. Usage Examples

### 7.1 CLI

```bash
# Run a deterministic controller pass under a named profile
qratum run --profile medium --seed 42 --ledger out/ledger.jsonl

# Replay and verify a previously emitted ledger
qratum verify --ledger out/ledger.jsonl

# Apply the falsification screen to the resulting trace
qratum falsify --ledger out/ledger.jsonl

# Pure simulation (no domain plugin, useful for CI smoke)
qratum simulate --steps 100 --seed 42

# Inspect ledger entries
qratum ledger --ledger out/ledger.jsonl --head 10
```

### 7.2 Programmatic API

```python
from qratum_framework import (
    Operator, MerkleLedger, PROFILES, FalsificationVerdict,
    check_health, check_readiness,
)
from qratum_framework.operator import StrictCIIRBackend

assert check_health().ok and check_readiness().ok

ledger = MerkleLedger()
op = Operator(backend=StrictCIIRBackend(), ledger=ledger, profile=PROFILES["medium"])

for step in range(100):
    entry = op.run_step(input_payload={"step": step, "seed": 42})
    if entry.verdict.kind == "reject":
        # Typed failure modes: Type I–IV
        print("rejected:", entry.verdict.failure_type, entry.verdict.reason)

# Replay-style audit
for prev_hash, payload_hash, entry in ledger.iter_with_hashes():
    assert entry.prev_hash == prev_hash

# Falsification screen
verdict: FalsificationVerdict = op.falsify(ledger)
assert verdict in {FalsificationVerdict.A, FalsificationVerdict.A0, FalsificationVerdict.B}
```

### 7.3 Custom backend (plugin)

```python
from qratum_framework.operator import Operator, OperatorBackend
from qratum_framework.trace import UnifiedTraceEntry, MerkleLedger

class MyBackend(OperatorBackend):
    def run_step(self, state, action, *, profile):
        # ... pure, deterministic, no global RNG ...
        return UnifiedTraceEntry(
            step=state.step + 1,
            payload={"action": action, "result": ...},
            metadata={"backend": "MyBackend"},  # excluded from hashing
        )

op = Operator(backend=MyBackend(), ledger=MerkleLedger())
```

---

## 9. Configuration Model

### 8.1 Hierarchy

Resolution order (highest precedence first):

1. **CLI flags** — `--profile`, `--seed`, `--ledger`, `--steps`, etc.
2. **Programmatic overrides** — keyword arguments to `Operator(...)`.
3. **Environment variables** — see §9.3.
4. **Profile** — one of the named entries in `qratum_framework/config.py::PROFILES`.
5. **Built-in defaults** — conservative; `quick` profile equivalent.

### 8.2 Profiles

| Profile | Intent |
|---|---|
| `quick` | Smoke / CI; minimal step budget. |
| `medium` | Default for development runs. |
| `strong` | Tighter safety-gate thresholds; longer rollouts. |
| `full_fast` | Production sweep; coverage over latency. |
| `full_report` | Coverage + falsification screen + full ledger emission. |

Profiles are *data*, not code paths. Adding a profile means adding a dict entry, not a branch.

### 8.3 Environment variables (conceptual)

| Variable | Purpose |
|---|---|
| `QRATUM_PROFILE` | Default profile when `--profile` is absent. |
| `QRATUM_LEDGER` | Default ledger path. |
| `QRATUM_SEED` | Default seed; required for reproducibility. |
| `QRATUM_LLM_BACKEND` | One of `deterministic`, `openai`, `anthropic`, `gemini`, `local`. Falls back to `deterministic` when SDK or key is absent. |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY` | Credentials for non-deterministic backends; *never required for default operation*. |

Missing credentials must not cause hard failures or process exits — the LLM layer is required to fall back deterministically (see `qagents/llm_backends.py`).

---

## 10. Development Guide

### 9.1 Adding a module

1. Place the module under an existing namespace (`qagents/<your_module>/` for control-layer code, `quasim/<your_module>/` for simulation-domain code).
2. Define typed inputs and outputs (dataclasses or TypedDict).
3. Forbid global mutable state and global RNG. Inject a `random.Random(seed)` if you need stochasticity.
4. Add a dedicated test file under `tests/test_<your_module>.py`. Follow the pattern of `tests/test_mvri.py` or `tests/test_control_geometry.py`.
5. Re-export your public surface from the appropriate `__init__.py`.

### 9.2 Registering an agent / adapter

1. Implement a `Simulator`, a `Proposer`, and a `make_<sys>_controller` builder following the templates in `qagents/adapters/{qratum,ciir,crs}.py`.
2. Re-export from `qagents/adapters/__init__.py`.
3. Add adapter tests to `tests/test_ric_adapters.py`.

### 9.3 Extending execution rules

To add a new constraint or invariant to CIIR–CRS–RIC:

1. Define the predicate in `qagents/ciir_crs_ric/ciir.py` (constraint) or `qagents/framework/ciir.py` (algebra).
2. Wire it into `phi` (precondition), `Inv` (invariant), or both.
3. If the violation is structurally new, declare a new failure type in `qagents/ciir_crs_ric/failures.py` (Type I–IV taxonomy).
4. The executor's 9-step `run_step` will pick it up via the existing `OperatorBackend` contract — do not modify the executor itself for domain rules.

### 9.4 Test commands (verified)

```bash
PYTHONPATH=. python -m pytest tests/test_qratum_framework.py        --override-ini="addopts=" -q
PYTHONPATH=. python -m pytest tests/test_ciir_crs_ric_strict.py     --override-ini="addopts=" -q
PYTHONPATH=. python -m pytest tests/test_mvri.py                    --override-ini="addopts=" -q
PYTHONPATH=. python -m pytest tests/test_realworld_bridge.py        --override-ini="addopts=" -q
PYTHONPATH=. python -m pytest tests/test_control_geometry.py        --override-ini="addopts=" -q
PYTHONPATH=. python -m pytest tests/test_ric_v2.py                  --override-ini="addopts=" -q
PYTHONPATH=. python -m pytest tests/test_ric_ciir_bridge.py         --override-ini="addopts=" -q
PYTHONPATH=. python -m pytest tests/test_plasma_reconnection.py     --override-ini="addopts=" -q
PYTHONPATH=. python -m pytest tests/test_ciir_falsification.py      --override-ini="addopts=" -q
```

---

## 11. Safety, Determinism & Verification Guarantees

### 10.1 Execution constraints

- The constraint algebra (`phi`, `Inv`) is checked **twice** per step: pre-transition and post-transition.
- The safety gate runs four sequential checks (`MAG`, `TGT`, `ROC`, `MVR`). All four are evaluated before a verdict is emitted, so failure reports are complete, not first-hit.
- The `ControlLoop` advances `model_state` only on *accepted* injection; a rejected step never mutates the state used by the next step's planner.
- `SensorExhausted` ends a run early but cleanly; the ledger remains valid.

### 10.2 Reproducibility enforcement

- Hashing is over **canonical JSON**: keys sorted, no whitespace, `metadata` field excluded from the hash domain.
- All RNG is seeded; no module reads `time.time()` or `os.urandom()` on the determinism path.
- Profiles are pure data; flipping a profile cannot inject non-determinism via a code path.
- CI compares ledger hashes across machines, not floating-point outputs. A diverging hash is a hard failure.

### 10.3 Invalid-state handling

- Failure modes are named: **Type I** (precondition violation), **Type II** (postcondition violation), **Type III** (invariant violation), **Type IV** (admissibility violation).
- The executor's 9-step `run_step` never raises into caller code; it returns a `TraceEntry` whose verdict carries the failure type.
- The ledger records rejections with the same hashing discipline as acceptances; an attacker cannot hide a rejected step by reordering.
- The falsifier reduces the ledger to one of `{A, A0, B}`:
  - **A** — claim verified within tolerance.
  - **A0** — claim verified but mechanism differs from declared (e.g. amplitude damping rather than causal geometry).
  - **B** — claim falsified.

### 10.4 Cryptographic verification surface

- Per-entry hash: SHA-256 over canonical-JSON payload.
- Chain link: each entry stores the prior `payload_hash` as `prev_hash`; `MerkleLedger.iter_with_hashes` is the public audit cursor.
- Optional Rust-side biokey signing (`qrVITRA/merkler-static/`) provides ephemeral SNP-derived keys, FIDO2 dual-signature, and ZKP attestation for sovereign-genomics use cases.

---

## 12. Roadmap

The following are tracked extension directions; none are required for current operation.

- **Spine hardening** — strict mypy on `qratum_framework/`, atomic ledger writes with `fsync`, ledger-load verification, size caps on trace metadata.
- **Distributed ledger mode** — multi-writer ledger with deterministic merge, suitable for federated agent runs.
- **Hardware falsification** — execute the 5-step falsification protocol against physical quantum hardware (`run_falsification.py` is the simulator entry point today).
- **Plugin SDK** — package the `OperatorBackend` Protocol, trace dataclasses, and failure taxonomy as a stand-alone wheel so out-of-tree plugins do not need to vendor `qagents/`.
- **Plasma reconnection (extended)** — controller-aware mesh refinement for `quasim/ciir/plasma/`, FKR/plasmoid scaling at higher Lundquist numbers.
- **Trajectory controller (k-step)** — extend RIC v2's k-step rollout into the trajectory controller for end-to-end horizon-aware planning over the real-world bridge.
- **Formal verification** — discharge invariants `phi` and `Inv` to an SMT backend on a bounded fragment of the action space.

Research directions (no committed timeline): falsification of causal-geometry claims at larger N, biokey-bound ledgers for genomics-grade reproducibility, and unification of MVRI's entropy/MI/TE metrics with CIIR's stability inequality.
