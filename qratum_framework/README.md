# `qratum_framework` — the unified operational spine

This package is the single integration surface described in the
*QRATUM Fully-Wired, Operational Framework* plan. It does **not** replace
any existing subsystem; it projects the existing CIIR–CRS–RIC strict
executor, MVRI, control-geometry, real-world bridge, and domain
simulators onto **one coherent control plane**:

| Concern                     | Module                                  |
|-----------------------------|-----------------------------------------|
| Unified TraceEntry schema   | `qratum_framework.trace`                |
| Merkle-chained ledger       | `qratum_framework.trace.MerkleLedger`   |
| Canonical end-to-end loop   | `qratum_framework.operator.Operator`    |
| Adapter contract (backends) | `qratum_framework.operator.OperatorBackend` |
| Falsification verdicts      | `qratum_framework.falsifier`            |
| Pipeline profile loader     | `qratum_framework.config`               |
| Health + readiness contract | `qratum_framework.health`               |
| Operator CLI (`qratum`)     | `qratum_framework.cli`                  |

## Quick start

```bash
# Run the canonical demo through the strict CIIR–CRS–RIC executor.
qratum run

# Use a profile and emit a JSON manifest.
qratum run --profile medium --json

# Health + readiness gate (non-zero exit on failure).
qratum verify

# Run the reference falsifier and see the verdict (A / A0 / B).
qratum falsify --json

# Verify a previously persisted JSONL ledger.
qratum ledger ./run.jsonl --verify-only
```

## The canonical request path (§3 of the plan)

Every workload — quantum, plasma, genomic, consensus, agentic — flows
through a single pipeline:

1. Pilot emits an **Intent**.
2. **RIC** parses it (strict 4-key output).
3. **CIIR Φ-gate** evaluates the candidate action.
4. **CRS T_C** applies the pure transition with pre/post Φ checks; named
   Type I–IV failure modes surface, never swallowed.
5. **Control geometry** scores the action (delegated to the backend).
6. **MVRI** performs the bounded reality injection (delegated to the
   backend).
7. **Domain executor** runs the simulated or physical step.
8. **Observer** updates state via EMA; **safety_gate** revalidates.
9. **Trace** is appended to the **Merkle ledger** and signed.

The default backend, `StrictCIIRBackend`, wraps
`qagents.ciir_crs_ric.executor.run_step` — the canonical 9-step strict
loop. Any domain backend that implements the `OperatorBackend` Protocol
plugs in transparently; the spine is otherwise backend-agnostic.

## Determinism

* No global RNG is touched anywhere in this package.
* The Merkle chain hashes a *canonical* JSON payload (sorted keys,
  compact separators, no wall-clock fields), so two runs with identical
  inputs produce byte-identical chains.
* The `metadata` field is excluded from canonical hashing precisely so
  wall-clock or host-specific provenance cannot perturb reproducibility.

## Profiles

Built-in profiles (`quick`, `medium`, `strong`, `full_fast`,
`full_report`) are defined in `qratum_framework.config.PROFILES` and can
be overridden by a YAML file via `load_profile(yaml_path=...)`. PyYAML is
optional; if absent, the YAML overlay is silently skipped.

## Falsification (§4 of the plan)

Every domain falsifier returns a `FalsificationVerdict` with a verdict
in `{A, A0, B}`, an SNR, and an overlap-with-ground scalar — matching
the existing multi-qubit falsifier's vocabulary. The framework provides
the **contract**, not the harness; real domain falsifiers (multi-qubit,
plasma, VITRA, …) live in their own packages and register through the
`Falsifier` Protocol.

## What this package is *not*

* It is **not** a re-implementation of any existing subsystem.
* It does **not** deprecate `qagents.ciir_crs_ric` — it consumes it.
* It does **not** ship a web server. The health/readiness contract is
  pure data; an HTTP wrapper is the responsibility of the deploying
  service.

See the parent plan for the rest of the integration roadmap (single
Dockerfile, single dashboard, schema unification across MVRI / framework
/ ciir_crs_ric / domain simulators, etc).
