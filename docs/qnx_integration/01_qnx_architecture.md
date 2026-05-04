# QNX substrate architecture

The QNX substrate is an additive orchestration layer that lets QuASIM route
simulations to multiple execution engines while keeping a unified interface and
result shape. It does not change existing public APIs and can be adopted
incrementally.

## Layering overview

```
QNX SUBSTRATE
  ├─ QuASIM (modern)
  ├─ QuASIM v1.2.0 (legacy)
  └─ QVR Windows runtime
```

The substrate chooses a backend at runtime and returns a
`SubstrateResult` that records which backend ran along with hashing,
execution timing, and (optionally) carbon estimates.

## Key types

* `qnx.types.SimulationConfig` — configuration for a simulation run; includes
  scenario identifier, timesteps, seed, security level, and backend selection.
* `qnx.types.SubstrateResult` — normalized result structure containing backend
  name, simulation hash, elapsed time, carbon estimate, and raw backend output.
* `qnx.backends.get_backend_registry()` — discovers available backends and makes
  them selectable via the `backend` field on `SimulationConfig`.

## Backends provided

The substrate currently exposes three adapters:

* `quasim_modern` — runs the current QuASIM runtime via
  `quantum.python.quasim_sim.simulate` and a deterministic scenario builder
  defined in `quasim.api.scenarios`.
* `quasim_legacy_v1_2_0` — stub for the 1.2.0 legacy engine to enable regression
  comparison wiring later.
* `qvr_win` — Windows-only bridge to the QVR runtime, assuming JSON input/output
  over a CLI process.

Each backend implements the `SimulationBackend` protocol and returns structured
content that is wrapped into a `SubstrateResult` by `QNXSubstrate`.

## CLI surface

A minimal CLI is available to run simulations without writing Python code:

```bash
qnx simulate \
  --scenario smoke \
  --timesteps 100 \
  --seed 42 \
  --backend quasim_modern
```

The CLI constructs a `SimulationConfig`, runs the substrate, and prints a small
summary including backend name, simulation hash prefix, execution time, and
estimated CO₂.
