# QNX scenarios and circuit generation

The QNX substrate builds small, deterministic circuits from a
`SimulationConfig` before invoking the modern QuASIM runtime. This keeps hashes
stable and makes regression comparisons straightforward.

## ScenarioSpec

`quasim.api.scenarios.ScenarioSpec` captures the inputs needed to construct a
circuit:

- `scenario_id`: logical identifier for the scenario
- `timesteps`: number of gates/steps to generate
- `seed`: RNG seed to keep outputs deterministic
- `extra`: optional dictionary for future extensions

## `build_circuit`

`quasim.api.scenarios.build_circuit(spec)` maps the spec to a synthetic circuit:

- Uses `random.Random(spec.seed)` for reproducibility.
- Applies a small scenario-dependent multiplier derived from `scenario_id`.
- Emits `timesteps` rows of complex values (two entries per row by default).

The resulting circuit is suitable for `quantum.python.quasim_sim.simulate` and
remains stable for identical inputs.

## Engine wiring

- The modern backend (`qnx.backends.quasim_modern`) constructs a
  `ScenarioSpec`, builds the circuit, and calls `quasim_sim.simulate` with
  `precision="fp8"`.
- Legacy and QVR adapters are present as stubs/placeholders, but they follow the
  same scenario contract so they can be upgraded without breaking callers.
