# Flaky / non-deterministic test surfaces

_Commit_: `8fc58a9107334a3b53b69a47580df64b185d3317`  _Generated_: 2026-04-29T22:27:58Z

Heuristic search of `tests/` for symbols that imply nondeterminism without seed-pinning:

- `qubic-viz/tests/test_mesh_generator.py` — uses `random.*` without explicit seeding
- `qubic/visualization/tests/test_backends.py` — uses `random.*` without explicit seeding
- `qubic/visualization/tests/test_pipelines.py` — uses `random.*` without explicit seeding
- `tests/benchmarks/compression/test_validators.py` — uses `random.*` without explicit seeding
- `tests/temporal/test_ftl.py` — uses `random.*` without explicit seeding
- `tests/test_qgh_nonspec_algorithms.py` — uses `random.*` without explicit seeding
- `tests/test_terc_bridge_observables.py` — uses `random.*` without explicit seeding
