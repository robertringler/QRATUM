# QRATUM Reproducibility Manifest

_Commit_: `8fc58a9107334a3b53b69a47580df64b185d3317`  _Generated_: 2026-04-29T22:27:58Z

## Audit reproduction

```bash
git checkout 8fc58a9107334a3b53b69a47580df64b185d3317
python3 /tmp/audit_work/run_audit.py            # generates audit/inventory, code, security, tests, cicd, theory, artifacts
python3 /tmp/audit_work/build_deliverables.py   # generates the merged audit/* deliverables
```

The audit is read-only and deterministic given the same commit SHA.

## CIIR / CRS / RIC subsystem reproduction (verified present)

```bash
# strict five-module CIIR-CRS-RIC
PYTHONPATH=. python3 -m qagents.ciir_crs_ric.demo
python3 -m pytest tests/test_ciir_crs_ric_strict.py --override-ini="addopts=" -q

# framework variant
PYTHONPATH=. python3 experiments/ciir_crs_ric_demo.py
python3 -m pytest tests/test_ciir_crs_ric.py --override-ini="addopts=" -q

# MVRI
PYTHONPATH=. python3 -m qagents.mvri.demo
python3 -m pytest tests/test_mvri.py --override-ini="addopts=" -q

# RealWorld bridge
PYTHONPATH=. python3 -m qagents.realworld_bridge.demo
python3 -m pytest tests/test_realworld_bridge.py --override-ini="addopts=" -q

# Control geometry
python3 -m pytest tests/test_control_geometry.py --override-ini="addopts=" -q

# Trajectory controller
PYTHONPATH=. python3 -m qagents.trajectory_controller.demo
python3 -m pytest tests/test_trajectory_controller.py --override-ini="addopts=" -q

# RIC v1 / v2 + adapters + bridge
python3 -m pytest tests/test_reality_interface_controller.py tests/test_ric_v2.py tests/test_ric_adapters.py tests/test_ric_ciir_bridge.py --override-ini="addopts=" -q

# Plasma reconnection (CIIR)
python3 run_plasma_reconnection.py
python3 -m pytest tests/test_plasma_reconnection.py --override-ini="addopts=" -q
```

## Reproductions blocked by missing artefacts

- `python3 run_multi_qubit_ciir.py` — file absent (GAP-CIIR-RUN-001)
- `python3 run_falsification.py`    — file absent (GAP-CIIR-RUN-002)
- `python3 -m pytest tests/test_ciir_falsification.py tests/test_ciir_gaps_1_4_8.py tests/test_ciir_extensions.py tests/test_multi_qubit_controller.py` — corresponding source modules under `quasim/ciir/multi_qubit/**` are absent (GAP-CIIR-MQ-001)
- Manuscript build for Ch9–Ch22 — chapter `.tex` files absent (GAP-DOC-MONO-001)

## Environment

- Python pinned (production): see `requirements-prod.txt` (43 lines, 100% `==` pinned).
- Python loose (default): `requirements.txt` (4 unpinned lines — flagged GAP-DEP-PIN-001).
- Containers: `Dockerfile.production` runs as `qratum` user with HEALTHCHECK; the other 3 Dockerfiles are root + no healthcheck (GAP-CTR-NOROOT-*/ GAP-CTR-HEALTH-*).
