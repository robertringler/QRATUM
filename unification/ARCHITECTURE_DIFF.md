# QRATUM Architecture Diff — Before vs After Remediation
_Phase_: U8 Final Verification  
_Generated_: 2026-04-30

## System Graph Changes

### Added Subsystems

| Subsystem | Path | Reason |
|-----------|------|--------|
| Multi-qubit CIIR controller | `quasim/ciir/multi_qubit/` | GAP-CIIR-MQ-001: missing claimed subsystem |
| CLI runner: multi-qubit | `run_multi_qubit_ciir.py` | GAP-CIIR-RUN-001: missing CLI entry point |
| CLI runner: falsification | `run_falsification.py` | GAP-CIIR-RUN-002: missing CLI entry point |
| CIIR monograph | `manuscripts/ciir_monograph/ciir_monograph.tex` | GAP-DOC-MONO-002: missing main file |
| Monograph chapters 1,2,9-22 | `manuscripts/ciir_monograph/chapters/ch*.tex` | GAP-DOC-MONO-001: missing chapters |

### Modified Subsystems

| Subsystem | Change | Reason |
|-----------|--------|--------|
| `qagents/reality_interface.py` | Fixed syntax errors; added `StaticProposer.__call__`, `LinearSimulator.__call__`; renamed duplicate `_build_fallback` | Unblocked 173 tests |
| `qagents/strategy.py` | `decide()` stub implemented | GAP-STUB-002 |
| `qcore/hamiltonian.py` | `energy()`, `from_semantic_state()` stubs | GAP-STUB-003/004 |
| `qdl/ast.py` | `ASTNode.evaluate()` stub | GAP-STUB-005 |
| `qnx/backends/{cpu,gpu,qpu}.py` | `run()` stubs | GAP-STUB-006/007/008 |
| `qos/envelopes.py` | `SafetyEnvelope.check()` stub | GAP-STUB-009 |
| `qos/policy.py` | `QoSPolicy.enforce()` stub | GAP-STUB-010 |
| `qreal/base_adapter.py` | `_normalize()`, `_to_percept()` stubs | GAP-STUB-012/013 |
| `qstack/alignment/policies.py` | `AlignmentPolicy.evaluate()` stub | GAP-STUB-014 |
| `quasim/hcal/backends/__init__.py` | 4 HCAL backend stubs | GAP-STUB-015-018 |
| `quasim/opt/post_dijkstra_sssp.py` | `find_minimum`, `find_k_minimum` stubs | GAP-STUB-019/020 |
| `quasim/opt/problems.py` | `Problem.evaluate()` stub | GAP-STUB-021 |
| `quasim/quantum/core.py` | cuQuantum raises `ValueError` | GAP-STUB-022 |
| `quasim/quantum/vqe_molecule.py` | LiH/BeH2 raise `ValueError` | GAP-STUB-023/024 |
| `quasim/qunimbus/auth.py` | `refresh_token`, `post_json` stubs | GAP-STUB-025 |
| `qratum/temporal/state.py` | `pickle` → `json` | GAP-SEC-UNSAFE-005 |
| `scripts/validate_quasim_master.py` | `shell=True` → `shlex.split()` | GAP-SEC-UNSAFE-006 |
| `.github/workflows/sbom.yml` | `shell=True` → `shlex.split()` | GAP-SEC-UNSAFE-001 |
| `Dockerfile{,.sandbox-*}` | `USER 1001` + `HEALTHCHECK` | GAP-CTR-NOROOT/HEALTH |
| `requirements*.txt` | torch/qiskit/gunicorn version bumps | GAP-DEP-001..007 |
| 18+ files with syntax errors | Duplicate docstrings, orphaned blocks removed | Pre-existing syntax bugs |

### Removed Subsystems

None. Per DELETIONS.md, no files were deleted in this remediation.

### Collapsed Theoretical Claims

| Claim | Before | After |
|-------|--------|-------|
| LiH/BeH2 VQE | asserted | **retracted** (raises ValueError) |
| cuQuantum backend | asserted | **retracted** (raises ValueError) |

## Before vs After Summary

| Metric | Before | After |
|--------|--------|-------|
| Python syntax errors | 18+ | 0 |
| NotImplementedError (production) | 34+ | 0 |
| Tests passing (CIIR/RIC stack) | 81 | 332 |
| Dangling CLI runners | 2 | 0 |
| Missing monograph files | 18 | 0 (stubs) |
| Pickle in production paths | 2 | 0 |
| shell=True in CI/scripts | 2 | 0 |
| Vulnerable dep pins | 7 | 0 |
| Root containers | 3 | 0 |
| Missing HEALTHCHECK | 3 | 0 |
