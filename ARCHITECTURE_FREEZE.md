# ARCHITECTURE FREEZE - QuASIM / Qubic Repository

**Version**: 1.0.0
**Date**: 2025-12-14
**Status**: FROZEN

---

## 1. What QuASIM / Qubic IS Today

### 1.1 Core Platform

QuASIM (Quantum-Inspired Autonomous Simulation) is a **simulation framework** providing:

1. **GPU-Accelerated Tensor Network Simulation**
   - NVIDIA cuQuantum integration stubs (C++/CUDA ready)
   - CPU fallback for environments without GPU
   - Deterministic execution with SHA-256 state hash verification

2. **Ansys Integration Adapter** (`sdk/ansys/quasim_ansys_adapter.py`)
   - Three integration modes: CO_SOLVER, PRECONDITIONER, STANDALONE
   - PyMAPDL session compatibility (stub implementation)
   - Material models: Mooney-Rivlin, Neo-Hookean, Ogden, Yeoh, Prony Series
   - Mesh import/export via CDB format (stub)

3. **Benchmark Execution Framework** (`evaluation/ansys/`)
   - BM_001: Large-Strain Rubber Block Compression benchmark
   - Statistical validation: Bootstrap CI, modified Z-score outlier detection
   - Multi-format reporting: CSV, JSON, HTML, PDF
   - Reproducibility verification via state hashing

### 1.2 Infrastructure

- **Language**: Python 3.10+ primary, C++/CUDA secondary (stubs)
- **Linting**: Ruff (PEP 8 compliant)
- **Testing**: pytest with coverage
- **CI/CD**: GitHub Actions (`.github/workflows/ci.yml`)

### 1.3 Compliance Posture

Designed for (not yet certified):
- DO-178C Level A readiness
- NIST 800-53 Rev 5 controls
- CMMC 2.0 Level 2 alignment

---

## 2. What QuASIM / Qubic IS NOT

### 2.1 Explicitly Out of Scope

1. **Production C++/CUDA Backend** - Current implementation uses production-ready stubs
2. **Actual cuQuantum Integration** - GPU acceleration is simulated
3. **Real PyMAPDL Execution** - Ansys integration uses mock data
4. **Certified DO-178C Compliance** - Framework is designed for compliance, not certified
5. **Multi-node Distributed Execution** - Single-node execution only
6. **Real-time Simulation** - Batch processing only

### 2.2 Not Included in This Freeze

- Marketing claims about performance
- Future funding assumptions
- Customer deployment specifics
- Hardware-specific optimizations (beyond GPU/CPU branching)

---

## 3. Frozen Subsystems

| Subsystem | Path | Status |
|-----------|------|--------|
| Ansys Adapter | `sdk/ansys/quasim_ansys_adapter.py` | FROZEN |
| BM_001 Executor | `evaluation/ansys/bm_001_executor.py` | FROZEN |
| Performance Runner | `evaluation/ansys/performance_runner.py` | FROZEN |
| CI Configuration | `.github/workflows/ci.yml` | FROZEN |
| Core Dataclasses | `RunMetrics`, `ComparisonMetrics`, `StatisticalSummary` | FROZEN |

---

## 4. CI Contract

All merges to `main` or `develop` must pass:

1. **Syntax Validation**: `python3 -m py_compile` on all `.py` files
2. **Linting**: `ruff check .` with zero errors
3. **Formatting**: `ruff format --check .`
4. **Unit Tests**: `pytest tests/` (coverage target: >90% for SDK/adapters)
5. **Security**: CodeQL analysis with zero high/critical alerts

---

## 5. Anti-Entropy Boundaries

### 5.1 DO NOT Expand

- New physics domains without explicit approval
- New integration adapters without benchmark validation
- New compliance frameworks without audit trail

### 5.2 DO NOT Rename

- `QuasimAnsysAdapter` class
- `SolverMode` enum values
- `MaterialModel` enum values
- `BM_001` benchmark identifier

### 5.3 DO NOT Refactor

- State hash computation algorithm
- Bootstrap confidence interval implementation
- Outlier detection (modified Z-score) algorithm

---

## 6. Change Protocol

Any modification to frozen subsystems requires:

1. **Written justification** documenting correctness, security, or CI necessity
2. **Regression test** proving no behavioral change (except intended fix)
3. **Code review** approval
4. **CI green** on all protected branches

---

## 7. Technical Debt Registry

| ID | Description | Priority | Resolution Path |
|----|-------------|----------|-----------------|
| TD-001 | C++/CUDA backend stubs need real implementation | HIGH | Requires cuQuantum SDK |
| TD-002 | PyMAPDL session integration is mock | MEDIUM | Requires Ansys license |
| TD-003 | PDF generation requires reportlab | LOW | Optional dependency |

---

## 8. Validation Checksums

Files frozen at this revision:

```
bm_001_executor.py: SHA-256 verified on commit
quasim_ansys_adapter.py: SHA-256 verified on commit
performance_runner.py: SHA-256 verified on commit
```

---

**END OF ARCHITECTURE FREEZE**
