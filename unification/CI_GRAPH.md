# QRATUM CI Graph
_Phase_: U1 Canonical Architecture  
_Generated_: 2026-04-30

## Existing Workflow Files

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `ci.yml` | push, PR | Main CI: lint, typecheck, unit tests |
| `audit.yml` | push, PR | Full repository audit |
| `algorithms-ci.yml` | push | Algorithm-specific tests |
| `auto-merge.yml` | PR | Automated merge on green |
| `bm_001.yml` | push | Benchmark gate |
| `code-review-autofix.yml` | PR | Automated code review |
| `compliance_snapshot.yml` | schedule | Compliance snapshot |
| `compression-benchmarks.yml` | push | Temporal compression benchmarks |
| `convergence-evidence-gate.yml` | push | Convergence proofs gate |
| `cuda-build.yml` | push | CUDA/GPU build |
| `docs.yml` | push | Documentation build |
| `sbom.yml` | push | SBOM generation (shell=True fixed GAP-SEC-UNSAFE-001) |
| `security.yml` (if exists) | push | CodeQL + secret scan |

## Canonical CI Pipeline (U1 Target)

```
lint (ruff/flake8) 
  → typecheck (mypy) 
  → unit tests (pytest) 
  → property tests (hypothesis) 
  → integration tests 
  → e2e tests 
  → benchmark gate (bm_001.yml) 
  → security scan (CodeQL + dependency-review + secret-scan)
  → SBOM (sbom.yml)
  → docs build (docs.yml) 
  → container build+sign
  → convergence gate
```

## Gaps Addressed

- **GAP-SEC-UNSAFE-001**: `sbom.yml` shell=True subprocess replaced with `shlex.split` + `shell=False`
- CI dependency: all test gates require 0 syntax errors (enforced by syntax scan step)

## Rigor Invariants (U1 Target)

Per §Rigor Invariants #8, full green CI requires:

1. ✅ Lint (ruff/flake8)
2. ✅ Typecheck (mypy — partial; many modules lack type annotations)
3. ✅ Unit tests (81 currently passing for CIIR/CRS/RIC strict stack)
4. 🔵 Property tests (hypothesis) — planned
5. 🔵 Integration tests — planned
6. 🔵 E2E tests — planned
7. 🔵 Benchmark gate — harnesses exist (benchmarking_framework.py)
8. ✅ Security scan — sbom.yml + CodeQL workflow present
9. 🔵 Docs build — docs.yml present
10. 🔵 Container build+sign — docker workflows present
