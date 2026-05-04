# QRATUM Hardening Report

_Phase_: U3 Harden  
_Generated_: 2026-04-30

## Security Changes Applied

### Code Safety

| GAP ID | File | Issue | Fix Applied |
|--------|------|-------|-------------|
| GAP-SEC-UNSAFE-003/004 | `production_workflow_checkpointing.py` | pickle.load/loads | Already using JSON — no change needed |
| GAP-SEC-UNSAFE-005 | `qratum/temporal/state.py` | pickle.dumps/loads | Replaced with `json.dumps`/`json.loads` + `dataclasses.asdict()` |
| GAP-SEC-UNSAFE-001 | `.github/workflows/sbom.yml` | shell=True subprocess | Replaced with `shlex.split()` + `shell=False` |
| GAP-SEC-UNSAFE-006 | `scripts/validate_quasim_master.py` | shell=True subprocess | Replaced with `shlex.split()` + `shell=False` |
| GAP-SEC-UNSAFE-002 | `aion/runtime/__init__.py` | eval() | Assessed as AION DSL method, not Python built-in eval — no change |

### Dependency Bumps

| GAP ID | Package | Before | After |
|--------|---------|--------|-------|
| GAP-DEP-001/002/003 | torch | `==2.1.2` | `>=2.6.0,<3.0.0` |
| GAP-DEP-004/005 | qiskit | `==0.45.1` | `>=1.4.2,<2.0.0` |
| GAP-DEP-004/005 | qiskit-aer | `==0.13.1` | `>=0.15.0,<1.0.0` |
| GAP-DEP-006/007 | gunicorn | `==21.2.0` | `>=22.0.0,<24.0.0` |
| GAP-DEP-PIN-001 | pandas/matplotlib/seaborn/numpy | unpinned | bounded ranges |

### Container Hardening

| GAP ID | File | Fix Applied |
|--------|------|------------|
| GAP-CTR-NOROOT-Dockerfile | `Dockerfile` | Added `USER 1001` |
| GAP-CTR-HEALTH-Dockerfile | `Dockerfile` | Added `HEALTHCHECK CMD curl -f http://localhost:8080/health \|\| exit 1` |
| GAP-CTR-NOROOT-Dockerfile.sandbox-platform | `Dockerfile.sandbox-platform` | Added `USER 1001` |
| GAP-CTR-HEALTH-Dockerfile.sandbox-platform | `Dockerfile.sandbox-platform` | Added `HEALTHCHECK` |
| GAP-CTR-NOROOT-Dockerfile.sandbox-qradle | `Dockerfile.sandbox-qradle` | Added `USER 1001` |
| GAP-CTR-HEALTH-Dockerfile.sandbox-qradle | `Dockerfile.sandbox-qradle` | Added `HEALTHCHECK` |

### Hardcoded Credentials (Documentation)

| GAP ID | File | Assessment |
|--------|------|------------|
| GAP-SEC-SECRET-001..014 | Various docs and tests | Pattern matches appear to be **example/placeholder** values (e.g., `sk-xxxx`, `test_api_key`, JWT example token). No real credentials found. All are in documentation files or clearly marked test fixtures. |

## Remaining Security Items

| Item | Status | Justification |
|------|--------|--------------|
| GAP-ART-001: 24 .zip artifacts | DEFERRED | Cannot delete without verifying artifact consumers |
| Image signing + SBOM | PARTIAL | sbom.yml workflow present; signing requires CI secrets configuration |
| TLS verify enforcement | ASSERTED | No explicit `verify=False` found in production code paths |
| Secrets via env/secret-manager | ASSERTED | No hardcoded production credentials found |

## Security Gate Status

- ✅ 0 `pickle.load`/`pickle.loads` in production paths
- ✅ 0 `shell=True` in scripts that process untrusted input
- ✅ Vulnerable dependency versions bumped
- ✅ Non-root containers for all sandbox images
- ✅ HEALTHCHECK present on all hardened images
- 🔵 CodeQL scan: configured via existing workflow
- 🔵 Image signing: planned
