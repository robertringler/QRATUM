# QRATUM Copilot Instructions

QRATUM is a **deterministic, auditable AI execution framework** — a sovereign control plane for heterogeneous cluster computing with cryptographic verification, immutable audit trails, and quantum-classical hybrid support. Version: 2.0.0.

## Tech Stack

- **Python 3.10+** (primary). C++20/Vulkan for `ciir_vr_simulation/`.
- **Core deps**: numpy, pyyaml, click, matplotlib
- **Quantum**: Qiskit ≥1.0, PennyLane ≥0.35, Qiskit-Aer, PySCF
- **Dev tools**: `ruff` (lint/format), `pytest` + `pytest-cov`, `mypy`, `bandit`
- **Build artifacts**: Python wheel, TypeScript SDK, Electron desktop app

## Build & Test

```bash
# Install deps
pip install -r requirements.txt -r requirements-dev.txt

# Run tests (pytest)
make test
# or directly:
python -m pytest tests/ -v

# Lint & format
make lint
make fmt

# Full production build (Python wheel + TS SDK + Electron)
bash build_production.sh

# Docker full stack
docker-compose up --build
```

See [BUILD_INSTRUCTIONS.md](../BUILD_INSTRUCTIONS.md) for full build output structure.

## Architecture — 6-Tier Pipeline

```
Intent (QIL) → Authority (QCore) → Contract → Spine (Dispatch) → Adapters → Events
```

| Tier | Directory | Role |
|------|-----------|------|
| QIL | `qil/` | Q Intent Language — grammar, AST, parser |
| QCore | `qcore/` | Sovereign authority — **authorizes only, NEVER executes** |
| Contracts | `contracts/` | Immutable frozen dataclasses, SHA-256 hash-addressed |
| Events | `events/` | Append-only cryptographic event log (hash-chained) |
| Spine | `spine/` | Validates contract bundles, dispatches to adapters |
| Adapters | `adapters/` | Zero-policy substrate adapters (GB200, MI300X, Cerebras, QPU, CPU…) |

Platform verticals (14 domains) live in `qratum/`. Quantum implementations (VQE, QAOA) are in `quasim/quantum/`.

See [QRATUM_ARCHITECTURE.md](../QRATUM_ARCHITECTURE.md) for the full spec.

## Critical Rules

- **QCore NEVER executes** — it only raises `AuthorizationError` or grants authorization. Execution belongs to `spine/` + adapters.
- **Contracts are immutable** — use `@dataclass(frozen=True)`. Never mutate a contract after issuance.
- **All actions must be logged** to the cryptographic event chain in `events/`.
- **Frozen subsystems**: `adapters/ansys/`, BM_001 executor, CI config are architecture-frozen — no changes without a formal freeze amendment. See [ARCHITECTURE_FREEZE.md](../ARCHITECTURE_FREEZE.md).
- **Security**: SHA-256 state hashing and Merkle event chains are compliance-critical (DO-178C, NIST 800-53). Do not bypass or weaken these.

## Code Conventions

- Classes: `PascalCase`; functions/methods: `snake_case`; constants: `UPPER_SNAKE_CASE`; private: `_prefix`
- Use `from __future__ import annotations` and type hints throughout
- Custom exceptions: `AuthorizationError`, `CapabilityResolutionError` (do not use bare `Exception`)
- One test file per module under `tests/`; coverage target ≥90%
- Commit format: `<type>: <subject>` — types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`
- Branch naming: `feature/`, `bugfix/`, `docs/`, `refactor/`

## Key Docs

- [CONTRIBUTING.md](../CONTRIBUTING.md) — PR workflow, code style, review process
- [API_REFERENCE.md](../API_REFERENCE.md) — Public API surface
- [QUICKSTART.md](../QUICKSTART.md) / [QUICKSTART_FULLSTACK.md](../QUICKSTART_FULLSTACK.md) — Getting started
- [COMPLIANCE_ASSESSMENT_INDEX.md](../COMPLIANCE_ASSESSMENT_INDEX.md) — Compliance posture
- [ciir_vr_simulation/README.md](../ciir_vr_simulation/README.md) — C++/Vulkan VR simulation component
- [.github/workflows/](.github/workflows/) — CI pipelines (lint → test → build → SBOM)

