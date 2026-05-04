# CI/CD audit findings

_Commit_: `8fc58a9107334a3b53b69a47580df64b185d3317`  _Generated_: 2026-04-29T22:27:58Z

- Workflows discovered: **56** (see `audit/cicd/matrix.csv`).
- Container hardening:
  - `Dockerfile.production`: ✅ non-root USER `qratum`, ✅ HEALTHCHECK present.
  - `Dockerfile`, `Dockerfile.sandbox-platform`, `Dockerfile.sandbox-qradle`: ❌ no `USER` directive (run as root), ❌ no HEALTHCHECK.
- `requirements.txt` has 4 unpinned dependencies (numpy / pandas / matplotlib / seaborn) — `requirements-prod.txt` is fully pinned.
- Pre-commit config: `.pre-commit-config.yaml` present.
- SBOM: `.sbom/` directory present.
- Pinned-but-vulnerable deps (advisory DB hits): see `audit/security/sbom_diff.md`.
- Entry points: `pyproject.toml [project.scripts]` defines `quasim-hcal`, `quasim-revultra`, `quasim-qgh`. `setup.py` redundantly defines `quasim-hcal=quasim.cli.main:main`. **Drift**: `pyproject.toml` points at `quasim.hcal.cli:main` while `setup.py` points at `quasim.cli.main:main`.
