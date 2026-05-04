# Compiled Realities Simulation Package

Pre-registered simulation framework for stress-testing the conditional no-go theorem on backward lab-time signaling in compiled/analog realities.

## Installation

```bash
pip install -e .
```

## Running Tests

```bash
pytest tests/ -v
```

## Running Simulation

```bash
python -m compiled_realities.run --config configs/main.yaml
```

## Structure

- `src/compiled_realities/` - Main package source
  - `models/` - Model implementations (CPTP, Lattice Field)
  - `diagnostics/` - S_lab and χ diagnostics
  - `audit/` - Causality audit protocol
  - `sensitivity/` - Anti-holographic sensitivity analysis
  - `io/` - Artifact I/O handlers
  - `utils/` - Utility functions
- `configs/` - Configuration files
- `tests/` - Unit tests
- `docs/` - Documentation
- `runs/` - Generated simulation artifacts

## Key Diagnostics

**S_lab(θ)**: Lab-time signaling diagnostic measuring backward signaling from future interventions to past observables.

**χ(θ)**: Effective-time loop diagnostic measuring closed timelike curves in the effective metric.

## Pre-Registered Hypotheses

**H1**: S_lab(θ) = 0 within statistical error for all θ
**H2**: χ(θ) may become negative in some parameter regimes
**H3**: Approaching χ < 0 triggers validity degradation
**H4**: χ more sensitive to bulk than boundary parameters
