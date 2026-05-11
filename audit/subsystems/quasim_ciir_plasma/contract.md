# Subsystem contract: `quasim_ciir_plasma`

_Commit_: `8fc58a9107334a3b53b69a47580df64b185d3317`  _Generated_: 2026-04-29T22:29:11Z

**Path**: `quasim/ciir/`

## Files

- mhd.py (2D RMHD ψ-ω)
- topology.py (X/O-points, current sheets)
- spectrum.py (FFT shells, FKR/plasmoid scaling)
- control.py (u=B_ext, E_drive, η_eff, F_ω + ReconnectionController)
- ciir.py (C/I/R + FAILURE_MODES F1-F5 + stability_inequality)
- engine.py (run_reconnection_control)

## Public types / API surface

- ReconnectionController
- FAILURE_MODES F1..F5
- stability_inequality

## Invariants

- RMHD evolution preserves ∇·B = 0 by construction (vector potential ψ)
- engine emits results.json + figures
