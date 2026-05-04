# Subsystem contract: `adapters`
_Commit_: `8fc58a9107334a3b53b69a47580df64b185d3317`  _Generated_: 2026-04-29T22:29:11Z

**Path**: `qagents/adapters/`

## Files
- qratum.py (OSR/CEI/SF/HRD; risk=HRD, stability=SF)
- ciir.py (loss/violation; risk=violation, stability=exp(-loss))
- crs.py (CRSI; mirrors IMMUTABLE_BOUNDARIES + PROHIBITED_GOALS, hard rules force risk=1.0)

## Public types / API surface
- Simulator, Proposer, make_<sys>_controller for each of qratum/ciir/crs

## Invariants
- CRS adapter MUST NOT relax IMMUTABLE_BOUNDARIES
- Hard rules force risk=1.0 immediately
