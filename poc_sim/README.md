# POC Simulation Suite

Pre-M0 simulation studies for the Physical Operator Computer (POC), answering
the simulation-tractable research questions of `docs/POC_APPLICATIONS.md`
(POC-APPS-1.0). Findings are written up in `docs/POC_RQ_FINDINGS.md`.

## Purpose

Validate — before any hardware exists — whether the three candidate POC
application domains actually fall inside the advantage envelope (dense
operator, ≤7-bit precision, high reuse) established in POC-MONO-3.0, and at
what precision cost. Two of the three studies corrected errors in the
application specification.

## Layout

| File | Research question | Result |
|---|---|---|
| `poc_precision_model.py` | Shared shot-noise / P-bit readout model | Single source of truth for all RQs |
| `rq1_reservoir_precision.py` | RQ-1: NARMA-10 reservoir vs precision | Showed NARMA-10 is the wrong (nonlinear) benchmark |
| `rq1b_memory_capacity.py` | RQ-1: linear memory capacity vs precision | 6-bit POC retains ~35–40% of fp64 capacity |
| `rq2_radar_music.py` | RQ-2: MUSIC DOA, fp64 vs POC precision | Precision is NOT the bottleneck (beamwidth binds first) |
| `rq3_mimo_conditioning.py` | RQ-3: MIMO condition number + convergence | Simple resolvent fails; Chebyshev mandatory |

Outputs land in `results/*.json` (raw data) and `figures/*.png` (plots).

## Running

```bash
pip install numpy scipy matplotlib
cd poc_sim
python3 rq1_reservoir_precision.py
python3 rq1b_memory_capacity.py
python3 rq2_radar_music.py
python3 rq3_mimo_conditioning.py
```

All scripts are seeded; re-running reproduces the published tables exactly.
Runtime is a few minutes total on a single core.

## The POC precision model

`poc_precision_model.py` abstracts the POC's analog readout as additive
shot noise at the given bit precision followed by uniform quantization. With
N_ph photons per mode per shot and S-shot averaging, effective precision is
δ ~ 1/√(N_ph·S), and the shot budget scales as 4^P (the precision wall,
POC-MONO-3.0 Theorem 11). This is the conservative engineering abstraction used
throughout the monograph series; all three RQ studies import it so the
precision assumption is identical and auditable.
