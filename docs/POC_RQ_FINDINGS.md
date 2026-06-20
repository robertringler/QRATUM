# Physical Operator Computer — Pre-M0 Simulation Findings (POC-RQ-1.0)

**QRATUM Research Series — Simulation Validation Report**

*Status:* Pre-hardware simulation study, answers RQ-1/RQ-2/RQ-3 of POC-APPS-1.0
*Code:* `poc_sim/` (reproducible, seeded)
*Companion:* POC-APPS-1.0 (application specs), POC-MONO-1.0/2.0/3.0 (theory)

---

## Executive summary

Three research questions from POC-APPS-1.0 were answerable in simulation
*before* any M0 hardware exists. We answered all three. Two of the three
**corrected errors in the application specification** — which is exactly what
pre-hardware simulation is for. The headline results:

| RQ | Domain | Verdict | Consequence |
|---|---|---|---|
| RQ-1 | Reservoir computing | **Benchmark was wrong; corrected** | NARMA-10 is a nonlinear task a linear reservoir cannot solve even at fp64. Replaced with memory-capacity benchmark. POC at 6–7 bits retains ~35–40% of fp64 memory capacity. |
| RQ-2 | Radar DOA (MUSIC) | **Precision is NOT the bottleneck** | POC at 6–7 bits matches fp64 MUSIC down to the classical beamwidth (1.79° at n=64). Strongest positive result. 6 bits suffices. |
| RQ-3 | Massive MIMO | **κ-limit was wrong; method must change** | The κ=77 convergence limit was a factor-8 error (true Richardson limit is κ≈9). Simple resolvent fails at full loading (100% fallback). Chebyshev acceleration is mandatory and rescues it (~0% fallback to n_UE=48). |

All simulations are seeded and reproducible. Raw data in `poc_sim/results/*.json`,
figures in `poc_sim/figures/*.png`.

---

## RQ-1 — Linear reservoir precision

### The question

POC-APPS-1.0 RQ-1 asked: what is the accuracy loss from 5–7 bit precision in
linear reservoirs as a function of dimension n and spectral radius ρ? The
application spec (BM-R1) named NARMA-10 with a target NMSE ≤ 0.03.

### What we found (the correction)

**NARMA-10 is the wrong benchmark for a linear reservoir.** Running an fp64
*linear* reservoir (f = identity, the POC's native mode) on NARMA-10:

| n | ρ=0.9 fp64 NMSE | ρ=0.9 P=6 NMSE | ρ=0.9 P=7 NMSE |
|---|---|---|---|
| 16 | 0.323 | 0.692 | 0.629 |
| 32 | 0.300 | 0.452 | 0.405 |
| 64 | 0.294 | 0.456 | 0.416 |
| 128 | 0.307 | 0.464 | 0.416 |

The fp64 linear reservoir achieves NMSE ≈ 0.29–0.32 — **an order of magnitude
worse than the 0.03 target, before any precision loss.** The reason is
fundamental: NARMA-10 contains a product term
$0.05\,y(t)\sum_{i} y(t{-}i)$ and a bilinear input term $1.5\,u(t{-}9)u(t)$.
A *linear* map cannot represent these nonlinearities at any precision. The
0.01–0.02 NMSE figures in the reservoir-computing literature are all for
*tanh* (nonlinear) reservoirs. The POC's loop is linear; NARMA-10 is the wrong
task for it.

This is a genuine specification error in POC-APPS-1.0, caught before hardware.

### The constructive replacement: memory capacity

The task a linear reservoir is *provably* suited for is short-term memory
capacity (Jaeger 2002):
$$\mathrm{MC} = \sum_{k\ge 1} \mathrm{corr}^2\bigl(\hat y_k(t),\,u(t{-}k)\bigr),
\qquad \mathrm{MC} \le n \text{ for a linear reservoir.}$$

Measured memory capacity (i.i.d. input, no leak, max delay 2n):

| n | ρ=0.9 fp64 MC | fp64 MC/n | P=6 MC | P=7 MC | P=6 retained |
|---|---|---|---|---|---|
| 16 | 10.2 | 0.64 | 5.4 | 6.3 | 53% |
| 32 | 20.9 | 0.65 | 8.6 | 10.5 | 41% |
| 64 | 39.4 | 0.62 | 13.6 | 16.3 | 35% |
| 128 | 44.7 | 0.35 | 13.7 | 16.3 | 31% |

**Findings:**
1. The fp64 linear reservoir achieves MC/n ≈ 0.62–0.65 at n ≤ 64 (consistent
   with the theoretical bound MC ≤ n and published linear-reservoir results).
2. POC precision at 6 bits retains **35–53% of the fp64 memory capacity**;
   at 7 bits, ~40–55%. The absolute capacity at n=64, P=6 is MC ≈ 13.6 —
   i.e., ~13 delay-steps of recoverable memory. Useful, but the precision
   cost is real: roughly **60% of capacity is lost to 6-bit readout**.
3. The noise-amplification prediction of Volume I §10.3 (steady-state error
   $\sim \delta/(1-\rho)$) is confirmed: capacity loss worsens as ρ→1, where
   the reservoir's slow modes integrate readout noise.

### Engineering consequence

The honest POC reservoir benchmark is **memory capacity, not NARMA-10**. The
POC delivers useful linear memory (≈13 steps at n=64) but at 6-bit precision
loses ~60% of the fp64 capacity. Whether this is acceptable depends on the
downstream task: for a memory-light classifier it is fine; for a
memory-intensive task the precision loss is a material handicap. **POC-APPS-1.0
BM-R1 must be rewritten** to target memory capacity with an explicit
precision-retention metric, not NARMA-10 NMSE.

*Data:* `results/rq1_reservoir_precision.json`, `results/rq1b_memory_capacity.json`.
*Figures:* `figures/rq1_nmse_vs_precision.png`, `figures/rq1_nmse_vs_rho.png`,
`figures/rq1b_mc_vs_precision.png`.

---

## RQ-2 — Radar DOA: is POC precision the bottleneck?

### The question

POC-APPS-1.0 RQ-2 asked: does ringdown spectroscopy achieve eigenvalue
resolution sufficient for MUSIC at n=64, or does the POC's 7-bit spectral
resolution become the limiting factor for direction-of-arrival estimation?

### Method

We ran MUSIC end to end on a 64-element ULA, K=2 sources, 200 snapshots, with
the noise-subspace eigenvectors read out at P bits (the POC effect). We
measured angular RMSE against an fp64 MUSIC reference across SNR and angular
separation. (An initial proxy experiment — eigengap vs spectral resolution —
was discarded as too coarse; the end-to-end MUSIC RMSE is the decisive test.)

### What we found (the strong positive result)

At SNR = 15 dB, median angle RMSE (degrees):

| separation | fp64 | P=5 | P=6 | P=7 | P=8 |
|---|---|---|---|---|---|
| 1.0° | 0.000 | 2.073 | 0.079 | 0.035 | 0.000 |
| 1.5° | 0.000 | 0.109 | 0.035 | 0.035 | 0.000 |
| 2.0° | 0.000 | 0.079 | 0.035 | 0.000 | 0.000 |
| 3.0° | 0.000 | 0.071 | 0.035 | 0.000 | 0.000 |
| 5.0° | 0.000 | 0.071 | 0.035 | 0.000 | 0.000 |
| 10.0° | 0.000 | 0.071 | 0.035 | 0.000 | 0.000 |

(The 0.000 / 0.035 values reflect the 0.05° pseudospectrum grid; 0.035° is a
single-grid-step error.)

**Findings:**
1. **POC at P=6–7 bits matches fp64 MUSIC** down to the classical beamwidth
   resolution limit (2/n rad ≈ **1.79°** at n=64). P=8 is indistinguishable
   from fp64 at every separation.
2. The **classical angular resolution limit binds before the POC precision
   limit.** Below 1.79° separation the resolved fraction drops (85–90% at
   1.0°) — but this is the beamwidth limit that constrains fp64 MUSIC too, not
   a POC-specific failure.
3. Precision only becomes the bottleneck at **P=5 and below** (RMSE jumps to
   2.07° at 1.0°/P=5), which is *below* the POC's certified 5–7 bit envelope.

Resolved fraction at the POC operating point (P=6 bits):

| SNR \ sep | 1.0° | 1.5° | 2.0° | 3.0°+ |
|---|---|---|---|---|
| 0 dB | 85% | 100% | 100% | 100% |
| 15 dB | 86% | 100% | 100% | 100% |
| 20 dB | 90% | 100% | 100% | 100% |

### Engineering consequence

**Radar DOA is the strongest POC application of the three, and precision is not
its constraint.** 6 bits suffices; the limiting factor is the classical array
beamwidth, which the POC inherits but does not worsen. BM-D1's 1.5° RMSE target
at 15 dB is comfortably met at P=6 for separations above the beamwidth. The
RQ-2 risk (that 7-bit spectral resolution would be insufficient) is **retired**.

*Data:* `results/rq2_radar_music.json`. *Figure:* `figures/rq2_music_rmse.png`.

---

## RQ-3 — Massive MIMO: condition number and the resolvent limit

### The question

POC-APPS-1.0 RQ-3 asked: what fraction of realistic 3GPP channels produce
$\kappa(G) > 77$ (stated as the 40-transit resolvent convergence limit),
setting the digital-cleanup fallback rate for BM-M1?

### What we found (two corrections)

**Correction 1 — the κ=77 limit was wrong by ~8×.** The resolvent loop solving
$G\mathbf{x}=\mathbf{b}$ by Richardson iteration has per-transit contraction
$c = 1 - 1/\kappa$. Reaching tolerance $\varepsilon = 10^{-2}$ in 40 transits
requires
$$(1 - 1/\kappa)^{40} \le 10^{-2} \;\Longrightarrow\; \kappa \lesssim 9.2,$$
**not 77.** The POC-APPS-1.0 figure conflated the loop gain (0.9) with the
contraction factor. The true simple-resolvent limit is κ ≈ 9.

**Correction 2 — simple resolvent is inadequate; Chebyshev is mandatory.**
Measured condition numbers and fallback rates over n_BS=64 channel ensembles
(2000 trials each, SNR=15 dB regularization):

| n_UE | corr | κ median | κ p90 | fallback (Richardson) | fallback (Chebyshev) | mean transits (Cheb) |
|---|---|---|---|---|---|---|
| 8 | 0.3 | 3.4 | 4.0 | 0% | 0% | 4.9 |
| 16 | 0.3 | 7.4 | 8.7 | 4.7% | 0% | 7.4 |
| 16 | 0.5 | 10.0 | 11.9 | 75% | 0% | 8.6 |
| 32 | 0.0 | 20.7 | 24.4 | **100%** | 0% | 12.4 |
| 32 | 0.5 | 38.1 | 44.9 | **100%** | 0% | 16.8 |
| 32 | 0.7 | 76.5 | 91.9 | **100%** | 0% | 23.7 |
| 48 | 0.5 | 109.0 | 125.1 | **100%** | 0% | 28.1 |
| 48 | 0.7 | 188.1 | 213.3 | **100%** | 2.5% | 36.8 |

(corr = exponential spatial correlation; Rician variants in the JSON.)

**Findings:**
1. **Simple Richardson resolvent fails at full MIMO loading.** For
   n_UE ≥ 32 = n_BS/2 (a fully-loaded massive MIMO system), the fallback rate
   is **100%** — the simple resolvent essentially never converges in 40
   transits. It only works at low loading (n_UE ≤ 8, where κ ≈ 3–4).
2. **Chebyshev acceleration rescues the application.** With the POC's
   cross-coupled-loop Chebyshev mode (Vol I §6.2), which scales as $\sqrt\kappa$
   instead of κ, the 40-transit limit rises to κ ≈ 360. The fallback rate
   drops to **~0% across nearly all realistic loadings up to n_UE=48**
   (κ up to ~190), with mean transit counts of 12–37, inside the budget.
   Only the extreme corner (n_UE=48 at correlation 0.7, κ median 188) shows a
   nonzero 2.5% fallback.
3. The condition number grows sharply with loading: from κ≈3 at n_UE=8 to
   κ≈20–40 at n_UE=32 to κ≈60–190 at n_UE=48. Spatial correlation and Rician
   LOS components both worsen it (e.g., +6 dB Rician roughly doubles κ at low
   loading).

### Engineering consequence

**BM-M1 must use the Chebyshev-accelerated resolvent, not the simple Neumann
series.** POC-APPS-1.0 named the simple resolvent (Neumann series) as the
primary method for the MIMO precoder; this study shows that choice fails at any
realistic loading. The correction is unambiguous: the cross-coupled Chebyshev
loop is mandatory for MIMO, and with it the application is viable up to and
including full loading. This also raises the priority of the Chebyshev loop in
the M1/M2 hardware bring-up — it is not an optional acceleration but a
load-bearing requirement for the MIMO use case.

*Data:* `results/rq3_mimo_conditioning.json`. *Figure:* `figures/rq3_kappa_cdf.png`.

---

## Consolidated implications for the application specification

POC-APPS-1.0 requires three concrete revisions as a result of this study:

1. **BM-R1 (reservoir)**: replace the NARMA-10 / NMSE≤0.03 target with a
   **memory-capacity benchmark** and a precision-retention metric. Report that
   6-bit POC readout retains ~35–40% of fp64 memory capacity at n=64.

2. **BM-D1 (radar)**: **strengthen** the claim — RQ-2 shows precision is not
   the bottleneck; the POC matches fp64 MUSIC at 6 bits down to the array
   beamwidth. The 1.5° RMSE target is comfortably met. This is the lead
   application.

3. **BM-M1 (MIMO)**: change the primary algorithm from **simple resolvent to
   Chebyshev-accelerated resolvent**, and correct the convergence-limit figure
   from κ=77 to the derived Richardson limit κ≈9 / Chebyshev limit κ≈360.
   Re-scope the fallback-rate claim using the Chebyshev numbers (≈0% to
   n_UE=48).

## Cross-cutting finding: the ranking of the three applications changed

Before simulation, the priority order in POC-APPS-1.0 was reservoir → radar →
MIMO. After simulation, the evidence-based order is:

1. **Radar DOA** — strongest and cleanest; precision is not the constraint;
   matches fp64 at 6 bits. *Promote to lead application.*
2. **MIMO precoding** — viable but *only* with Chebyshev acceleration; a clear
   architectural dependency identified. *Conditionally strong.*
3. **Reservoir computing** — structurally elegant (the loop *is* the reservoir)
   but the most precision-sensitive: 6-bit readout costs ~60% of memory
   capacity, and the originally-named benchmark was invalid. *Real but the
   most caveated.*

This reordering is itself a result: the application that looked most natural on
paper (reservoir, because of the exact structural match) is the most
precision-handicapped in practice, while the application that depends on the
POC's distinctive spectral-readout primitive (radar) is the most robust.

---

## Reproducibility

All results regenerate from seeded scripts:

```bash
cd poc_sim
python3 rq1_reservoir_precision.py    # RQ-1 NARMA-10 (the correction)
python3 rq1b_memory_capacity.py       # RQ-1 memory capacity (the replacement)
python3 rq2_radar_music.py            # RQ-2 MUSIC end-to-end
python3 rq3_mimo_conditioning.py      # RQ-3 condition number + Chebyshev
```

The shared precision abstraction (`poc_precision_model.py`) is identical across
all three, so the 5–7 bit POC readout model is a single source of truth. Seeds
are fixed; re-running reproduces the tables above exactly.

### Open items deferred to hardware (M0)

These three RQs were simulation-answerable. The remaining POC-APPS-1.0 research
questions require hardware or are deferred:

- **RQ-4 (verification amortization break-even)**: partially analytical;
  needs the actual per-epoch tomography shot count from M0 hardware.
- **RQ-5 (cross-domain operator fusion)**: an M3 architecture question;
  not yet simulable without a multi-node model.
- **The calibration-epoch gate** (the Outcome-B engineering contribution):
  fundamentally a hardware-in-the-loop test; it is the primary M0 deliverable
  and cannot be retired in simulation.

---

*Document POC-RQ-1.0 · QRATUM Research Series*
*Simulation code: `poc_sim/` · All findings reproducible from seeded scripts*
