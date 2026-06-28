# XENON — Tier-0 Correctness Remediation + In-Silico Recovery Validation

*Follow-up to [`audit/PHASE1_FORENSIC_AUDIT.md`](audit/PHASE1_FORENSIC_AUDIT.md). Scope chosen by
the user: implement the Tier-0 correctness fixes and a closed-loop in-silico recovery harness.*

Reproduce:
```bash
pip install numpy scipy click
PYTHONPATH=$PWD python xenon_campaigns/audit/audit_findings.py     # 5/5 Tier-0 FIXED
PYTHONPATH=$PWD python xenon_campaigns/recovery/run_recovery.py    # recovery + identifiability
python -m pytest xenon/tests/test_insilico_recovery.py xenon/tests/test_gillespie.py \
                 xenon/tests/test_runtime.py -p no:cacheprovider -o addopts=""   # 31 passed
```

---

## 1. What changed and why

The audit showed XENON was *reproducible but computing an invalid posterior fed by a simulator that
did not simulate the declared model*. Tier-0 fixes that core. Status now reported by the audit
verifier against the **actual code** (not hand-rolled reproductions):

| ID | Finding | Fix | Status |
|----|---------|-----|--------|
| **F1** | SSA dropped reverse reactions | Compile each reversible `Transition` into forward + reverse channels (`gillespie.py::_build_reactions`); detailed balance now holds | ✅ FIXED |
| **F2** | ±1 unimolecular only | Net-stoichiometry updates + mass-action propensities with falling-factorial reactant orders (bimolecular supported) | ✅ FIXED |
| **F4** | Likelihood averaged χ² over `n` | Joint Gaussian/log-normal likelihood `exp(−½χ²)` (summed, not averaged) so evidence accumulates | ✅ FIXED |
| **F5** | Regeneration reset posteriors | `MechanismPrior.initialize_new_priors` initializes only *new* mechanisms; the kernel uses it | ✅ FIXED |
| **F6** | Likelihood on one 0.1 s transient | Replicated SSA to steady state; mean prediction + Monte-Carlo SE folded into the noise model in quadrature | ✅ FIXED |
| (bonus) | `record_interval` dropped the final state for fast systems | Always record the terminal state in `GillespieSimulator.run` | ✅ FIXED |
| **F12** | Missing `Optional`/`List` imports | Added | ✅ FIXED |
| **F3** | Perturbation likelihood inert | — | ⏳ Deferred (honestly reported PRESENT) |

**Audit verifier output (actual code):**

```
[F1] symmetric A<->B mean A=52.0 nM (expect ~50)                         -> FIXED
[F2] A+B->C final A=19.9 B=0.0 C=39.9 (fired, conserved, B-limiting)     -> FIXED
[F4] concentration likelihood n=1 -> 1.353e-01, n=5 -> 4.540e-05         -> FIXED
[F5] after regeneration: evidenced=0.709 other=0.022 new=0.269           -> FIXED
[F6] predicted states carry Monte-Carlo SE (conc_mc_se): True            -> FIXED
[F3] perturbation likelihood (kernel format) = 0.1                       -> PRESENT (deferred)
Tier-0: 5/5 findings FIXED.
```

Determinism is preserved end to end (all sub-seeds derive from the master seed); the EGFR campaign
remains bit-reproducible and the determinism + numerical-stability gates still pass. **No regressions:
the full `xenon/tests` failure count dropped from 20 → 18 (the remaining 18 are pre-existing in
`test_integration.py`/`test_mechanism.py`, files untouched here; `test_cli.py` has a pre-existing
`SyntaxError`).** The two simulation/runtime suites and the new recovery suite are green (31 passed).

---

## 2. The honest consequence: mock-data convergence is meaningless

With a now-*correct, sharp* likelihood, the legacy EGFR campaign — which still uses the
**mechanism-independent mock experiment executor (F7)** — collapses to one mechanism in a single
iteration (entropy ≈ 0). That is an artifact, not a discovery, and the EGFR report is banner-flagged
as superseded. The real test of the engine therefore requires a **generative ground truth**, which is
what the recovery harness provides.

---

## 3. In-silico recovery validation (the real test)

`xenon/validation/insilico_recovery.py` generates observations from a **known** ground-truth
mechanism M* (reversible EGFR activation, `k_f=2, k_r=1`, so `K=2`, total 100 nM) and asks whether
XENON recovers it.

### Scenario A — recovery + evidence accumulation
Candidates span the identifiable quantity `K ∈ {0.5, 1, 1.5, 2(true), 2.5, 4}`. M* steady state came
out at **inactive 33.6 / active 66.0 nM** (analytic `K=2` ⇒ 33.3/66.7 — correct, confirming the F1
fix).

| Candidate | K | Final posterior |
|---|---|---|
| **C_K2 (truth)** | 2.0 | **0.99993** |
| C_K2_5 | 2.5 | 6.8×10⁻⁵ |
| C_K05, C_K1, C_K1_5, C_K4 | 0.5–4 | ~0 |

Posterior on the truth over the 20 experiments:

```
0.59, 0.44, 0.16, 0.08, 0.11, 0.21, 0.42, 0.87, 0.94, 0.86, 0.88, 0.89,
0.98, 0.999, 0.9994, 0.998, 0.998, 0.999, 0.9999, 0.9999
```

This is a realistic Bayesian learning curve: it **dips** when a few noisy early measurements transiently
favour a neighbour (K=2.5), then climbs decisively as evidence accumulates — exactly the behaviour
the F4 fix is supposed to enable, and which the *old* averaged likelihood could not produce.
**Verdict: true mechanism recovered; evidence accumulates with data.**

### Scenario B — identifiability, reported honestly
Candidates `C_K2` (`k_f=2,k_r=1`) and `C_K2_alt` (`k_f=4,k_r=2`) share `K=2`. A two-state chain's
stationary distribution depends **only on K**, so they are analytically indistinguishable from a
steady-state concentration. With well-resolved predictions (64 replicates, `t_max=10`):

- predicted active: **C_K2 = 66.5 nM**, **C_K2_alt = 64.9 nM** — difference **1.56 nM ≤ 2×MC-SE (2.05 nM)** ⇒ agree within Monte-Carlo error.
- final posterior split **0.37 / 0.63** — i.e. roughly even, the residual tilt being simulation noise, not signal.

**Verdict:** XENON correctly recovers the *identifiable* quantity (K) but **cannot** separate equal-K
mechanisms from concentration data alone. The harness reports this as a non-identifiability finding —
the correct scientific behaviour, and the concrete motivation for Phase 5 (identifiability analysis)
and Phase 6 (optimal experimental design: a kinetic/time-course experiment would break the degeneracy).

---

## 4. A subtlety this surfaced (logged for Phase 4/5)

The likelihood treats each candidate's predicted observable as a fixed point estimate and sums
independent-experiment χ² against it. But the prediction's Monte-Carlo error is **systematic** across
repeated experiments, so naively accumulating many experiments against one noisy prediction can
over-discriminate truly-equivalent mechanisms (an early run split an equal-K pair 0.97/0.03 at low
replicate count). Mitigations already in place: fold MC-SE into σ (F6) and raise replicate count.
**Proper fix (Phase 4):** marginalize the likelihood over prediction uncertainty (or share a
prediction-error term across experiments) rather than treating each experiment as fully independent.
This is recorded as a known limitation, not silently left.

---

## 5. What is and isn't established now

**Established (verified, reproducible):**
- Reversible/bimolecular kinetics simulate correctly (F1/F2); detailed balance and conservation hold.
- The likelihood accumulates evidence with data (F4); sequential evidence is no longer wiped (F5);
  simulation noise enters the noise model (F6).
- On a known ground truth, XENON **recovers the true identifiable mechanism** and **flags
  non-identifiable degeneracies** instead of overclaiming.

**Not established (out of scope / requires resources unavailable here):**
- Any real biological claim about EGFR — needs real evidence (Phase 2 databases / assays).
- Parameter-level posteriors and a full identifiability suite (Phase 5).
- The deferred F3 perturbation channel, optimal experimental design (Phase 6), external/clinical
  validation (Phase 7), and production hardening/publication (Phases 9–10).

The platform has moved from *reproducible-but-invalid* to *reproducible, correct on synthetic ground
truth, and honest about what it cannot yet identify* — the necessary foundation before any real data
is connected.
