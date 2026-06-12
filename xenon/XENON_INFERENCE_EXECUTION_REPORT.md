# XENON Leadership Directive — Execution Report (Phases 1 & 3)

**Branch:** `claude/qratum-forensics-audit-aoYX2`. **Standard:** evidence-based; every claim
below is backed by a committed, executed test. Full xenon corpus: **236 passed / 0 failed.**

This report covers what was actually executed against the XENON Leadership Directive
(`xenon/XENON_LEADERSHIP_DIRECTIVE.md`), and states plainly what was not.

---

## Deliverable A — Inference Redesign Report (Phase 1: the unlock) ✅ DONE

### Root cause
The Bayesian likelihood compared observations to a **statically stored** concentration
(`xenon/learning/bayesian_updater.py`, previously line 164:
`predicted_conc = mechanism._states[name].concentration`). The model was never run, so the
"inference" could not actually discriminate mechanisms by their dynamics. Per the directive this
was task #1; nothing else was valid until it landed.

### What was built (`xenon/learning/forward_inference.py`, new)
- **Forward model (mean-field).** For the unimolecular networks XENON models (`A→B`, propensity
  `k[A]`), the first moment obeys the linear ODE `dC/dt = Q C`, solved exactly via
  `scipy.linalg.expm(Q t) C0`. Deterministic, exact for the mean, and fast (one matrix
  exponential). The stochastic Gillespie path (`xenon/simulation/gillespie.py`) remains for
  discrete-count regimes.
- **Likelihood.** Gaussian log-likelihood scoring forward **predictions** vs observations
  (per-species measurement noise).
- **Parameter inference.** `metropolis_infer` — adaptive random-walk Metropolis MCMC over
  `log10(rate)` (log-uniform prior), proposal scale auto-tuned during burn-in.
- **Model selection.** `log_marginal_likelihood` / `model_selection` — Monte-Carlo evidence over
  the prior gives Bayesian posterior probabilities over competing topologies.
- **Wiring.** `ExperimentResult` gained `initial_conditions`/`time`; the concentration likelihood
  now forward-simulates the mechanism from those initial conditions to predict the observations,
  with a graceful legacy fallback when no protocol is supplied (so prior callers are unaffected).

### Acceptance evidence (`xenon/tests/test_forward_inference.py`, green)
- On a synthetic `A→B→C` network (`k1=0.8, k2=0.3`), the posterior **recovers the true rates
  within the 95% credible interval**, posterior mean within 50% of truth, sampler healthily mixing.
- **Bayesian model selection ranks the true topology above four decoys** (wrong ordering, missing
  edge, wrong edge, reversed) with >0.5 posterior mass.
- The `BayesianUpdater` now **discriminates a dynamically-correct mechanism from an incorrect one
  via forward simulation** (correct posterior >0.9), proving the defect is fixed end-to-end.
- Forward model verified exact for analytic decay (`C_A(t)=A0 e^{-kt}`).

---

## Deliverable C — Active Experiment Design Report (Phase 3: the differentiator) ✅ DONE

### What was built (`xenon/learning/active_design.py`, new)
The directive's distinctive capability: a **simulate → infer → design → repeat** loop over one
mechanism object. Given the current posterior, candidate experiments are scored by
posterior-predictive variance (a tractable expected-information-gain surrogate under the Gaussian
observation model); the most informative is run, its (simulated, noisy) outcome folded back into
the posterior, and the loop repeats. Reuses the Phase-1 forward model + Metropolis sampler — no new
modelling assumptions.

### Acceptance evidence (`xenon/tests/test_active_design.py`, green)
- Averaged over seeds, the **active-design loop reaches a tighter posterior** (smaller mean
  `log10`-rate std) than random experiment selection for the same number of experiments.
- The selector provably prefers higher-predictive-variance (more informative) candidates.

---

## Honest status of the remaining directive phases

| Phase / Deliverable | Status | Note |
|---|---|---|
| **Phase 1 — forward-model inference** | ✅ Done, tested | Deliverable A |
| **Phase 3 — active experiment design** | ✅ Done, tested | Deliverable C |
| **Phase 2 — SBML interoperability** | ✅ Done, tested | `xenon/learning/sbml_io.py` round-trips mechanisms to/from SBML L3 (validated vs libRoadRunner). Identifiability and *coupling* the information-theoretic / neuro-symbolic layers into the loop remain. |
| **Phase 4 — head-to-head benchmarks** (Deliverable B) | ✅ Done, tested | `xenon/benchmarks/` + `BENCHMARK_REPORT.md`: forward model matches libRoadRunner ODE to 1.7e-6; XENON inference recovers params to 0.17% vs pyABC 4.8% on the shared problem. Extend to more networks/PyMC/COPASI. |
| **Phase 5 — CI/security/docs** (D, E) | ◐ Partial | `xenon-ci.yml` runs the inference suite on the matrix plus a separate benchmark job (installs `requirements-bench.txt`); `scipy` pinned. Security review of untrusted-input paths and the README/whitepaper rewrite remain. |
| **Release recommendation** (F) | — | See below. |

### Release recommendation (evidence-based)
**Pre-Production** for the mechanism-inference vertical on the validated network class. The
inference is real and validated to numerical precision against the gold-standard SBML engine, it
recovers parameters more accurately than the ABC reference on the benchmark, the differentiating
active-design loop works, and SBML interop is in place. A **narrow, benchmark-substantiated
leadership claim** is now defensible (see `xenon/benchmarks/BENCHMARK_REPORT.md`) — *exact-likelihood
Bayesian inference + active design for small mass-action networks* — and nothing broader is claimed.
Remaining gaps to a full "Production" / general claim: more network classes and references in the
benchmark, layer coupling (Phase 2 cont.), security review, and the documentation rewrite.

### Recommended next step
Phase 4 is now the critical path: stand up the benchmark harness (start with pyABC and PyMC on a
small published signaling network from BioModels), and report parameter-recovery / model-selection
accuracy and runtime head-to-head. Phase 2's SBML import is the enabling dependency for using
curated community models in that harness.
