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
| **Phase 2 — specialist parity** | ⏳ Not done | SBML import/export (libSBML), profile-likelihood identifiability, and *coupling* the existing information-theoretic (`omics/information_engine.py`) and neuro-symbolic (`inference/neural_symbolic.py`) layers into the loop remain. Foundations now exist to do so. |
| **Phase 4 — head-to-head benchmarks** (Deliverable B) | ⏳ Not done | Requires installing and running ABC-SysBio / pyABC / PyMC / COPASI / PySB and a shared-problem harness. This is the gate for any literal "industry-leading" claim and has **not** been run. |
| **Phase 5 — CI/security/docs** (D, E) | ◐ Partial | The existing `xenon-ci.yml` already runs the new inference tests (they live under `xenon/tests`, and `scipy` is pinned in `xenon/requirements.txt`). Security review of untrusted-input paths, container/packaging, and the README/whitepaper rewrite remain. |
| **Release recommendation** (F) | — | See below. |

### Release recommendation (evidence-based)
**Research Platform → approaching Pre-Production** for the mechanism-inference vertical. The
inference is now *real and validated against ground truth* (the previous blocker), and the
differentiating active-design loop works. It is **not** yet entitled to an "industry-leading"
claim, because that is defined by the directive as a reproducible head-to-head benchmark vs named
tools (Phase 4), which has not been executed. Per the directive's own non-negotiable #6, no such
claim should be made until that benchmark substantiates a specific, narrow result.

### Recommended next step
Phase 4 is now the critical path: stand up the benchmark harness (start with pyABC and PyMC on a
small published signaling network from BioModels), and report parameter-recovery / model-selection
accuracy and runtime head-to-head. Phase 2's SBML import is the enabling dependency for using
curated community models in that harness.
