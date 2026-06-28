# XENON — Phase 1 Forensic Scientific Audit

**Reviewer stance:** independent scientific reviewer + engineering lead. Standard applied:
publication-grade computational biology (Nature Biotechnology / Cell Systems / PNAS).
**Scope of this document:** Phase 1 of the 10-phase program — a complete, evidence-grounded audit
of the existing scientific kernel.

> **Remediation status.** Findings **F1, F2, F4, F5, F6, F3** (and bonus final-state-recording +
> F12 imports) are now **FIXED** and validated by a closed-loop in-silico recovery harness and unit
> tests — see [`../TIER0_REMEDIATION_AND_RECOVERY.md`](../TIER0_REMEDIATION_AND_RECOVERY.md).
> [`audit_findings.py`](audit_findings.py) has been converted to a **status verifier** that exercises
> the actual code and reports `6/6 findings FIXED`. The audit text below is preserved as the original
> diagnosis; read it together with the remediation report.

> **Scope honesty.** Phases 2–10 of the tasking program (live ingestion from GEO/TCGA/PRIDE/…,
> multiple inference backends, identifiability suites, optimal experimental design, external/clinical
> validation, lab automation, full publication package) are **months of engineering and require
> network access to ~20 external databases plus real laboratory/clinical data that this environment
> does not have.** I will not fabricate those integrations or invent validation numbers. What I have
> done — and what the program itself designates as the prerequisite — is execute Phase 1 in full, at
> reviewer rigor, against the real code, and then give an honest, prioritized engineering roadmap for
> 2–10. The audit's central conclusion: **the inference engine has correctness defects that must be
> fixed before any external data is connected, because connecting real data to the current kernel
> would produce confidently wrong results.**

---

## Executive summary

Determinism is solved (verified separately). But determinism only guarantees that XENON reproduces
**the same answer**, not a **correct** one. This audit finds that the current kernel reliably and
reproducibly computes a posterior that is **not a valid Bayesian posterior over mechanisms**, fed by
a simulator that **does not simulate the declared model**.

The five load-bearing defects:

| ID | Severity | One-line statement |
|----|----------|--------------------|
| **F1** | 🔴 Critical | Gillespie SSA **silently drops reverse reactions** — reversible mechanisms are simulated as irreversible; detailed balance is impossible. |
| **F2** | 🔴 Critical | **Stoichiometry is hardcoded to ±1 unimolecular**; the `Transition.stoichiometry` field is ignored and there is no bimolecular propensity. |
| **F4** | 🔴 Critical | The likelihood **averages χ² over the number of measurements**, so evidence does not accumulate with data — a direct violation of the likelihood principle. |
| **F5** | 🔴 Critical | Hypothesis regeneration **resets accumulated posteriors back to the prior**, destroying sequential learning. |
| **F6** | 🔴 Critical | The likelihood is evaluated on **a single stochastic SSA sample at a fixed 0.1 s transient**, conflating Monte-Carlo noise with model evidence and never reaching steady state. |

Consequence: the headline EGFR result (dominant posterior 0.9986, entropy 0.0123) is a
**reproducible artifact of noise concentrating under an improper update**, not a measurement of which
mechanism is true. It must not be reported as a scientific finding.

---

## Component readiness map

| Component | File | Status | Notes |
|---|---|---|---|
| Determinism / seeding | `xenon_kernel.py` | ✅ Production | Fixed; loop is a pure function of the seed. |
| Mechanism data model | `core/mechanism.py` | 🟡 Prototype | Sound DAG + thermo/conservation checks, but two-state only in practice; stoichiometry unused downstream. |
| Gillespie SSA | `simulation/gillespie.py` | 🔴 Incorrect | Forward-only, ±1-only (F1, F2). Not the declared model. |
| Bayesian updater | `learning/bayesian_updater.py` | 🔴 Incorrect | Improper likelihood normalization (F4); inert perturbation channel (F3). |
| Mechanism prior | `learning/mechanism_prior.py` | 🟡 Mislabeled | "Conservation" prior is a thermodynamics re-read; literature prior constant (F8). |
| Experiment executor | `xenon_kernel.py::_execute_experiment` | 🔴 Mock | Observations independent of mechanism (F7). |
| Convergence / acquisition | `xenon_kernel.py` | 🔴 Prototype | Entropy over improper posterior; random experiment selection; no identifiability (F9). |
| Provenance | `core/mechanism.py` | 🟢 Usable | Lineage strings recorded; needs structured records for Phase 2. |

---

## Findings (each: explanation · severity · scientific consequence · fix strategy · verification)

### F1 — Reverse reactions are silently dropped 🔴 Critical
**Explanation.** `GillespieSimulator._compute_propensities` (gillespie.py:184–194) iterates
`mechanism._transitions` and builds one forward propensity `k·n_source` per edge. The `reversible`
and `reverse_rate` fields on `Transition` are never read; no reverse reaction channel is added.
`_update_state` (lines 230–231) only does `source−=1, target+=1`.
**Empirical.** A symmetric reversible reaction `A⇌B` (`k_f=k_r=1`) that must equilibrate at ~50/50
instead runs to completion: **final A=0.0, B=99.6 nM** (`audit_findings.py` F1).
**Scientific consequence.** Every reversible mechanism — including the EGFR templates, which set
`reversible=True, reverse_rate=k/10` — is simulated as an irreversible cascade. Steady states,
occupancies, and therefore all concentration likelihoods are computed from the wrong dynamical
system. The README's claimed "detailed balance for reversible reactions" does not hold.
**Fix strategy.** Expand each reversible `Transition` into two reaction channels at build time
(forward `k_f·n_source`, reverse `k_r·n_target`), or add a reverse propensity term in
`_compute_propensities` and a matching branch in `_update_state`. Add a detailed-balance unit test:
symmetric rates ⇒ 50/50 ± MC error; `K=k_f/k_r` ⇒ split `K/(1+K)`.
**Verification.** `audit_findings.py::f1_reverse_reactions_dropped` (currently fails); post-fix it
should equilibrate to the analytic split within Monte-Carlo error over ≥8 seeds.

### F2 — Stoichiometry and bimolecular kinetics unsupported 🔴 Critical
**Explanation.** `_update_state` hardcodes ±1; `_compute_propensities` uses `k·n_source` only.
`Transition.stoichiometry` exists but is ignored. No `k·n_A·n_B/(V·N_A)` bimolecular propensity.
**Empirical.** Declared `stoichiometry={'A':-2,'B':1}` is carried on the object but never applied
(`audit_findings.py` F2).
**Scientific consequence.** Directly blocks Phase 3: dimerization (`2R→R₂`), receptor–ligand binding
(`R+L→RL`), complex formation, and cooperativity are all bimolecular / non-unit-stoichiometry and
cannot be expressed. Mass conservation is only trivially satisfied because every reaction is 1→1.
**Fix strategy.** Generalize reactions to reactant/product multisets with integer stoichiometric
vectors; compute mass-action propensities including combinatorial terms
(`a = k·∏ C(n_i, ν_i)` with proper volume scaling for order ≥2). Validate conservation against the
stoichiometric matrix nullspace.
**Verification.** Add SSA tests against analytic/ODE limits for `A+B→C` and `2A→B`; check the
conserved moiety totals are invariant along trajectories.

### F3 — Perturbation likelihood is inert under the kernel's own format 🔴 High
**Explanation.** `_likelihood_perturbation` (bayesian_updater.py:213–236) reads
`conditions['perturbation_source'/'perturbation_target']`. But `_execute_experiment` emits
perturbation experiments with `observations={'response':…}` and `conditions={'temperature':310}` —
those keys are absent, so the function always returns the `0.1` fallback.
**Empirical.** Likelihood = `0.1` for a kernel-format perturbation experiment regardless of mechanism
(`audit_findings.py` F3).
**Scientific consequence.** Perturbation is one of three experiment types selected uniformly at
random, so ≈⅓ of all "experiments" carry **zero discriminating information** yet are counted as
evidence-gathering steps. The acquisition loop believes it is learning when it is not.
**Fix strategy.** Either (a) make `_execute_experiment` emit the keys the likelihood expects, or
(b) define a perturbation likelihood over the actual observable (dose–response), and unit-test that
the producer and consumer share a schema. Long term, replace ad-hoc dict keys with typed
`Observation` objects (Phase 2 evidence objects).
**Verification.** Property test: for every experiment type the kernel can emit, the matching
likelihood must depend on at least one mechanism parameter (∂lik/∂θ ≠ 0 for some θ).
**✅ RESOLVED.** Both (a) and (b) implemented: `_execute_experiment` now emits
`perturbation_source/target`, and `_likelihood_perturbation` is a Gaussian over a topology-predicted
response (per-step signal attenuation, parallel paths combined as a union; no path ⇒ predicted 0).
Added a networkx-free `get_causal_paths` fallback so the channel works without the optional
dependency. Verified topology-dependent (path 0.135 vs no-path 3.7×10⁻⁶) and covered by
`xenon/tests/test_perturbation_likelihood.py` (producer/consumer schema contract + fallback).

### F4 — Likelihood averages χ², so evidence does not accumulate 🔴 Critical
**Explanation.** `_likelihood_concentration` returns `exp(−χ²/(2·n·scale))` (line 177) and
`_likelihood_kinetics` returns `exp(loglik/n_rates)` (line 210). Dividing by the measurement count
turns the joint likelihood into a **per-measurement geometric mean**.
**Empirical.** For `n = 1, 5, 20` identical 2σ residuals the likelihood is **0.1353 in all three
cases** (`audit_findings.py` F4) — it should shrink sharply with `n`.
**Scientific consequence.** A core promise of Bayesian sequential inference — that more data sharpens
the posterior — is broken. Ten consistent measurements count the same as one. This caps achievable
posterior contrast and makes "convergence" depend on noise realization rather than evidence volume;
it is a direct violation of the likelihood principle and of standard χ² (joint Gaussian) likelihoods.
**Fix strategy.** Use the joint likelihood `exp(−½ χ²)` with `χ² = Σ (oᵢ−pᵢ)²/σᵢ²` (no `1/n`).
Work in log-space and accumulate log-likelihoods to avoid underflow; carry a proper per-observation
noise model `σᵢ` (from the measurement, not `0.1·observed`).
**Verification.** Unit test: log-likelihood of `k` i.i.d. identical residuals = `k×` that of one;
posterior contraction rate ∝ data volume on a synthetic identifiable problem.

### F5 — Hypothesis regeneration resets accumulated posteriors 🔴 Critical
**Explanation.** `_generate_hypothesis_mechanisms` calls
`mechanism_prior.initialize_mechanism_priors(self.mechanisms[target.name])` over the **whole pool**
(xenon_kernel.py:225). `initialize_mechanism_priors` (mechanism_prior.py:229–236) **overwrites**
`mechanism.posterior` with the prior and renormalizes. Regeneration fires whenever `iteration==0 or
len(pool) < 5` (line 151) — i.e., every time pruning shrinks the pool below five.
**Empirical.** A mechanism with `posterior=0.97` becomes `1.0000` (its renormalized prior) after one
regeneration call (`audit_findings.py` F5).
**Scientific consequence.** Sequential evidence is periodically erased, making the posterior
non-monotone and path-dependent on when regeneration happens to trigger. The learning curve is not a
valid belief-update trajectory.
**Fix strategy.** Initialize priors **only for newly created mechanisms**; never touch the posterior
of mechanisms already carrying evidence. Keep a normalized mixture over the union (new priors scaled
into the residual mass). Add an invariant check that total posterior mass is conserved across a
generation step.
**Verification.** Regression test: inject an evidenced mechanism, trigger regeneration, assert its
posterior is unchanged up to the renormalization constant for genuinely new entries.

### F6 — Likelihood evaluated on a single transient SSA sample 🔴 Critical
**Explanation.** `_simulate_mechanisms` (xenon_kernel.py) runs **one** Gillespie trajectory to
`t_max=0.1 s` and overwrites each `state.concentration` with that single final value; the likelihood
then compares observations to this one noisy draw. There is no replication, no steady-state check,
and the predicted concentration is re-randomized every iteration.
**Scientific consequence.** `P(E|M)` is approximated by `P(E | one Monte-Carlo draw from M)`, so
simulation noise is indistinguishable from model–data mismatch. At `t=0.1 s`, slow mechanisms have
barely moved from the initial condition, so the "prediction" is an arbitrary transient, not a
model-characteristic observable. Combined with F1, the draw is also from the wrong dynamics.
**Fix strategy.** Define the observable explicitly (steady-state occupancy or a specified time
point), estimate it from **N replicate trajectories** with a Monte-Carlo standard error, and fold
that SE into the likelihood's noise term (`σ² = σ²_meas + σ²_MC`). For two-state systems prefer the
analytic master-equation steady state; reserve SSA for systems without closed forms. Cache the
predicted observable per (mechanism, condition) instead of mutating state in place.
**Verification.** Show likelihood variance across seeds collapses as `N→∞`; confirm predicted
observable matches the ODE/master-equation steady state within MC error.

### F7 — Mock experiment executor is mechanism-independent 🔴 High (acknowledged limitation, quantified)
**Explanation.** `_execute_experiment` draws observations from fixed Gaussians
(e.g. `N(80,10)`/`N(20,5)`) that do **not** depend on the mechanism or its parameters.
**Scientific consequence.** No ground-truth mechanism generates the data, so the "posterior" measures
which mechanism's randomized transient happened to land near a random observation — confirmation of
noise, not inference. This is the gap Phase 2 exists to close, but it must be stated plainly: **no
biological claim is currently supported.**
**Fix strategy.** Phase 2 — replace with (a) real evidence objects ingested from databases, and/or
(b) a held-out *in silico* ground-truth mechanism for closed-loop validation (simulate from a known
M*, verify XENON recovers it). The latter is buildable in-repo immediately and is the right first
validation harness.
**Verification.** Closed-loop recovery test: generate data from a known M*, require the posterior to
concentrate on M* as data grows (and to stay diffuse when data are uninformative).

### F8 — Prior is mislabeled and partly constant 🟡 Medium
**Explanation.** `_conservation_prior` returns `0.7` if `is_thermodynamically_feasible()` else `0.3`
— a **thermodynamics re-read mislabeled as evolutionary conservation**, double-counting a signal that
is also a hard validation gate. `_literature_prior` reads an empty `_literature_db`, so it is the
constant `0.1` for every mechanism.
**Scientific consequence.** Two of three "independent" prior channels are either redundant with the
thermodynamic gate or contribute no information, while the documentation implies homology/citation
evidence is in play. Prior mass is not what it claims to be.
**Fix strategy.** Phase 2 — wire real conservation (UniProt/Pfam homology) and citation counts
(PubMed), keep channels statistically independent, and document the exact functional form and units.
Until then, label them honestly as placeholders and exclude double-counted signals.
**Verification.** Ablation: prior rankings must change when literature/conservation data are supplied
and be invariant to the thermodynamic gate (no double counting).

### F9 — Convergence ≠ correctness; no identifiability 🟡 Medium→High
**Explanation.** Convergence is `entropy < 0.1` over the (improper) posterior. There is no structural
or practical identifiability analysis. For a two-state reversible system the steady-state split is set
by `K = k_f/k_r` alone, so the individual rate constants are **non-identifiable** from a steady-state
concentration measurement.
**Scientific consequence.** The platform can report high posterior confidence in a mechanism whose
parameters are not estimable from the data used — exactly the failure mode reviewers probe. Duplicate
mutant mechanisms (topologically identical to templates) also split/inflate posterior mass.
**Fix strategy.** Phase 5 — profile-likelihood and Fisher-information identifiability, posterior
covariance / sloppiness eigenanalysis, and explicit reporting of non-identifiable directions. Add
mechanism de-duplication via `compute_mechanism_hash()` before posterior bookkeeping.
**Verification.** On a known non-identifiable pair, require the report to flag the degenerate
direction rather than concentrating posterior on one arbitrary representative.

### F10 — Kinetics likelihood key depends on state names 🟡 Medium
**Explanation.** Kinetics observations are keyed `"{source}->{target}"`; topology mutants whose states
are renamed by `mutate_topology` won't match and fall through to the `0.1` fallback, so kinetics
evidence silently applies only to template-shaped mechanisms.
**Fix.** Key observables by stable identifiers (mechanism-relative reaction IDs), not display names;
test across mutated topologies. **Verify** that a renamed-but-equivalent reaction still receives the
kinetics likelihood.

### F11 — `reactions_per_second` is a placeholder 🟢 Low
`get_performance_metrics` returns `reactions_per_second == total_reactions` (gillespie.py:253) — a
count, not a rate. The performance-gate narrative should not cite it as throughput. **Fix:** divide by
measured wall-time (the campaign driver already times this correctly and should be the source of truth).

### F12 — Missing runtime imports (`Optional`, `List`) 🟢 Low
`bayesian_updater.py` and `mechanism_prior.py` annotate with `Optional`/`List` but import neither;
this only survives because `from __future__ import annotations` makes annotations strings (PEP 563).
It breaks under `typing.get_type_hints()` / Pydantic / runtime introspection. **Fix:** import them (or
use `X | None` / `list[...]`). **Verify:** `python -c "import typing; typing.get_type_hints(...)"`.

---

## Cross-cutting scientific assessment

- **Is the posterior a valid posterior?** No — F4 (no evidence accumulation) and F5 (periodic reset)
  break the update; F6/F1 feed it predictions from the wrong, noisily-sampled dynamics; F7 means
  there is no generative ground truth. The number is reproducible (determinism) but not meaningful.
- **Data leakage / confirmation bias.** With mechanism-independent mock data (F7), the loop cannot
  leak real signal because there is none; the risk inverts in Phase 2, where the same prior sources
  (literature) must not also be used as evidence. Flag now, enforce later.
- **Statistical power.** Capped by F4 regardless of how much data Phase 2 supplies — **fix F4 before
  connecting databases**, or more data will not help.

---

## Prioritized roadmap for Phases 2–10 (honest scope)

**Tier 0 — correctness prerequisites (in-repo, no external data; do these first).**
Fix F1, F2, F4, F5, F6, then F3/F10. Add the closed-loop *in-silico* recovery harness (F7 fix-b).
Without Tier 0, Phases 2–10 build on an invalid kernel. *These are the changes I recommend
implementing next and can do here.*

**Tier 1 — inference & identifiability (Phases 4–5, in-repo).** Pluggable likelihood/inference
backends (start: exact enumeration for small mechanism sets, then SMC/ABC for parameter posteriors);
profile-likelihood + Fisher-information identifiability; posterior-predictive checks.

**Tier 2 — real evidence (Phase 2, requires network + data engineering).** Typed `Evidence` objects
(source, accession, version, timestamp, confidence, citation, quality) and adapters for
UniProt/Reactome/KEGG/PubMed/ChEMBL/OpenTargets/GEO/PRIDE/DepMap/TCGA/… **Not possible in this
sandbox** (no outbound access to these resources, and several require credentialed/controlled access,
e.g. TCGA/CPTAC). Deliverable here would be the interface + one mock-backed reference adapter + a
contract test; live wiring belongs in an environment with the right network and data-use approvals.

**Tier 3 — active design, external/clinical validation, stress, hardening, publication
(Phases 6–10).** Optimal experimental design (EIG/BOED), held-out and cross-lab validation,
adversarial stress, CI/Docker/property-fuzz-mutation testing, and the publication package. These
depend on Tiers 0–2 and on data/lab resources outside this environment; scoped here, not faked.

---

## What I am explicitly **not** claiming

- No biological conclusion about EGFR is supported by the current system (F7).
- No external validation has been performed (no network/data access here).
- The 0.9986 posterior is an artifact, not evidence — see F4/F5/F6.
- No database integration, inference backend, or identifiability suite has been *implemented* in this
  pass; this document is the audit (Phase 1) plus a scoped plan, with all defects reproduced by
  `audit_findings.py`.

## Reproduce

```bash
pip install numpy scipy click
PYTHONPATH=$PWD python xenon_campaigns/audit/audit_findings.py   # 6/6 defects confirmed
```
