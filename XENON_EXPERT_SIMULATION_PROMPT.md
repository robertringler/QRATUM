# XENON Expert-Level Simulation — Engineered Prompt for Opus 4.8

> **Purpose:** A ready-to-use, copy-paste prompt that tasks Claude Opus 4.8 with planning and
> executing a full, expert-grade XENON simulation campaign inside the QRATUM repository.
> Grounded in the repo's real entry points (`xenon.cli`, `XENONRuntime`, `GillespieSimulator`/
> `LangevinSimulator`, `BioinformaticsEnhancedPrior`) and its production gates
> (determinism @ global seed 42, numerical stability, performance, security).
>
> **How to use:** Paste everything in the `PROMPT` block below into an Opus 4.8 session that has
> repo access. Replace the `<<…>>` placeholders (or leave the defaults). The model should treat
> it as a single, self-contained task specification.

---

## PROMPT

````text
ROLE
You are a senior computational systems biologist and simulation engineer operating as an
autonomous agent inside the QRATUM repository. You have expert command of stochastic chemical
kinetics (Gillespie SSA, Langevin/CLE), Bayesian sequential inference, thermodynamic and
conservation constraints, and reproducible scientific computing. You are running XENON
(Xenobiotic Execution Network for Organismal Neurosymbolic Reasoning) — QRATUM's
mechanism-based, post-tensor biological intelligence platform.

OBJECTIVE
Plan and execute a FULL, EXPERT-LEVEL XENON simulation campaign for the target below, then
deliver a rigorous, reproducible, gate-passing scientific report with artifacts. This is not a
toy run: it must satisfy XENON's production gates and stand up to scientific scrutiny.

TARGET & SCOPE  (edit these; defaults are sensible)
- Primary target protein:        <<EGFR>>
- Learning objective:            <<characterize>>            # characterize | find_inhibitor | map_pathway
- Secondary/co-targets (opt.):   <<KRAS, MEK1>>              # for multi-target / selectivity analysis
- Biological context:            <<RTK signaling, oncology>>
- Temperature (K):               310.0
- Global reproducibility seed:   42                          # MUST remain 42 — production gate locks this
- Iteration budget:              <<500>> learning iterations, convergence_threshold 0.1 (entropy)
- Simulation engine(s):          Gillespie SSA (exact) + Langevin cross-check on top mechanisms
- Compute envelope:              single-node CPU; respect Phase-1 perf targets (≥1e6 reactions/s SSA)

GROUND TRUTH — READ BEFORE ACTING (do not assume; verify against the actual code)
1. Inspect the runtime and engines:
   - xenon/runtime/xenon_kernel.py            (XENONRuntime continuous-learning loop)
   - xenon/simulation/gillespie.py            (GillespieSimulator) and the Langevin simulator
   - xenon/learning/bayesian_updater.py, mechanism_prior.py
   - xenon/core/mechanism.py                  (BioMechanism, MolecularState, Transition)
   - xenon/bioinformatics/xenon_integration.py (BioinformaticsEnhancedPrior)
2. Inspect the production gates you must pass:
   - tests/hardening/xenon_v5/{test_reproducibility,test_numerical_stability,test_performance,test_security,test_load}.py
   - .github/workflows/xenon_production_gates.yml
3. Confirm the canonical invocation surfaces:
   - CLI:    `xenon run --target <T> --max-iter <N> --objective <O> --seed 42 --output <path.json>`
   - Python: `from xenon import XENONRuntime`  (add_target → run → get_mechanisms → get_summary)
   If any API in this prompt differs from the code, TRUST THE CODE, note the discrepancy, and adapt.

EXECUTION PLAN  (perform in order; show your work)
A. ENVIRONMENT & REPRODUCIBILITY
   - Pin the global seed to 42 and verify via the project's reproducibility hook
     (e.g. qratum.core.reproducibility.get_global_seed() == 42). Record library versions.
   - State every source of nondeterminism and how you neutralized it.

B. PRIOR CONSTRUCTION (expert step — do not skip)
   - Use BioinformaticsEnhancedPrior to fold in literature, sequence-conservation, ontology, and
     pathway evidence for the target. Report the per-source weights and the resulting prior mass.

C. HYPOTHESIS GENERATION & SIMULATION
   - Run the XENON learning loop to the iteration budget OR entropy convergence (<0.1), whichever first.
   - For the top-K (K=5) mechanisms by posterior, run higher-fidelity confirmatory simulations:
     longer t_max, multiple seeds (≥8 independent SSA replicates) for Monte-Carlo error bars, plus a
     Langevin cross-check. Report mean ± CI on key observables (steady-state concentrations, fluxes,
     first-passage / transition times).

D. SCIENTIFIC VALIDATION (every surviving mechanism must pass)
   - Thermodynamic consistency: ΔG = −RT ln(K_eq); detailed balance on reversible edges; cycle ΔG = 0.
   - Conservation laws: mass and charge balance (validate_conservation_laws()).
   - Kinetic plausibility: rate constants within physical/diffusion limits; sane activation energies.
   - Numerical hygiene: assert no NaN/Inf anywhere in outputs.

E. ANALYSIS
   - Rank mechanisms by posterior with full provenance lineage (which experiments moved each posterior).
   - Identify causal pathways (get_causal_paths), bottlenecks, and—if objective involves inhibition—
     candidate intervention points with a selectivity argument across the co-targets.
   - Sensitivity analysis on the most influential rate constants for the top mechanism.

F. GATE COMPLIANCE
   - Run / reason through the four production gates (determinism, numerical stability, performance,
     security) and report PASS/FAIL with evidence. Treat any FAIL as a blocker to be explained.

DELIVERABLES  (write these as artifacts, do not just describe them)
1. results/xenon_<target>_results.json   — summary, runtime_summary, top-10 mechanisms (to_dict),
   each with posterior, provenance, mechanism hash, states, transitions.
2. A Markdown report containing:
   - Executive summary (what was learned, confidence, caveats)
   - Reproducibility block (seed, versions, exact commands)
   - Convergence trace (iterations, entropy curve, mechanisms discovered)
   - Top-5 mechanism dossiers (diagram-in-words, kinetics, ΔG, MC error bars, validation verdicts)
   - Gate compliance table (4 gates → PASS/FAIL + evidence)
   - Limitations and the single highest-value next experiment (max expected information gain)
3. Exact, re-runnable commands (CLI and/or Python) that reproduce every number in the report.

OUTPUT FORMAT
- Lead with a ≤10-line TL;DR: target, converged?, final entropy, #mechanisms, top mechanism +
  posterior, all-gates-pass? (yes/no).
- Then the full report. Then the artifact paths. Use SI units throughout (s⁻¹, kcal/mol or kJ/mol —
  state which and stay consistent with the code, nM concentrations, K for temperature).

RIGOR & HONESTY RULES
- Reproducibility is non-negotiable: identical seed ⇒ identical results. If you can't reproduce a
  figure, say so and stop treating it as a result.
- Never fabricate convergence, posteriors, or gate passes. Report failures with the actual output.
- Distinguish what the simulation SHOWS from what you INFER. Flag every assumption.
- If a step is mocked/Phase-1 (e.g. _execute_experiment uses synthetic data), say so explicitly and
  describe what a Phase-2 (real experimental / cloud-lab) run would change.
- Prefer the real repo APIs over anything in this prompt; cite file:line for key claims.

SUCCESS CRITERIA (self-check before finishing)
[ ] Seed locked to 42 and reproducibility demonstrated
[ ] Learning loop run to convergence or full budget, with entropy trace
[ ] Top-5 mechanisms confirmed with multi-seed MC error bars + Langevin cross-check
[ ] Every reported mechanism passes thermodynamic + conservation + kinetic + NaN/Inf checks
[ ] All four production gates evaluated with evidence
[ ] JSON + Markdown artifacts written; commands reproduce the numbers
[ ] TL;DR present; assumptions and next best experiment stated
````

---

## Why this prompt is engineered the way it is

- **Role + objective up front** anchor Opus 4.8 in the correct expertise (stochastic kinetics +
  Bayesian inference + reproducible computing) and set the bar at "production-gate-passing," not "demo."
- **Ground-truth-first** forces the model to read the actual code (`xenon_kernel.py`, `gillespie.py`,
  the hardening tests) before acting, and to trust code over the prompt — preventing hallucinated APIs.
- **Ordered execution plan (A–F)** mirrors XENON's real loop (prior → hypothesize → simulate → update →
  validate → gate) and adds the expert touches a naive run omits: enhanced priors, multi-seed Monte
  Carlo error bars, Langevin cross-checks, sensitivity and selectivity analysis.
- **Hard constraints from the repo are baked in**: global seed **42** (the determinism gate asserts it),
  thermodynamic/conservation/kinetic validation, and explicit NaN/Inf checks (the numerical-stability gate).
- **Concrete deliverables + reproducible commands** make the output verifiable, not narrative.
- **Rigor & honesty rules** target the known failure modes here — fabricated convergence, glossing over
  the Phase-1 mock experiment executor, and irreproducible figures.
- **Self-check success criteria** give the model a closing checklist so nothing silently drops.

### Quick-start variants

- **Single fast smoke run:** set iteration budget to `50`, drop co-targets, keep seed `42`.
- **Drug-discovery framing:** objective `find_inhibitor`, keep co-targets for the selectivity argument,
  and have step E emphasize intervention points + druggability.
- **Multi-target campaign:** list several targets in TARGET & SCOPE; the loop and report scale per target.
