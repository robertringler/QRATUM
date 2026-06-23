# XENON — Technical Whitepaper

**Scope:** The `xenon/` subsystem of the QRATUM repository.
**Method:** Evidence-based. Every capability claim below is corroborated by source inspection and live test execution (Python 3.11, numpy/scipy present). Documentation is cited only where it matches implementation. Confidence is marked High (executed/observed), Medium (read + corroborating signal), or Low (single-source).
**Snapshot:** branch `claude/qratum-forensics-audit-aoYX2`.

---

## 1. Executive Summary

XENON is, in substance, a **classical computational-biology library**: stochastic chemical-kinetics simulators, a mechanism-graph (DAG) data model with Bayesian bookkeeping, sequence-alignment dynamic programming, information-theoretic dependence measures, and a molecular-dynamics/docking toolkit. The genuinely working core is real, numerically sound, and test-backed.

Its documentation frames this as something much larger — *"XENON: Xenobiotic Execution Network for Organismal Neurosymbolic Reasoning … a post-GPU biological intelligence platform that replaces tensor-based approaches with mechanism-based continuous learning"* (`xenon/README.md`). That framing is **aspirational marketing**: the "quantum-enhanced" alignment runs a classical algorithm by default, and there is no learned/neural component beyond Bayesian posterior updates over hand-built mechanism graphs.

**Size (High):** 76 Python files, **22,487 LOC** across seven subpackages.
**Test health (High):** **161 passed / 61 failed** across the xenon test corpus. The failures are concentrated in two subsystems (`adapters/`, `molecular_dynamics_lab/`) and are caused by **API drift between tests and implementation**, not by absent functionality.

**Verdict:** A **functional prototype** with a solid, correct simulation/bioinformatics core, surrounded by over-stated documentation and two subsystems whose test/implementation contracts have diverged.

---

## 2. Module Census

| Subpackage | Files | LOC | Substance | Test status |
|---|---|---|---|---|
| `simulation/` | 3 | 505 | Gillespie SSA + Langevin SDE | **9/9 pass** |
| `core/` | 3 | 388 | `BioMechanism` DAG, `MolecularState`, `Transition` | **18/18 pass** |
| `bioinformatics/` | 34 | 10,889 | alignment, information theory, inference, genome sequencing | mostly passing (see §6) |
| `molecular_dynamics_lab/` | 16 | 6,711 | MD integrators, Lennard-Jones, PDB loader, docking, viewers | **mostly failing** (API drift) |
| `adapters/` | 4 | 589 | mechanism→visualization adapter | **11/13 fail** (stale API) |
| `learning/` | 3 | 541 | Bayesian updater, mechanism mutation/recombination | passing where exercised |
| `runtime/` | 2 | 462 | execution/runtime glue | n/a |

Public surface (`xenon/api.py`, High): `create_mechanism`, `simulate_mechanism`, `run_xenon`, `validate_mechanism`, `compute_mechanism_prior`, `update_mechanism_posterior`, `mutate_mechanism`, `recombine_mechanisms`.

---

## 3. Verified Capabilities (real, test-backed)

### 3.1 Stochastic chemical kinetics — Gillespie SSA (High)
`xenon/simulation/gillespie.py` implements a **textbook-correct Gillespie Stochastic Simulation Algorithm**:
- Exact exponential waiting time: `tau = -np.log(self._rng.random()) / total_propensity` (line 119).
- Propensity computation `propensity = transition.rate_constant * source_count` (line 190).
- Propensity-weighted reaction selection via cumulative sum (`_select_reaction`, lines 196–217).
- Seeded `np.random.default_rng(seed)` for reproducibility.

This is a genuine implementation of Gillespie (1977) direct-method SSA, not a mock. **9/9 `test_gillespie.py` tests pass.**

### 3.2 Langevin stochastic dynamics (Medium-High)
`xenon/simulation/langevin.py` (`LangevinSimulator`) integrates the SDE `dX/dt = drift(X) + sqrt(diffusion(X))·noise(t)` with Euler–Maruyama style updates: `state[species] += drift*dt + stochastic*np.sqrt(dt)` (line 108), thermal-noise injection, seeded RNG. A legitimate, if simplified, chemical-Langevin integrator.

### 3.3 Mechanism graph data model (High)
`xenon/core/mechanism.py` (`BioMechanism`) is a directed mechanism graph over `MolecularState` (name, molecule, properties, concentration, free_energy) and `Transition` (source, target, rate_constant, activation_energy, reversible). It carries Bayesian bookkeeping fields — `posterior` (init 1.0), `provenance: list[str]`, `confidence_intervals` — with `add_state`/`add_transition` chaining validation, thermodynamic-feasibility checks, optional NetworkX backing, and `to_dict`/`from_dict` serialization. **18/18 `test_mechanism.py` tests pass.**

### 3.4 Sequence alignment (High, with a caveat)
`xenon/bioinformatics/quantum_alignment.py` (`QuantumAlignmentEngine`) advertises "quantum-enhanced alignment," but the operative method `align_classical` (line 187) is a **standard Needleman–Wunsch dynamic-programming global alignment**: a full `(m+1)×(n+1)` score matrix with match/insert/delete recurrence `score_matrix[i][j] = max(match, delete, insert)` (lines 209–227) and traceback. **The quantum path is disabled by default** — `AlignmentConfig.enable_quantum = False` ("Classical by default for stability", line 60). **15/15 `test_quantum_alignment.py` tests pass** — i.e., they exercise the classical DP.

> Caveat (Medium): the "quantum" branding describes an aspirational alternative path; the verified, tested behavior is classical DP with a documented "classical–quantum equivalence" tolerance that, in practice, validates the classical result against itself.

### 3.5 Information theory (High)
`xenon/bioinformatics/xenon/omics/information_engine.py` (`InformationEngine`) computes mutual information, Shannon entropy, joint entropy, and transfer entropy via histogram estimators, with NaN input validation that raises `ValueError`. **18/18 information-fusion tests and all 26 `tests/hardening/xenon_v5` hardening tests pass.**

> Note (High): joint-entropy estimation contained a real defect — `np.histogram2d(..., bins="auto")` (which `histogram2d`, unlike 1-D `histogram`, does not accept) raised a numpy `UFuncTypeError` for all valid inputs. This was fixed during audit (per-axis `np.histogram_bin_edges`), restoring the entire MI/entropy path.

### 3.6 Neuro-symbolic inference & genome sequencing (High)
`bioinformatics/inference/neural_symbolic.py` (`NeuralSymbolicReasoner`) — **8/8 pass**. The genome-sequencing pipeline — **17/17 `test_full_genome_sequencing.py` pass**. Sequence analyzer — **10/11 pass**.

### 3.7 CLI (High)
`xenon/cli.py` exposes a `click` command group (`run`, `query`, `validate`) and an argparse `main` entry point with a lightweight demonstration mechanism model. **4/4 `test_cli.py` pass** (after a demo-model/API correction made during audit).

---

## 4. Real Implementations with Failing Tests (API drift)

These subsystems contain substantive code but **fail their tests because the test contract and the implementation have diverged** — the dominant defect class across this repository.

### 4.1 `adapters/bio_mechanism_adapter.py` — 11/13 fail (High)
The adapter and its tests were written against an **older mechanism API**: they reference `MolecularState.state_id` / `.protein_name`, `Transition.delta_g`, and a `BioMechanism(mechanism_id=, states=, transitions=, evidence_score=)` positional/keyword constructor. The current `core.mechanism` types use `name`/`molecule`/`source`/`target` and `BioMechanism(name)` + `add_state`/`add_transition`. Observed failures: `TypeError: BioMechanism.__init__() takes 2 positional arguments but 4 were given`, `AttributeError: 'MolecularState' object has no attribute 'state_id'`, `'Transition' object has no attribute 'delta_g'`. (The CLI shared this drift and was repaired with a local demo model during audit; the adapter remains on the old contract.)

### 4.2 `molecular_dynamics_lab/` — md_simulator 16 fail, pdb_loader 15 fail (High)
The implementation is **genuine MD**: `md_simulator.py` defines Verlet / velocity-Verlet integrators, `_compute_forces`, kinetic/potential/total energy; `docking_engine.py` implements a Lennard-Jones potential (`_lennard_jones`, line 350) for van-der-Waals scoring. The failures are constructor/method drift: `TrajectoryFrame.__init__() got an unexpected keyword argument 'velocities'`, `SimulationState.__init__() got an unexpected keyword argument 'total_energy'`, `MDConfig.__init__() got an unexpected keyword argument 'thermostat_coupling'`, `'MDSimulator' object has no attribute 'set_masses'`, and a recurring `ValueError: Position array shape mismatch`. So: real physics, drifted API.

### 4.3 `test_integration.py` — 5/6 fail (Medium)
Integration failures chain off the adapter drift (§4.1).

---

## 5. Architecture Assessment

The actual architecture (independent of README claims) is a layered classical pipeline:

```
mechanism (core: BioMechanism DAG)
   ├─ simulate ──► simulation/ (Gillespie SSA, Langevin SDE)
   ├─ analyze  ──► bioinformatics/ (alignment DP, information theory, inference, genomics)
   ├─ learn    ──► learning/ (Bayesian posterior update, mutation, recombination)
   ├─ visualize► adapters/ (mechanism → viz model)   [drifted]
   └─ MD       ──► molecular_dynamics_lab/ (Verlet, LJ, PDB, docking) [drifted]
```

There is **no GPU/tensor stack, no quantum hardware path, and no gradient-based neural learning** present — consistent with the README's *stated* "post-GPU / not-tensors / not-gradient-descent" philosophy, but inconsistent with the "quantum" and "biological intelligence platform" framing. The "learning" is Bayesian updating over hand-specified mechanism graphs (`update_mechanism_posterior` → `BayesianUpdater.update_mechanisms`), which is real but modest.

---

## 6. Testing Assessment (High)

| Area | Result |
|---|---|
| simulation (Gillespie) | 9/9 ✅ |
| core (mechanism) | 18/18 ✅ |
| bioinformatics: quantum_alignment | 15/15 ✅ |
| bioinformatics: information_fusion | 18/18 ✅ |
| bioinformatics: neural_symbolic | 8/8 ✅ |
| bioinformatics: genome sequencing | 17/17 ✅ |
| bioinformatics: sequence_analyzer | 10/11 |
| hardening (xenon_v5) | 26/26 ✅ |
| cli | 4/4 ✅ |
| adapters | 2/13 ❌ |
| molecular_dynamics_lab (md_simulator) | 2/18 ❌ |
| molecular_dynamics_lab (pdb_loader) | 1/16 ❌ |
| integration | 1/6 ❌ |

**Aggregate: 161 passed / 61 failed.** Behavioral coverage of the **core** (simulation, mechanism, information theory, alignment) is strong and genuinely validates correctness. The failing 61 are localized to the adapter/MD subsystems and reflect contract drift, not missing capability.

---

## 7. Documentation Accuracy (Medium-High)

- **Over-statement:** "post-GPU biological intelligence platform," "quantum-enhanced," "neurosymbolic reasoning" overstate a classical bioinformatics library. The quantum alignment path is off by default and the tested path is classical DP.
- **Stale install instructions:** `README.md` directs users to `git clone https://github.com/robertringler/Qubic.git` — a different/older repository name than the current `QRATUM`, so the documented install path is broken (Medium).
- **Accurate where it counts:** the README's description of the *computational primitive* (mechanism DAGs), *Bayesian updating*, and *provenance* matches `core/mechanism.py` and `api.py`.

---

## 8. Strengths

1. **Correct, reproducible stochastic simulation** — a real Gillespie SSA and Langevin integrator with seeded RNG; the strongest, best-tested part of XENON.
2. **A clean, validated mechanism data model** with serialization, thermodynamic checks, and Bayesian provenance fields — 18/18 tests.
3. **Substantial, mostly-passing bioinformatics layer** (~10.9 KLOC): alignment DP, information theory (MI/entropy/transfer entropy), genome sequencing, neuro-symbolic inference.

## 9. Weaknesses / Risks

1. **API drift** between tests and implementation in `adapters/` and `molecular_dynamics_lab/` (61 failures) — the same merge/rename-corruption pattern seen repo-wide; the MD physics is real but unreachable through its current tests.
2. **Marketing–implementation gap** — "quantum" and "biological intelligence platform" claims are not realized; risk of misleading evaluators.
3. **Latent correctness bugs surface only on execution** — e.g., the `histogram2d(bins="auto")` defect silently broke all mutual-information computation until run; suggests uneven CI exercise of the bioinformatics layer.
4. **Broken documented install path** (wrong clone URL).

---

## 10. Production Readiness (0–10 per dimension, evidence-based)

| Subsystem | Impl | Test | Maintainability | Doc accuracy |
|---|---|---|---|---|
| simulation (Gillespie/Langevin) | 7 | 8 | 7 | 6 |
| core (mechanism) | 7 | 8 | 7 | 7 |
| bioinformatics (align/info/genomics) | 6 | 7 | 6 | 4 |
| adapters | 4 | 2 | 4 | 4 |
| molecular_dynamics_lab | 5 | 2 | 5 | 4 |
| learning | 5 | 5 | 5 | 5 |

**XENON readiness ≈ Functional Prototype.** The simulation/mechanism/information core is **pre-production**; the adapter and MD subsystems are **prototype-with-drift**; the "quantum/AI platform" narrative is **aspirational**.

---

## 11. Final Findings

1. **What XENON actually is:** a classical computational-biology library — Gillespie/Langevin kinetics, a Bayesian mechanism-DAG model, sequence-alignment DP, information theory, and an MD/docking toolkit.
2. **What works today (tested):** stochastic simulation, the mechanism model, alignment, information theory, genome sequencing, neuro-symbolic inference, the CLI — **161 passing tests**.
3. **What is drifted (real code, failing tests):** the visualization adapter and the molecular-dynamics lab (**61 failures**), due to test/implementation API divergence.
4. **What is aspirational:** "quantum-enhanced" alignment (classical by default), and the "post-GPU biological intelligence platform" framing.
5. **To reach pre-production:** reconcile the `adapters/` and `molecular_dynamics_lab/` test/implementation contracts; fix the documented install URL; align README claims with the classical reality; and broaden CI so latent numerical defects (like the `histogram2d` bug) are caught before release.

*All conclusions rest on executed tests and direct source inspection; where a capability could not be exercised it is stated as such, and no claim is taken from documentation without code corroboration.*
