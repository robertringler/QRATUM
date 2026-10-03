# FLUXA / QRATUM End-to-End Energy-System Simulation

**An executed experiment on the QRADLE substrate, with its negative results.**

| | |
|---|---|
| Vertical | FLUXA (energy systems), implemented at `fluxa/` |
| Substrate | QRADLE 1.0.0 (`qradle/`), pre-existing and unmodified |
| Compute | CPU only — 4 logical cores, `Linux-6.18.44-fc-v64-x86_64`, CPython 3.11.15, NumPy 2.4.6, SciPy 1.17.1 |
| Campaign | `python -m fluxa.experiments.run_all` — 361.3 s wall, 8 stages, **0 failed stages**, 73 artefact files (20 MB) in `results/fluxa/` |
| System config hash | `b3fc57aaf4e8aaf02b370953789058dc03ec27b545907f2f015f0a75bd31d8f7` |
| Model hash | `fac1fd29898a3dbc9547280635762a556435afe641833b817b100e6cf2ea59d9` |
| FLUXA test suite | 297 tests, all passing |
| Quantum | **No quantum acceleration was demonstrated in this experiment.** |

Every claim below is tagged. `[OBSERVED]` = directly measured by the executed
campaign. `[DERIVED]` = computed from observed results. `[ASSUMED]` = a
modelling assumption. `[IMPLEMENTED]` = demonstrated by executable code.
`[NOT IMPLEMENTED]` = discussed architecturally, not demonstrated.
`[HYPOTHESIS]` = requires further research.

---

## 1. Executive finding

A real, executable energy-system simulation now runs as a QRADLE domain
contract, and it was attacked. What survived:

1. **Determinism is real at the FLUXA level, after fixing three defects.**
   Three replays of each of three scenarios (288 timesteps each) produced
   byte-identical per-timestep state hashes, event sequences, Merkle roots,
   metric hashes and provenance identities. `[OBSERVED]`
   Getting there required fixing three determinism defects that a
   less adversarial test would have missed — two in FLUXA's own provenance
   design, one in QRADLE (§9).

2. **One QRADLE determinism defect remains, and is reported rather than
   worked around.** `qradle.core.merkle.MerkleChain.append` stamps every node
   with `datetime.now(timezone.utc)`. Three identical runs produced **three
   different** QRADLE chain roots. `[OBSERVED]` Any claim that QRADLE chain
   roots are reproducible across executions is false as the code stands.

3. **Tamper evidence works — and QRADLE's own Merkle proof does not.**
   FLUXA detected every event, state and input modification attempted
   (8 of 20 attacks DETECTED, 7 REJECTED, 3 CONSTRAINED). But a
   `qradle.core.merkle.MerkleProof` carrying a **fabricated event hash and a
   garbage proof path** verified successfully, because
   `MerkleProof.verify()` is `self.root_hash == claimed_root` and never
   recomputes a root. `[OBSERVED]` FLUXA therefore does not rely on it.

4. **Rollback genuinely restores state and resumes deterministically.** In
   three independent experiments (including one during an active compound
   disturbance), an injected conservation-law violation was detected,
   rollback was refused without authorization, the checkpoint verified, the
   restored state hash matched the recorded one exactly, and the resumed
   trajectory was **bit-identical** to the uncorrupted reference.
   `[OBSERVED]`

5. **Authorization gating works mechanically but is not authentication.**
   SENSITIVE scenarios are refused before any physics runs. But
   `RunConfig.authorized` is a boolean the caller asserts: attack
   `E3_SELF_ASSERTED_AUTHORIZATION` **succeeded**. There is no signature, no
   credential, no second party. `[OBSERVED]`

6. **Auditability is cheap.** The full provenance layer — 529 events plus 12
   checkpoints over 288 timesteps — costs **2.49%** of wall time
   (0.037 s on 1.474 s) and produces bit-identical physics. `[OBSERVED]`

7. **The base-case economic result is visibly suboptimal, and the report
   says so.** Changing one documented modelling parameter
   (`stored_energy_value_usd_per_mwh`, 48 → 60) reduces operating cost by
   **1.25%** and triples battery cycling on the identical physical system.
   `[OBSERVED]` The base case's operating cost is therefore **not** an
   optimum and must not be read as one (§8.4).

8. **No quantum computation occurred.** `[OBSERVED]`

The honest summary: QRADLE delivers deterministic execution, auditable
events, working rollback and a functioning authorization *gate*, at
negligible cost. It does not deliver authentication, and two of its own
cryptographic components (the Merkle proof, the wall-clock chain) do not do
what their names suggest.

---

## 2. System architecture

### 2.1 Layering as executed

```
FLUXA  (new; fluxa/, 18 modules + 3 experiment drivers + 15 test modules, 10,800 lines)
  model.py       frozen, hashable physical + economic description
  network.py     linearised DC power flow → PTDF
  profiles.py    seeded, reproducible load / solar / wind drivers
  scenarios.py   declarative perturbations + safety classification
  dispatch.py    receding-horizon economic dispatch LP (SciPy/HiGHS)
  state.py       machine-readable timestep state + operating-state classification
  events.py      FLUXA event taxonomy + deterministic Merkle ledger
  provenance.py  binary Merkle tree, recomputing inclusion proofs, bundle
  engine.py      the QRADLE-integrated run loop
  recovery.py    checkpoint / detect / authorize / restore / verify / resume
  metrics.py     reliability, economic, renewable, resilience metrics
  montecarlo.py  seeded stochastic campaign
  benchmark.py   runtime, scaling, parallel, provenance overhead
  io.py          JSON / CSV / Parquet artefacts
  cli.py         command-line entry points
        │
        ▼
QRADLE  (pre-existing, unmodified)
  DeterministicEngine.execute_contract   wraps the whole trajectory
  FatalInvariants.enforce_human_oversight   the authorization gate
  MerkleNode                             the event-hashing primitive FLUXA reuses
  RollbackManager                        checkpoint creation, verification, restore
        │
        ▼
Q-Substrate
  CPU: exercised.  GPU / HPC / QPU: NOT exercised.
```

### 2.2 What FLUXA reuses versus what it had to build

| QRADLE capability | Status in repository | How FLUXA uses it |
|---|---|---|
| `DeterministicEngine.execute_contract` | `[IMPLEMENTED]` | The entire trajectory computes inside one contract execution, inheriting invariant enforcement, `output_hash` and a contract checkpoint. |
| `FatalInvariants.enforce_human_oversight` | `[IMPLEMENTED]` | Called *before* any physics; a SENSITIVE scenario without authorization raises `InvariantViolation` and the run never starts. |
| `MerkleNode` (SHA-256 content hashing) | `[IMPLEMENTED]` | Reused byte for byte as the event-hashing primitive, with the **simulation** timestamp substituted for the wall clock. |
| `MerkleChain` | `[IMPLEMENTED]` but wall-clock stamped | Written to by QRADLE automatically; its root is excluded from FLUXA's provenance identity and the non-determinism is reported. |
| `MerkleProof` | `[IMPLEMENTED]` but **not load-bearing** | Not used. Root-equality only; forgeable (§10, H1). |
| `RollbackManager` | `[IMPLEMENTED]` | All FLUXA checkpoints and restores go through it. `Checkpoint.verify()` recomputes the state hash and works correctly. |
| Safety levels ROUTINE→EXISTENTIAL | `[IMPLEMENTED]` | FLUXA maps timestep conditions onto four of the five; EXISTENTIAL is never emitted. |
| Security zones Z0–Z3, dual control | `[IMPLEMENTED]` in `qradle/core/zones.py` | **Not wired into `DeterministicEngine`.** `ZoneContext.has_dual_control` exists and is unused by the authorization path — which is exactly why E3 succeeds. `[NOT IMPLEMENTED]` as an integration. |
| Energy-system model, power flow, dispatch | **absent from the repository** | Built from scratch in `fluxa/`. |
| Binary Merkle tree + recomputing inclusion proof | **absent** | Built in `fluxa/provenance.py`. |
| GPU / HPC substrate execution | abstractions exist in `qratum/platform/substrates.py` | `[NOT IMPLEMENTED]` — not invoked. |
| QPU execution | `qratum/quantum/core.py` is a Qiskit wrapper, `QISKIT_AVAILABLE=False` here; `q-substrate/src/quantum.rs` is a 12-qubit Rust state-vector simulator, not built | `[NOT IMPLEMENTED]` — §12. |

### 2.3 A naming collision, stated plainly

`qratum/verticals/fluxa.py` and `verticals/fluxa.py` already implement a
**supply-chain and logistics** module named FLUXA, returning hard-coded
constants (`"total_distance_km": 1250`). This study's brief defines FLUXA as
the energy-system vertical. The new `fluxa/` package is the energy-system
FLUXA. The two share a name and nothing else; neither imports the other and
**no pre-existing file was modified**. Reconciling the naming is a product
decision, not a research result.

---

## 3. Model

### 3.1 Physical system

A six-bus island transmission system, 420 MW peak demand:

| Bus | Role | Load share |
|---|---|---|
| B1 | Gas Hub — `GAS_CCGT_1` 220 MW, **angle reference** | 0.10 |
| B2 | Solar Plateau — `SOLAR_PV_1` 180 MW | 0.00 |
| B3 | Wind Ridge — `WIND_1` 150 MW | 0.00 |
| B4 | Industrial Park | 0.35 |
| B5 | Urban Center — `BESS_1` 320 MWh / 80 MW | 0.40 |
| B6 | Interconnect — `GAS_GT_2` 90 MW peaker, ±100 MW tie | 0.15 |

Eight corridors, reactances 0.04–0.09 p.u. on a 100 MVA base, ratings
110–200 MW. Battery: 0.95/0.95 one-way efficiencies (0.9025 round trip),
SOC window 0.10–0.95, 4 USD/MWh throughput cost.

**Every parameter's origin is documented in
`fluxa/configs/PARAMETER_SOURCES.md`**, classified `[ASSUMED]`,
`[CONVENTIONAL]` or `[DERIVED]`. **No parameter is an observation of a real
power system.** The system is synthetic, sized so the experiments exercise
real mechanisms (ramp limits binding, congestion, reserve adequacy, load
shedding) rather than to resemble any particular grid.

### 3.2 Power flow: the DC approximation and what it costs

`[ASSUMED]` Four assumptions, stated in `fluxa/network.py`:

1. Bus voltage magnitudes fixed at 1.0 p.u.
2. Branch resistance negligible relative to reactance (lossless branches).
3. Angle differences small, so `sin(θᵢ−θⱼ) ≈ θᵢ−θⱼ`.
4. Reactive power not modelled.

Real-power flows are then an **exact** linear function of bus injections,
`f = PTDF · P`, which is what lets network limits enter the LP directly
rather than through a proxy.

What this forecloses: FLUXA **cannot** speak to voltage collapse,
reactive-reserve adequacy, or transmission losses. Reported costs omit losses
and are therefore optimistic by the loss fraction a real 420 MW system would
incur (order 2–4%). `[DERIVED]`

The approximation is checked against itself rather than assumed: the PTDF
path is validated against an **independent** angle-based solution built from
`B_bus` in `test_ptdf_flows_match_independent_angle_solution` (agreement to
`rtol=1e-10`), Kirchhoff's current law is verified at every bus
(`atol=1e-9`), and
`test_angles_are_small_so_the_dc_linearisation_is_self_consistent` asserts
the solved angle spread stays under 0.35 rad — so the model is consistent
with its own linearisation. `[OBSERVED]`

### 3.3 Dispatch: receding-horizon LP

At each timestep a linear program is solved over a 12-step (1 hour)
look-ahead; the first step is committed and the window advances. Perfect
foresight holds **inside** the window only.

**Decision variables** (276 for the base configuration), per horizon step:
dispatchable generator output, accepted variable-renewable output, battery
charge, discharge, state of charge and reserve contribution, import, export,
per-bus unserved load, per-line overload, reserve shortfall.

**Objective** (minimise, USD over the window):

```
Σ_h dt_h · [ Σ_g c_g·p_disp + Σ_j (c_j − c_curt)·p_var
           + Σ_b c_cycle·(p_ch + p_dis)
           + Σ_l (π_imp·p_imp − π_exp·p_exp)
           + VOLL·Σ_k unserved + C_ovl·Σ_m overload + C_res·reserve_short ]
  − Σ_b v_stored·E_b·soc[H−1,b]        (terminal stored-energy value)
  + K                                  (curtailment offset constant)
```

**Constraints:** power balance (equality, per step); storage energy
conservation (equality, per step per battery); network limits in both
directions with a measured overload slack; ramp limits; an operating-reserve
requirement; and a battery reserve contribution that is **energy-tested**,
not merely a power-headroom claim —
`r_batt·T_res/η_dis ≤ (soc − soc_min)·E`, so a nearly empty battery
contributes nearly nothing (verified in
`test_battery_reserve_contribution_is_energy_tested`).

**Solver:** `scipy.optimize.linprog(method="highs")`, SciPy 1.17.1, NumPy
2.4.6. Problem size 276 variables, 24 equality rows, 276 inequality rows.
Convergence is HiGHS' own primal/dual feasibility and optimality tolerances
at SciPy defaults. **The optimality claim is not trusted**: the engine
rejects any status other than 0, and
`fluxa.engine.validate_physical_state` independently re-derives power
balance, storage conservation, capacity limits, availability limits,
non-negativity and overload consistency from the returned primal. A solver
returning a wrong answer would be caught. `[IMPLEMENTED]`

**Three approximations, stated:** `[ASSUMED]`

- **No unit-commitment binaries.** The LP is a continuous relaxation, so a
  unit with `p_min > 0` is a must-run whenever available. The peaker is given
  `p_min = 0` specifically so it can sit at zero without a binary. Start
  costs, minimum up/down times and start-up trajectories are absent.
- **Charge and discharge are not mutually exclusive by construction.**
  Round-trip loss plus cycle cost make simultaneous operation strictly
  suboptimal. This is **verified at every committed timestep**
  (`simultaneous_charge_discharge_mw`) rather than assumed, and was zero in
  every run. `[OBSERVED]`
- **A generator trip is not a ramp.** The ramp constraint carries an
  availability-dependent relaxation `p_max·(1 − min(a_h, a_{h−1}))`. Without
  it, forcing an online unit to zero in one timestep contradicts its ramp
  limit and the LP is **infeasible** — a forced-outage scenario could not be
  simulated at all. This was discovered by execution, not by inspection (§9.1).
  The term vanishes when a unit is fully available, so normal operation
  remains exactly ramp-constrained.

### 3.4 Frequency: a flagged proxy, not a dynamic simulation

FLUXA does not integrate swing equations. It reports a steady-state
deviation implied by the power primary response failed to arrest:

```
Δf = −(unserved + reserve_shortfall) / β,   β = 30 MW/Hz   [ASSUMED]
```

Beyond ±2 Hz this linear extrapolation is outside any regime a real system
survives without under-frequency load shedding and probable cascading. Such
states are **flagged** `frequency_proxy_in_range = False` and counted
separately. The compound event produced 32 out-of-range steps `[OBSERVED]`:
its reported 39.07 Hz minimum is a **severity ordering**, not a predicted
system frequency.

### 3.5 Exogenous drivers

Load is a double-peaked daily shape with a weekday/weekend factor; solar is a
clear-sky sinusoid modulated by a seeded AR(1) cloud process; wind is a mean
plus diurnal and synoptic sinusoids plus a seeded AR(1) term. AR(1) rather
than white noise, because white noise is averaged away by the dispatch and
understates ramping stress.

All randomness flows through explicitly constructed
`numpy.random.Generator(PCG64(seed))` objects. The global NumPy RNG is never
read. Seeds are independent, verified by
`test_driver_seeds_are_independent`: changing the wind seed leaves the load
and solar series bit-identical. `[OBSERVED]`

`[ASSUMED]` These are synthetic profiles chosen to be physically plausible.
They are **not measurements of any real grid**, and no statistic derived from
them is an empirical grid statistic.

---

## 4. Inputs

Machine-readable and hashed:

| Artefact | Path | Hash |
|---|---|---|
| System | `fluxa/configs/system_6bus.json` | `config_hash b3fc57aa…` / `model_hash fac1fd29…` |
| Scenarios | `fluxa/configs/scenarios.json` | per-scenario, see §5 |
| Parameter origins | `fluxa/configs/PARAMETER_SOURCES.md` | — |

Run parameters for every primary scenario (recorded in each
`*_provenance.json`): `timestep_s 300.0`, `n_steps 288`,
`horizon_steps 12`, `load_seed 10001`, `solar_seed 20002`,
`wind_seed 30003`, `checkpoint_every 24`, `stabilization_steps 6`,
`detection_threshold_fraction 0.02`, `start_iso 2026-01-05T00:00:00Z`.

Two hashes are kept deliberately distinct: `config_hash` covers the raw
document (changes with key order or formatting), `model_hash` covers the
validated `EnergySystem` (changes only when the physics changes).

---

## 5. Scenarios executed

All eight at 24 h / 5-minute resolution (288 timesteps), plus two 7-day
extended runs (2016 timesteps).

| ID | Name | Perturbation | Required level | Scenario hash |
|---|---|---|---|---|
| A | Baseline | none | ROUTINE | `3a3068259da0` |
| B | Solar shock | PV −70% over 15 min at 12:00, hold 2 h, recover 45 min | ELEVATED | `26e7e6cabf50` |
| C | Wind shock | Wind −85% over 20 min at 18:00, hold 3 h, recover 1 h | ELEVATED | `8f0f540db9a7` |
| D | Generator outage | CCGT trips instantly at 19:00, out 3 h | **SENSITIVE** | `566bb266b57a` |
| E | Load spike | Demand +30% over 30 min at 17:30, hold 2 h | ELEVATED | `5fccafb0fe00` |
| F | **Compound** | Wind −80% @18:00 **+** CCGT trip @18:50 **+** demand +25% @19:00 | **SENSITIVE** | `4201e9fd5709` |
| G | Transmission derate | L5 (B3–B5) −55% for 4 h from 06:00 | ELEVATED | `d6101fd5f26b` |
| H | Solar congestion | L1+L3 −70% **+** midday demand −40%, 4 h from 10:00 | ELEVATED | `e8e09d271b0e` |

G and H were **designed empirically, not analytically.** An initial G
(L6 derated 60%) had no measurable effect: L6 runs at 0.18 utilisation. L5
was identified as the binding corridor by measuring per-line utilisation
across the baseline (max 0.836 at 08:00), and the derate was retargeted.
Similarly, an initial H (midday demand −50%) produced only
1.14 MWh of curtailment, because export and storage absorb the surplus.
Reporting that as a renewable-integration result would have been vacuous, so
H was rebuilt as a **network-congestion** curtailment case — which is also
the dominant real-world curtailment mechanism. `[OBSERVED]`

---

## 6. Results

### 6.1 Primary scenarios — 24 h at 5-minute resolution

| Scenario | Served (MWh) | Unserved (MWh) | Unserved % | Violation steps | Max line util. | Reserve short (MWh) | Min f-proxy (Hz) | Operating cost (USD) | USD/MWh | CO₂ (t) | VRE penetration | Curtailed (MWh) | Events |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A Baseline | 7,956.7 | 0.0 | 0.00% | 0 | 0.836 | 0.00 | 50.000 | 248,109 | 31.18 | 1,474 | 0.400 | 0.0 | 529 |
| B Solar shock | 7,956.7 | 0.0 | 0.00% | 0 | 0.836 | 0.00 | 50.000 | 260,942 | — | 1,569 | 0.368 | 0.0 | 544 |
| C Wind shock | 7,956.7 | 0.0 | 0.00% | 0 | 0.856 | 0.00 | 50.000 | 261,338 | — | 1,545 | 0.381 | 0.0 | 557 |
| D Gen outage | 7,550.4 | **406.3** | 5.11% | 37 | 0.921 | 5.45 | 42.870 | 240,885 | 31.90 | 1,371 | 0.421 | 0.0 | 601 |
| E Load spike | 8,168.2 | **127.5** | 1.54% | 23 | **1.000** | 19.00 | 46.021 | 267,319 | 32.73 | 1,580 | 0.389 | 0.0 | 618 |
| **F Compound** | 7,493.9 | **716.9** | **8.74%** | 33 | 0.909 | 11.99 | **39.072** | 246,658 | 32.91 | 1,404 | 0.401 | 0.0 | 629 |
| G Line derate | 7,956.7 | 0.0 | 0.00% | 0 | **1.000** | 0.00 | 50.000 | 265,199 | 33.33 | 1,574 | 0.361 | **312.5** | 607 |
| H Solar congestion | 7,404.5 | 0.0 | 0.00% | 0 | **1.000** | 0.00 | 50.000 | 240,297 | 32.45 | 1,391 | 0.383 | **347.2** | 596 |

`[OBSERVED]` Source: `results/fluxa/primary_scenario_summary.csv` and
`results/fluxa/scenarios/primary_*_metrics.json`.

Readings that matter:

- **Single renewable shocks were absorbed without loss of load.** B and C
  cost 5.2% more to operate (gas and imports replacing the lost renewable)
  but served every MWh, with no violation and no reserve shortfall. The
  system has enough dispatchable and tie capacity for a single VRE event.
- **The compound event is the severe case, and non-additively so.**
  F's 716.9 MWh unserved exceeds D (406.3) + C (0.0) = 406.3 by **76%**.
  `[DERIVED]` Losing the CCGT while wind is collapsing and demand is rising
  is qualitatively worse than losing it alone, because the resources that
  would cover the outage are already committed.
- **Operating cost falls in the outage cases.** D costs *less* than the
  baseline ($240,885 vs $248,109) because 406 MWh of load was not served —
  unserved energy is cheap in production cost and catastrophic in VOLL
  ($4.06 M). Quoting operating cost alone for a scenario with load shedding
  is misleading, which is why VOLL is reported separately and never folded
  into operating cost.
- **Line utilisation reaching exactly 1.000 in E, G and H is the constraint
  binding, not being violated.** Max overload was 0.000 MW in all three; the
  LP redispatched to respect the limit. The soft overload slack never
  activated in any scenario run — it activated only under the deliberate
  adversarial squeeze (§10, F3). `[OBSERVED]`
- **Curtailment is structurally zero on this capacity mix** and only appears
  through network congestion (G: 312.5 MWh, 9.8% of available VRE;
  H: 347.2 MWh, 10.9%). Combined VRE nameplate is 79% of peak load — not an
  overbuild — so there is no energy surplus to curtail. This is a property
  of the system, not an achievement of the dispatch. `[OBSERVED]`

### 6.2 Economic decomposition

| Scenario | Generation | — CCGT | — Peaker | — Solar | — Wind | Storage | Import | **Operating** | Curtailment | VOLL | Soft penalty |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 202,613 | 191,787 | 5,899 | 1,434 | 3,493 | 699 | 44,797 | **248,109** | 0 | 0 | 0 |
| D | 193,765 | 158,787 | 30,051 | 1,434 | 3,493 | 814 | 46,306 | **240,885** | 0 | 4,063,229 | 2,725 |
| E | 220,393 | 191,787 | 23,679 | 1,434 | 3,493 | 730 | 46,196 | **267,319** | 0 | 1,274,914 | 9,499 |
| F | 197,740 | 163,627 | 29,538 | 1,434 | 3,140 | 794 | 48,124 | **246,658** | 0 | 7,168,725 | 5,996 |
| G | 215,834 | 203,012 | 8,519 | 1,434 | 2,868 | 756 | 48,610 | **265,199** | 4,687 | 0 | 0 |
| H | 191,712 | 178,693 | 8,439 | 1,087 | 3,493 | 736 | 47,849 | **240,297** | 5,209 | 0 | 0 |

All figures USD over 24 h. `[OBSERVED]`

Operating cost is strictly a production cost: fuel, variable O&M, storage
cycling and net interchange. Soft-constraint penalties (overload, reserve
shortfall) are **not market prices** — they exist so a violation is measured
instead of making the LP infeasible — and are reported separately. The
disturbances shift cost from the CCGT to the peaker (5,899 → 30,051 USD in D,
a 5.1× increase) and to imports, exactly the merit-order substitution the
model should produce. `[OBSERVED]`

### 6.3 Extended simulation — 7 days at 5-minute resolution

| | A Baseline (7 d) | F Compound, day 5 (7 d) |
|---|---|---|
| Timesteps | 2,016 | 2,016 |
| Events | 3,517 | 3,615 |
| Checkpoints | 14 | 14 |
| Ledger verifies | **yes** | **yes** |
| Energy served | 53,713.6 MWh | 53,241.0 MWh |
| Unserved | 0.0 MWh | **726.4 MWh** |
| Operating cost | 1,731,574 USD | 1,734,422 USD |
| CO₂ | 10,527.7 t | 10,473.8 t |
| VRE penetration | 0.387 | 0.386 |
| Operating states | NORMAL 2,016 | NORMAL 1,974 · ALERT 3 · EMERGENCY 33 · RESTORATIVE 6 |
| Detection step | — | 1,369 |
| Recovery time | — | 180.0 min |
| Wall time | 10.48 s | 10.32 s |

`[OBSERVED]` The disturbance was shifted to day 5 so the extended run is not
the 24 h run repeated. The compound event's 7-day unserved energy
(726.4 MWh) exceeds its 24 h value (716.9 MWh) because the weekday load shape
differs on the later day. Throughput held at 191.3 timesteps/s across the
7× longer horizon, i.e. **no superlinear degradation** with trajectory
length. `[OBSERVED]`

### 6.4 Reliability and resilience

| Scenario | Detect step | First degraded | First violation | Stabilised at | Recovery (min) | Deficit (MWh) | Max stress | Min reserve margin | Out-of-range f-proxy steps |
|---|---|---|---|---|---|---|---|---|---|
| B | 145 | — | — | — | — | 0.0 | 0.836 | −0.000 | 0 |
| C | 217 | — | — | — | — | 0.0 | 0.856 | −0.000 | 0 |
| D | 228 | 228 | 228 | 266 | **190** | 406.3 | 0.921 | −1.000 | 36 |
| E | 211 | 218 | 218 | 246 | **140** | 127.5 | 1.000 | −1.000 | 12 |
| F | 217 | 226 | 226 | 263 | **185** | 716.9 | 0.909 | −1.000 | 32 |
| G | 73 | 118 | — | 124 | **255** | 0.0 | 1.000 | −0.000 | 0 |
| H | 121 | 168 | — | 174 | **260** | 0.0 | 1.000 | −0.000 | 0 |

`[OBSERVED]`

Definitions, stated so the numbers cannot be over-read:

- **Detection** is the first timestep at which a perturbed driver departs
  from its baseline by more than 2% — i.e. **observability latency at a
  stated threshold**. It is *not* a claim about SCADA or telemetry delay.
  Measured delay was exactly one timestep (5 min) in every ramped case,
  which is the floor: a trapezoid has zero intensity at its own onset.
- **First degraded** is the first ALERT or EMERGENCY step. G and H degrade
  (ALERT, from line utilisation ≥ 0.95) without any hard violation — the
  network margin eroded but nothing was shed or overloaded.
- **Stabilisation** is the start of the first run of 6 consecutive
  non-degraded steps after the first degraded step. **Recovery time** runs
  from first degradation to stabilisation. Every disturbed scenario
  stabilised within the 24 h horizon. `[OBSERVED]`
- **Deliberately not reported:** SAIDI, SAIFI, CAIDI, ENS, LOLP, LOLE, EUE.
  These have regulatory definitions tied to customer counts, interruption
  events and multi-year observation windows that this model does not
  represent. Emitting a number under one of those names from a single
  synthetic 24 h trajectory would be a fabrication, and
  `test_no_standard_reliability_index_is_fabricated` enforces their absence
  from the metric schema. `[IMPLEMENTED]`

### 6.5 Storage

| Scenario | Charge (MWh) | Discharge (MWh) | Equiv. cycles | SOC range | SOC end | Contribution | Energy-balance residual (MWh) |
|---|---|---|---|---|---|---|---|
| A | 28.0 | 146.8 | 0.273 | 0.100–0.478 | 0.100 | 1.85% | −9.9 × 10⁻¹⁴ |
| D | 43.0 | 160.4 | 0.318 | 0.100–0.478 | 0.100 | 2.12% | −1.4 × 10⁻¹⁴ |
| F | 40.4 | 158.0 | 0.310 | 0.100–0.478 | 0.100 | 2.11% | −4.3 × 10⁻¹⁴ |
| G | 35.4 | 153.6 | 0.295 | 0.100–0.478 | 0.100 | 1.93% | +5.7 × 10⁻¹⁴ |

`[OBSERVED]` The storage energy balance
`E_final − E_initial = η_ch·charge − discharge/η_dis` closes to ~10⁻¹⁴ MWh in
every scenario — machine precision over 288 timesteps.

**The battery drains to its floor and never refills.** This is economically
rational given the base-case parameters and is a visible modelling artefact,
quantified in §8.4. The 1.85–2.12% storage contribution is therefore **not**
an optimal storage dispatch.

### 6.6 Event volume and composition

| Event type | A (288 steps) | F (288 steps) | G (288 steps) |
|---|---|---|---|
| SIMULATION_CREATED | 1 | 1 | 1 |
| SCENARIO_INITIALIZED | 1 | 1 | 1 |
| STATE_ADVANCED | 288 | 288 | 288 |
| LOAD_CHANGED | 29 | 41 | 29 |
| GENERATION_CHANGED | 108 | 112 | 110 |
| STORAGE_DISPATCHED | 89 | 93 | 98 |
| CHECKPOINT_CREATED | 12 | 12 | 12 |
| DISTURBANCE_DETECTED | — | 1 | 1 |
| CONSTRAINT_APPROACHED | — | — | 51 |
| CONSTRAINT_VIOLATED | — | 37 | — |
| MITIGATION_EXECUTED | — | 41 | 14 |
| SYSTEM_STABILIZED | — | 1 | 1 |
| SIMULATION_COMPLETED | 1 | 1 | 1 |
| **Total** | **529** | **629** | **607** |

`[OBSERVED]` Every event carries enough provenance to reconstruct what
happened: `STATE_ADVANCED` carries the timestep's full state hash plus the
decisive scalars; `CONSTRAINT_VIOLATED` carries per-line utilisation,
overload and reserve shortfall; `MITIGATION_EXECUTED` names the corrective
action taken (peaker increase, import increase, battery discharge increase,
or load shed) with its magnitude.

---

## 7. Determinism

### 7.1 Result

Three replays of each of three scenarios, 288 timesteps each, identical
inputs and run label:

| Property compared | A Baseline | F Compound | G Line derate |
|---|---|---|---|
| Per-timestep state hashes (288 each) | **identical** | **identical** | **identical** |
| First divergence step | none | none | none |
| Final state hash | **identical** | **identical** | **identical** |
| Event sequence (type, step, safety level) | **identical** | **identical** | **identical** |
| FLUXA event-chain root | **identical** | **identical** | **identical** |
| Event Merkle-tree root | **identical** | **identical** | **identical** |
| State Merkle-tree root | **identical** | **identical** | **identical** |
| Metric output hash | **identical** | **identical** | **identical** |
| Provenance identity hash | **identical** | **identical** | **identical** |
| QRADLE `ExecutionResult.output_hash` | **identical** | **identical** | **identical** |
| **QRADLE native chain root** | **3 distinct values** | **3 distinct values** | **3 distinct values** |

`[OBSERVED]` Source: `results/fluxa/determinism.json`.

Run 1 == Run 2 == Run 3 **at the bit level** for everything FLUXA controls.
Comparison is on SHA-256 over canonical JSON with exact float `repr`, so
agreement means agreement on the last bit of every value — not agreement to
some tolerance. `test_canonical_hash_is_sensitive_to_the_last_float_bit`
confirms the digest separates `1.0` from `1.0 + 2⁻⁵²`.

**Floating-point non-determinism was looked for and not found.** HiGHS
produced bit-identical primals across repeated solves on this build and
machine. `[OBSERVED]` This is a property of *this* SciPy/HiGHS build on *this*
hardware; it is not a guarantee, and the environment fingerprint
(Python, platform, NumPy, SciPy versions) is recorded in every provenance
bundle precisely so a future divergence can be attributed to the numerical
stack rather than to the model. `[HYPOTHESIS]` Cross-platform and
cross-BLAS bit-reproducibility is untested and should not be assumed.

### 7.2 Sensitivity in the correct direction

Determinism is worthless if the identity is insensitive to inputs. Each of
these changed the identity hash: `[OBSERVED]`

| Change | Identity changed | Bundle fields that differ |
|---|---|---|
| `wind_seed 30003 → 424242` | yes | `profile_hash`, all roots |
| `horizon_steps 12 → 6` | yes | `run_parameters`, all roots |
| `timestep_s 300 → 900` | yes | `run_parameters`, `solver`, all roots |
| `stochastic_profiles → False` | yes | `profile_hash`, all roots |
| CCGT cost `48 → 49` USD/MWh | yes | `config_hash`, `model_hash`, all roots |
| Scenario A → B | yes | `scenario_hash`, all roots |
| **Run label only** | **no** | `run_id` and `bundle_hash` only |

The last row is the designed property: a label must not be able to change a
run's cryptographic identity, or two executions of the same experiment could
never be compared.

---

## 8. Provenance

### 8.1 The three required tests

`[OBSERVED]` Source: `results/fluxa/provenance.json`. Subject: scenario F,
288 timesteps, 629 events.

**Test 1 — untampered simulation verifies.** PASS.
`ledger.verify()` → `True`, no problems. All **629** inclusion proofs
recomputed the recorded event-tree root. The recomputed state-tree root
matched the bundle's, and the last timestep's recomputed hash matched
`final_state_hash`.

**Test 2 — modify one historical event; verification must fail.** PASS, with
an important detail. Event 314's payload was edited
(`unserved_mw → 0.0`). `ledger.verify()` → `False`, reporting exactly:

```
node 314: content hash does not match data
```

**The chain head did not change** (`tamper_changes_chain_head: False`), and
neither did a Merkle tree recomputed over the stored per-node hashes —
because the attacker did not update the stored hash. **A head comparison
alone would have missed this.** Detection came from per-node content
recomputation. A second, more capable attacker who *did* rehash the edited
node was also detected, by the linkage check at the following node:

```
node 6: previous_hash 5baa5269b537… != expected 31754b4d755a…
```

Both detection paths are necessary; neither alone is sufficient.

**Test 3 — modify one input; the provenance identity must change.** PASS.
`BESS_1.energy_capacity_mwh 320.0 → 321.0` (a 0.3% change to one number)
changed **ten** bundle fields: `config_hash`, `model_hash`,
`event_chain_root`, `event_tree_root`, `state_tree_root`,
`final_state_hash`, `output_hash`, `qradle_output_hash`,
`qradle_chain_root`, `identity_hash`.

### 8.2 Why FLUXA does not use QRADLE's Merkle proof

`qradle.core.merkle.MerkleProof.verify` is:

```python
def verify(self, claimed_root: str) -> bool:
    return self.root_hash == claimed_root
```

It compares a field the proof itself carries against a claimed root. It never
recomputes anything from `event_hash` or `proof_path`. Consequence, executed:
a proof with `event_hash = "00"*32` and `proof_path = ["ff"*32]`, keeping
only the genuine `root_hash`, **verifies successfully** — through both
`MerkleProof.verify()` and `MerkleChain.verify_proof()`. `[OBSERVED]`

`fluxa/provenance.py` therefore implements a binary Merkle tree whose
inclusion proofs **recompute the root from the leaf and sibling path**, with
domain-separated leaf (`\x00FLUXA_LEAF`) and internal-node
(`\x01FLUXA_NODE`) tags to close the standard second-preimage attack on trees
with duplicated odd leaves. Verified across tree sizes 1, 2, 3, 4, 5, 7, 8,
16, 17 and 100: every leaf's proof verifies, and a substituted leaf, a
corrupted sibling, a reordered path or a foreign root all fail. `[OBSERVED]`

### 8.3 A limit of append-only logs, stated

Attack B4 appended a fabricated `SYSTEM_STABILIZED` event to a closed ledger.
The ledger remained **internally self-consistent** — an append-only log
cannot detect its own extension from the inside. Detection came from the
externally recorded root in the provenance bundle no longer matching.
`[OBSERVED]` This means the tamper-evidence guarantee is conditional on the
provenance bundle being published or held separately from the ledger it
attests. FLUXA writes them as separate artefacts; nothing in the
architecture *enforces* separate custody. `[NOT IMPLEMENTED]`

### 8.4 The storage-value sensitivity: the strongest result against the economics

`[OBSERVED]` Source: `results/fluxa/storage_value_sensitivity.json`,
pinned by `fluxa/tests/test_storage_value.py`.

A receding-horizon LP **must** price energy left in the battery at the window
edge, or it empties the battery as an artefact of the horizon. The base case
sets `v_stored = 48` USD/MWh, equal to the CCGT marginal cost, which looked
like the natural choice. Sweeping it on the identical physical system:

| `v_stored` (USD/MWh) | Charge (MWh) | Discharge (MWh) | Equiv. cycles | SOC range | SOC end | Contribution | Operating cost (USD) | USD/MWh |
|---|---|---|---|---|---|---|---|---|
| 0 | 34.4 | 152.6 | 0.292 | 0.100–0.478 | 0.100 | 1.92% | 250,947 | 31.54 |
| **48 (base case)** | **28.0** | **146.8** | **0.273** | **0.100–0.478** | **0.100** | **1.85%** | **248,109** | **31.18** |
| **60** | **211.7** | **312.6** | **0.819** | **0.100–0.950** | **0.100** | **3.93%** | **245,008** | **30.79** |
| 75 | 177.1 | 125.8 | 0.473 | 0.449–0.950 | 0.612 | 1.58% | 256,324 | 32.21 |
| 92 | 170.5 | 22.5 | 0.302 | 0.488–0.950 | 0.932 | 0.28% | 266,895 | 33.54 |
| 120 | 170.5 | 22.5 | 0.302 | 0.488–0.950 | 0.932 | 0.28% | 267,786 | 33.66 |

At `v_stored = 48` the battery delivers its initial 128 MWh of stock and then
sits at `soc_min`: recharging costs 48 (gas) + 4 (cycle) per MWh to store
0.95 MWh worth 48, so the LP correctly refuses. At `v_stored = 60` — between
the CCGT cost (48) and the import price (75) — the battery cycles **3.0×**
more, spans its full SOC window, and operating cost falls **1.25%** to
245,008 USD.

**Therefore the base case's 248,109 USD is not an optimum, and the
1.85% storage contribution is an artefact of a parameter choice, not a
property of the system.** `[DERIVED]` The base case was left unchanged rather
than tuned, because tuning a parameter until the answer improves and then
reporting only the improved answer is the failure mode this section exists to
prevent. The honest statement is: this dispatch formulation's storage result
is dominated by a terminal-value assumption, and the assumption was not
chosen well.

---

## 9. Defects found by execution

Four determinism defects had to be found and fixed before the §7 result was
achievable. None was visible by inspection; all surfaced by running the
experiment. `[OBSERVED]`

### 9.1 A generator trip was infeasible (FLUXA)

Scenario D failed with `HiGHS Status 8: model_status is Infeasible`. Cause:
`dispatchable_availability` drops to 0 instantly on a trip, while the ramp
constraint `|p_t − p_{t−1}| ≤ ramp·Δt` still binds. From 180 MW with a
20 MW/step ramp, reaching 0 in one step is infeasible — so **forced-outage
scenarios could not be simulated at all**. Fixed by relaxing the ramp bound
by `p_max·(1 − min(a_t, a_{t−1}))`, which vanishes when the unit is fully
available. Pinned by `test_generator_trip_is_feasible_despite_ramp_limit`
and `test_ramp_limit_binds_when_unit_is_available`.

### 9.2 The run label leaked into the state hash (FLUXA)

`FluxaState.state_hash()` hashed the whole state dict, `run_id` included, so
two runs of the same experiment under different labels produced different
state hashes and different state-tree roots. Diagnosed by diffing the
first state's dict: the **only** difference was `run_id`. Fixed by excluding
labels from the content hash via `HASH_EXCLUDED_FIELDS`, keeping `run_id`
on the state and in the CSV for traceability.

### 9.3 The QRADLE contract id included the run label (FLUXA × QRADLE)

QRADLE folds `contract_id` into `ExecutionResult.output_hash`. FLUXA's
contract id was `fluxa:{run_id}:{scenario_id}`, so the QRADLE output hash was
label-dependent. Fixed by addressing the contract by **experiment**:
`FLUXA:{system_id}:{scenario_id}:{hash(config, model, scenario, params)[:16]}`,
with the run label moved to `ExecutionContext.metadata`.

### 9.4 QRADLE checkpoint ids embed the wall clock (QRADLE)

`RollbackManager.create_checkpoint`, when no id is supplied, generates
`f"checkpoint_{state_hash[:16]}_{int(datetime.now(timezone.utc).timestamp())}"`.
That id goes into FLUXA's `CHECKPOINT_CREATED` event payload, so **the entire
event-chain root became wall-clock dependent**. Worked around by supplying
explicit deterministic ids (`{contract_id}:ckpt{step:06d}`), which the QRADLE
API supports. The defect remains in QRADLE's default path.

### 9.5 Unfixed and reported: QRADLE's wall-clock Merkle chain (QRADLE)

`MerkleChain.append` stamps every node with `datetime.now(timezone.utc)`.
Three identical runs produced three distinct chain roots. This is **not**
worked around: `qradle_chain_root` is excluded from FLUXA's provenance
identity, and `test_qradle_native_chain_root_is_wall_clock_dependent`
asserts the non-determinism so the claim fails loudly if upstream changes.
FLUXA's own ledger reuses QRADLE's `MerkleNode` primitive byte for byte but
supplies the **simulation** timestamp, which is what makes it reproducible
with unchanged hashing semantics.

### 9.6 A metric that silently meant the wrong thing (FLUXA)

A conservation test failed with a "realised round-trip efficiency" of
**3.91** against a 0.9025 nameplate. The physics was correct; the *metric*
was wrong. `discharge_mwh / charge_mwh` ignores the initial stored energy, so
a battery that delivers its starting stock shows an "efficiency" far above 1
without creating any energy. Replaced with the exact storage energy balance
and its residual (now ~10⁻¹⁴ MWh, §6.5), equivalent full cycles, and net
withdrawal. A metric that does not mean what its name says is worse than a
missing metric.

### 9.7 Two further execution-found defects

- **A stabilisation deadlock.** `RESTORATIVE` was treated as degraded, and a
  system could not leave `RESTORATIVE` until it had already been stable, so
  stabilisation was unreachable and every disturbed scenario reported
  `stabilised: None`. Fixed by defining degradation as `ALERT` or
  `EMERGENCY` only.
- **A ragged CSV export.** `active_perturbations` has different keys at
  different timesteps, so flattening it gave different timesteps different
  columns and the table could not be loaded. Caught by a strict schema check
  in `write_states_csv`; fixed by serialising variable-key dicts as one
  canonical-JSON column plus scalar summaries.

---

## 10. Rollback

`[OBSERVED]` Source: `results/fluxa/rollback.json`. Sequence per the brief:
t0 → t1 → t2 → t3 → t4, checkpoint at t2, invalid state injected at t4.

| | Baseline / power balance | Baseline / storage | **Compound, mid-disturbance** |
|---|---|---|---|
| Fault injected | +25 MW phantom generation at t4 | SOC raised 0.40 above trajectory at t4 | +40 MW phantom generation at step 230 |
| Forward steps | 5 | 5 | 232 |
| Checkpoint / corrupted step | 2 / 4 | 2 / 4 | 226 / 230 |
| **Invalid state detected** | **yes, step 4** | **yes, step 4** | **yes, step 230** |
| Detecting signal | `POWER_BALANCE: supply 323.634648219 MW != demand 298.634648219 MW (residual 2.500e+01)` | `STORAGE_CONSERVATION: BESS_1 stored 252.912280702 MWh != expected 124.912280702 MWh` | `POWER_BALANCE: supply 507.661964789 MW != demand 467.661964789 MW (residual 4.000e+01)` |
| **Rollback refused without authorization** | **yes** | **yes** | **yes** |
| **Checkpoint verified before restore** | **yes** | **yes** | **yes** |
| **Restored hash == recorded hash** | **yes** | **yes** | **yes** |
| **Resumed trajectory bit-identical to clean reference** | **yes** | **yes** | **yes** |
| First resume divergence | none | none | none |
| Steps resumed | 2 | 2 | 5 |
| Ledger verifies | yes | yes | yes |
| Events | 19 | 19 | 476 |

Event sequence (baseline case), every stage recorded:

```
SIMULATION_CREATED → CHECKPOINT_CREATED ×5 interleaved with STATE_ADVANCED ×5
→ INVALID_STATE_DETECTED → AUTHORIZATION_DENIED → ROLLBACK_AUTHORIZED
→ ROLLBACK_EXECUTED → STATE_ADVANCED ×2 → SYSTEM_STABILIZED
→ SIMULATION_COMPLETED
```

Three things make this a demonstration rather than an assertion:

1. **Detection is independent of the fault's nature.** The storage case was
   caught by the *conservation* check, not the SOC-bound check: raising SOC by
   0.40 from 0.478 gives 0.878, still inside the 0.10–0.95 window. The
   step-to-step energy balance caught what a bounds check would have missed.
2. **Restoration is verified by hash, not by absence of an exception.**
   QRADLE's `Checkpoint.verify()` recomputes the state hash from stored data;
   the restored state's recomputed hash is then compared to the one recorded
   at t2.
3. **Resume uses the same code as the original run.** `step_once` was
   extracted from the main loop specifically so the recovered trajectory is
   produced by identical physics. The resumed steps matched the uncorrupted
   reference hash-for-hash — including in the compound case, where the
   rollback happened with three perturbations active.

`test_tampered_checkpoint_fails_verification` separately confirms QRADLE
rejects a checkpoint whose stored state was altered after creation.

---

## 11. Safety model and authorization

### 11.1 Mapping

| QRADLE level | FLUXA condition | Observed in |
|---|---|---|
| ROUTINE | ordinary state advancement | all scenarios (110–131 steps each) |
| ELEVATED | major dispatch change, material storage action, or eroded margin (ALERT) | all scenarios (139–172 steps each) |
| SENSITIVE | a forced generator outage is in effect | D (2 steps) |
| CRITICAL | load shedding, or ≥2 simultaneous violations | D (37), E (23), F (33) |
| EXISTENTIAL | **never emitted** | — |

`[OBSERVED]` All four used levels occur in executed runs
(`test_all_four_levels_are_exercised_across_the_library`).

**EXISTENTIAL is never emitted, by design and by test.** An ordinary grid
contingency — even one shedding 717 MWh — is not an architecture-level
catastrophe. `test_fluxa_never_emits_existential` enforces this. Inflating a
power-system event to the top of a safety scale would make the scale
meaningless.

**Precedence is deliberate: CRITICAL dominates SENSITIVE.** A step that both
has a unit out *and* is shedding load is CRITICAL, so an outage cannot mask
load shedding in the audit trail. This is why F shows **no** SENSITIVE steps:
every one of its 33 outage steps was also shedding load. D's 2 SENSITIVE
steps are its recovery ramp, where the unit is partially back and nothing is
being shed. `[OBSERVED]`

### 11.2 The gate works

`[OBSERVED]`

- Scenarios D and F (`required_safety_level = SENSITIVE`) raise
  `InvariantViolation [human_oversight_requirement]` when run without
  authorization.
- The refusal happens **before any physics**:
  `engine.qradle.get_stats()["total_executions"] == 0` after a denial.
- The refusal is itself audited: an `AUTHORIZATION_DENIED` event is appended,
  naming the scenario, the required level and the gated perturbation ids.
- The level is derived from **perturbation kinds**, not from any text field.
  Relabelling D as "Routine maintenance check" with tags `("routine",)`
  changes nothing: it is still refused (attack E2).
- Scenarios A, B, C, E, G, H require no authorization and execute without it.

### 11.3 The gate is not authentication — and this is the biggest gap

`RunConfig.authorized` is a **boolean the caller asserts**. FLUXA and QRADLE
verify that an authorization flag was *set*, not that a human set it. There
is no signature, no credential, no second party, no challenge. Attack
`E3_SELF_ASSERTED_AUTHORIZATION` set `authorized=True` with
`authorizer_id="adversarial"` and the gated scenario ran to completion.
**The attack succeeded.** `[OBSERVED]`

This is not a FLUXA shortcut: it is the shape of
`FatalInvariants.enforce_human_oversight(operation, safety_level, authorized)`.
Notably, QRADLE **already contains** the missing mechanism —
`qradle/core/zones.py` implements `ZonePolicy.require_dual_control` and
`ZoneContext.has_dual_control()`, which requires an actor plus a distinct
approver and explicitly forbids self-approval. The `DeterministicEngine`
authorization path does not use it. `[NOT IMPLEMENTED]` as an integration.

Until that is wired up, QRADLE's "human oversight" invariant is an
**auditability** control (the claim is recorded and attributable) rather than
an **access** control (the claim is verified).

---

## 12. Quantum / Q-Substrate

**No quantum acceleration was demonstrated in this experiment.** `[OBSERVED]`

What exists in the repository, and why none of it was used:

| Component | Status | Why not used |
|---|---|---|
| `qratum/quantum/core.py` (`QuantumBackend`, `QiskitAerBackend`, `IBMQBackend`) | A Qiskit wrapper. `QISKIT_AVAILABLE = False` in this environment. | No functioning backend without installing Qiskit. |
| `quasim/hybrid_quantum/backends.py` (IBM, IonQ, Braket, Azure, Quantinuum) | Interface definitions. | All require external credentials and a Qiskit stack. |
| `q-substrate/src/quantum.rs` | A **real** 12-qubit deterministic state-vector simulator in Rust (full gate set, ~32 KB state). | Not built, not invoked. Wiring it to an LP that already solves to proven optimality would add a build dependency and no scientific result. |

The brief's condition — "only perform this section if the repository contains
a functioning quantum-computation abstraction" — is not met: the abstraction
exists but has no functioning backend here.

Stating the categories separately, as required:

- **Hardware execution:** none.
- **Simulator execution:** none.
- **Conceptual architecture:** a QPU member exists in
  `ComputeSubstrate` and `SubstrateSelector` routes "quantum tasks" to it.
  `[NOT IMPLEMENTED]` for FLUXA.

`[HYPOTHESIS]` Unit commitment with binary on/off variables — which FLUXA
deliberately relaxes (§3.3) — is the natural candidate for a QAOA or
quantum-annealing comparison, because it is the part of this problem that is
genuinely combinatorial. Establishing whether that helps requires a MILP
baseline first, which does not exist here. Nothing in this experiment
supports any claim about quantum advantage.

---

## 13. Performance

Host: 4 logical / 4 physical cores, `Linux-6.18.44-fc-v64-x86_64`,
16.9 GB RAM, CPython 3.11.15. All figures `[OBSERVED]`, from
`results/fluxa/benchmark.json`. These are specific to this container and are
not a hardware comparison.

### 13.1 Single-run throughput

| Configuration | Steps | Horizon | Events | Wall (s) | Solve (s) | Solve % | Steps/s | Events/s | CPU util. | ΔRSS (MB) |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline 24 h / 5 min | 288 | 12 | 529 | 1.537 | 1.442 | 93.8% | 187.4 | 344.2 | 100% | 0.00 |
| Compound 24 h / 5 min | 288 | 12 | 629 | 1.538 | 1.438 | 93.6% | 187.3 | 409.1 | 100% | 0.00 |
| Baseline **7 d** / 5 min | 2,016 | 12 | 3,587 | 10.536 | 9.899 | 94.0% | **191.3** | 340.4 | 100% | −0.49 |
| Baseline 24 h / 15 min | 96 | 4 | 175 | 0.315 | 0.282 | 89.6% | 305.2 | 556.4 | 100% | 0.00 |
| Baseline, myopic (H=1) | 288 | 1 | 517 | 0.734 | 0.638 | 86.9% | **392.1** | 703.9 | 100% | 0.00 |

**93.6–94.0% of wall time is inside the LP solver.** Everything FLUXA and
QRADLE add — state assembly, validation, event emission, Merkle hashing,
checkpointing, metric computation — is the remaining ~6%. Memory is flat:
ΔRSS ≈ 0 over 2,016 timesteps, because states are plain dataclasses and the
LP matrices are built once per configuration, not per solve.

Throughput *rises* slightly from 288 to 2,016 steps (187.4 → 191.3 steps/s),
confirming no superlinear cost in trajectory length.

The myopic (H=1) run is 2.1× faster, which is the price of the receding
horizon: 109% more wall time to make the battery respond to the hour ahead
rather than the instant. `[DERIVED]`

### 13.2 Scaling with scenario count

| Mode | Scenarios | Wall (s) | s/scenario | Timesteps/s | Workers |
|---|---|---|---|---|---|
| Serial | 1 | 1.44 | 1.438 | 200.3 | 1 |
| Serial | 10 | 14.64 | 1.464 | 196.7 | 1 |
| Serial | 100 | 146.49 | 1.465 | 196.6 | 1 |
| Parallel | 1 | 1.46 | 1.459 | 197.4 | 1 |
| Parallel | 10 | 4.47 | 0.447 | 644.3 | 4 |
| Parallel | 100 | **37.01** | **0.370** | **778.1** | 4 |

Serial scaling is **linear to within 1.9%** across two orders of magnitude
(1.438 → 1.465 s/scenario). The three points are nested subsets of one
experiment, not three workloads: sample *k* is derived from its own
`SeedSequence` child, so a 10-sample and a 100-sample campaign agree exactly
on their first 10 samples (`test_sample_k_is_independent_of_the_campaign_size`).

### 13.3 Serial versus parallel

| | |
|---|---|
| Samples | 16 × 288 timesteps |
| Serial wall | 23.050 s |
| Parallel wall | 6.072 s |
| Workers | 4 |
| **Speed-up** | **3.796×** |
| **Parallel efficiency** | **94.9%** |
| **Results identical** | **yes** — no mismatch in any sample's provenance identity hash |

The last row is the point. A speed-up that changes the answer is not a
speed-up; `test_parallel_and_serial_campaigns_agree_exactly` asserts it in
the suite as well as measuring it here. Determinism is what makes the
parallel result trustworthy.

### 13.4 The cost of auditability

The central engineering question about this architecture: are the guarantees
affordable? Measured by running identical physics with the provenance layer
on and off, minimum of 3 repeats each:

| | Events on | Events off |
|---|---|---|
| Events emitted | 529 | 3 |
| Checkpoints | 12 | 0 |
| Wall time (min of 3) | 1.511 s | 1.474 s |
| **Overhead** | **0.037 s — 2.49%** | — |
| **Physics identical** | **yes** (state hashes match) | — |

**2.49%** buys 529 Merkle-chained events, 12 verified checkpoints, a
tamper-evident ledger and a full provenance bundle, with bit-identical
results. `[OBSERVED]` On this workload the auditability layer is not a
material cost — but note that the LP dominates at 94% of runtime, so the
*relative* overhead would be larger for a cheaper physics kernel.
`[DERIVED]`

---

## 14. Monte Carlo / stochastic extension

`[OBSERVED]` 100 samples, `campaign_seed = 777001`, 288 timesteps each,
36.8 s on 4 workers, **0 failures**. Source:
`results/fluxa/monte_carlo.json`, per-sample rows in
`results/fluxa/monte_carlo_samples.csv`.

Randomised per sample, all from one `SeedSequence`: load level (lognormal,
clipped to 0.70–1.35), solar availability (uniform 0.25–1.05), wind
availability (uniform 0.15–1.15), CCGT forced outage (p = 0.25, duration
uniform 60–240 min), a VRE shock (p = 0.50, depth uniform 0.30–0.90, target
chosen uniformly), disturbance onset (uniform 04:00–21:00), and the three
driver seeds.

| Metric | n | Mean | P50 | P90 | P95 | P99 | Max |
|---|---|---|---|---|---|---|---|
| Unserved energy (MWh) | 100 | 69.92 | 0.01 | 271.44 | 422.48 | 639.55 | 648.91 |
| Unserved fraction | 100 | 0.008 | 0.000 | 0.031 | 0.050 | 0.066 | 0.070 |
| Energy deficit (MWh) | 100 | 62.11 | 0.00 | 254.55 | 422.25 | 603.18 | 623.38 |
| Operating cost (USD) | 100 | 321,855 | 327,071 | 393,716 | 412,108 | 486,946 | 497,357 |
| Min reserve margin | 100 | −0.251 | −0.080 | 0.960 | 1.422 | 2.181 | 2.331 |
| Max line utilisation | 100 | 0.877 | 0.872 | 1.000 | 1.000 | 1.000 | 1.000 |
| **Recovery time (min)** | **60** | 110.58 | 107.50 | 215.00 | 250.50 | **461.40** | 485.00 |
| VRE penetration | 100 | 0.266 | 0.262 | 0.386 | 0.429 | 0.462 | 0.484 |
| Curtailment (MWh) | 100 | **0.000** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| CO₂ (t) | 100 | 1,800.3 | 1,862.7 | 2,078.9 | 2,119.2 | 2,284.1 | 2,403.4 |

**These are simulation-derived statistics, not empirical grid statistics.**
They describe this model's behaviour under *this* sampling distribution, and
that distribution is itself an assumption documented in `CampaignConfig`.
They are not frequencies of real-world events.

Percentile reporting is gated: a percentile is withheld unless at least one
observation backs its tail (P99 needs ≥100, P95 ≥20, P90 ≥10). The gate was
**corrected during this work** to count *finite observations* rather than
campaign size — `recovery_time_min` exists for only **60** of 100 samples
(40 samples never degraded), so its P99 is backed by 60 observations and is
labelled accordingly rather than being presented as a 100-sample statistic.
`test_percentile_gate_uses_finite_observations_not_the_campaign_size` pins
this.

Readings:

- **The median sample sheds essentially nothing (P50 = 0.01 MWh) while the
  upper decile sheds 271 MWh.** The distribution is strongly right-skewed:
  the system is secure under typical conditions and fails hard under
  coincident stress, which is the characteristic failure signature of a
  capacity-adequate but reserve-thin island.
- **Max line utilisation is exactly 1.000 at P90 and above**, i.e. in at
  least 10% of samples the network constraint binds. It is the second
  active limit after energy adequacy.
- **Curtailment is zero in all 100 samples**, consistent with §6.1: on this
  capacity mix curtailment requires network congestion, which the Monte
  Carlo does not randomise. That is a gap in the sampling design, stated
  rather than papered over.

---

## 15. Security — adversarial results

`[OBSERVED]` 20 attacks executed. Source: `results/fluxa/adversarial.json`.
Outcomes: **7 REJECTED, 8 DETECTED, 3 CONSTRAINED, 2 SUCCEEDED.**

| ID | Attack | Outcome | Defence that produced the outcome |
|---|---|---|---|
| A1 | Raise a generator's capacity in the loaded raw config after validation | REJECTED | `EnergySystem` is frozen and built once; the engine never re-reads `raw` |
| A2 | Assign to a live `Generator` field | REJECTED | `dataclasses.FrozenInstanceError` |
| A3 | Change one reactance by one float step (0.06 → 0.060000000000000005) | DETECTED | SHA-256 over canonical JSON with exact float `repr` |
| B1 | Rewrite a stored event's payload to conceal a violation | DETECTED | per-node content-hash recomputation (chain head unchanged — head comparison alone would miss it) |
| B2 | Edit an event **and rehash it** | DETECTED | `previous_hash` linkage break at the following node |
| B3 | Splice an event out of the ledger | DETECTED | chain linkage + monotonic sequence numbering |
| B4 | Append a fabricated `SYSTEM_STABILIZED` to a closed ledger | DETECTED | externally recorded root in the bundle no longer matches (ledger stays self-consistent — see §8.3) |
| C1 | Change one recorded timestep's load by 1 × 10⁻⁹ MW | DETECTED | Merkle tree over exact-precision state hashes |
| C2 | Inject 50 MW of phantom generation mid-run | REJECTED | in-loop `validate_physical_state` raises and halts |
| D1 | Present the clean baseline ledger as the compound run's audit trail | DETECTED | bundle binds `scenario_hash`, `profile_hash`, event root and state root together |
| D2 | Restore a checkpoint from a different scenario | REJECTED | experiment-addressed checkpoint ids; the foreign id is absent |
| E1 | Run the SENSITIVE scenario without authorization | REJECTED | `FatalInvariants.enforce_human_oversight`, checked before any physics |
| E2 | Relabel the gated scenario's name, description and tags as routine | REJECTED | level derived from perturbation kinds, not free text |
| **E3** | **Set `authorized=True` with an arbitrary `authorizer_id`** | **SUCCEEDED** | **NONE — see §11.3** |
| F1 | Demand 420 MW with both gas units out and the battery at `soc_min` | CONSTRAINED | SOC bounds hold; deficit reported as unserved energy, not drawn from an empty store |
| F2 | Demand 5,000 MW from 310 MW of dispatchable capacity | CONSTRAINED | capacity limits are LP variable bounds; 1,000+ MW reported unserved |
| F3 | Derate every line to 1 MW and require 380 MW of delivery | DETECTED | overload slack is a measured variable; reported magnitude equals flow − rating |
| F4 | Drive deep into surplus and look for negative power/energy | CONSTRAINED | non-negativity as LP lower bounds; all minima ≥ −10⁻⁹ |
| G1 | Execute three scenarios forward then reversed in one process | REJECTED | no mutable module state; every RNG explicitly constructed |
| **H1** | **Forge a `qradle.core.merkle.MerkleProof`** | **SUCCEEDED** | **NONE in QRADLE — see §8.2** |

### The two successful attacks

**E3 — self-asserted authorization.** Any caller able to construct a
`RunConfig` can authorize itself for a SENSITIVE operation. The gate records
a claim; it does not verify one. §11.3.

**H1 — forgeable QRADLE Merkle proof.** A proof with a fabricated event hash
and a garbage path verifies, because verification is root-equality. FLUXA's
own recomputing proof rejects the equivalent forgery. §8.2.

Both are **architectural**, not implementation slips: E3 is the shape of
QRADLE's authorization signature; H1 is the body of
`MerkleProof.verify`. Neither can be fixed from inside a vertical.

### What the three CONSTRAINED outcomes mean

F1, F2 and F4 are the cases where the right answer is neither "reject" nor
"detect" but "clamp and report". Asked for 5,000 MW from a 310 MW system,
FLUXA does not fail and does not fabricate capacity: it dispatches every unit
to nameplate and reports the balance as unserved energy. That is the correct
behaviour for a planning tool — the violation becomes a *measurement*.

---

## 16. Validation

### 16.1 FLUXA test suite

**297 tests, all passing.**

| Module | Tests | Covers |
|---|---|---|
| `test_units.py` | 8 | MW/MWh, kW/MW, per-unit conversions and their inverses |
| `test_model.py` | 17 | every physically impossible description is rejected |
| `test_network.py` | 10 | PTDF vs independent angle solution, Kirchhoff, linearity, DC self-consistency, disconnection |
| `test_dispatch.py` | 22 | balance, capacity, ramp (binding and relaxed), SOC, storage conservation, network limits, merit order, energy-tested reserve |
| `test_conservation.py` | 16 | per-step and integrated energy conservation, storage balance, capacity, availability, overload consistency, CO₂ accounting |
| `test_scenarios.py` | 26 | trapezoid shapes, perturbation application, non-mutation, safety classification, hashing |
| `test_determinism.py` | 31 | 3-way replay at bit level, input sensitivity, seed independence, QRADLE wall-clock non-determinism |
| `test_provenance.py` | 35 | Merkle trees at 10 sizes, inclusion proofs, forgery rejection, the 3 required tests, QRADLE proof limitation |
| `test_rollback.py` | 16 | detect/authorize/restore/verify/resume, tampered checkpoints, strict-mode halt |
| `test_authorization.py` | 27 | gate blocks before physics, denial is audited, relabelling fails, level mapping, never EXISTENTIAL |
| `test_adversarial.py` | 35 | all six attack classes plus execution-order independence |
| `test_metrics.py` | 26 | metric identities, resilience ordering, no fabricated standard indices |
| `test_montecarlo.py` | 13 | sample reproducibility, size-independence, percentile gating |
| `test_io.py` | 11 | uniform schema across scenarios, Parquet round-trip, artefact completeness |
| `test_storage_value.py` | 4 | the terminal-value finding of §8.4 |

Every claim in §1 traces to a test or a result artefact.

### 16.2 Pre-existing QRATUM test suite

Run before and after, per-file with an identical protocol
(`pytest <file> -q --continue-on-collection-errors`, 180 s per file).

Scope: the 97 test files present both before and after.

| | Before | After |
|---|---|---|
| Passed | 2,140 | **2,140** |
| Failed | 80 | **80** |
| Collection errors | 6 | **6** |
| Skipped | 4 | **4** |
| Files whose result changed | — | **0 of 97** |

**No regression: not one pre-existing test file changed its pass/fail
counts.** `[OBSERVED]` The 297 FLUXA tests are additional, in 15 new files. The 80 pre-existing failures and 6 collection
errors are unchanged and unrelated to FLUXA: they are in
`tests/test_opt.py` (21), `tests/test_qcmg_sim.py` (31),
`tests/test_pr000_spine.py` (9), `tests/test_ciir_*` (9),
`tests/test_qunimbus_global_rollout.py` (3),
`tests/test_video_autocapture.py` (4), and others, plus collection errors
from missing optional dependencies. None was introduced or fixed here.

One pre-existing failure is worth naming because it sits in the substrate
FLUXA depends on: `qradle/tests/test_merkle.py::test_proof_generation`
asserts `chain.get_proof(2).event_hash == node3.node_hash`, but index 2 is
the *second* appended event (index 0 is genesis), so the assertion is simply
wrong about the chain's indexing. It is a defect in the test, not the code,
and it was left unchanged to keep the baseline comparison clean.

No pre-existing file in the repository was modified. FLUXA is additive:
`fluxa/`, `results/fluxa/`, and this report.

---

## 17. Limitations

Model: `[ASSUMED]` unless noted.

1. **DC power flow.** No voltage magnitudes, no reactive power, no losses, no
   voltage stability. Reported costs omit losses and are optimistic by order
   2–4%.
2. **No unit commitment.** Continuous relaxation: no start costs, no minimum
   up/down times, no start-up trajectories. `p_min > 0` is a must-run.
3. **No true topology outage.** `LINE_DERATE` reduces a rating; it does not
   open a branch. A real N−1 line outage changes the PTDF, which FLUXA does
   not recompute mid-run. `[NOT IMPLEMENTED]`
4. **Frequency is a flagged proxy**, not a dynamic simulation. No inertia, no
   governor response, no RoCoF. Beyond ±2 Hz the proxy is explicitly out of
   range (32 such steps in the compound event).
5. **Storage result is dominated by a terminal-value assumption** whose
   base-case choice is demonstrably not cost-minimising (§8.4).
6. **Reserve is a deterministic linear proxy**, not a probabilistic adequacy
   calculation. The battery's contribution is energy-tested for 15 minutes
   only.
7. **Receding horizon with perfect in-window foresight.** No forecast error
   inside the 1-hour window. Real dispatch faces forecast error at every
   horizon, which this omits — so the reported costs are optimistic and the
   reported unserved energy is optimistic.
8. **Synthetic profiles.** Load, irradiance and wind are analytic shapes plus
   seeded AR(1) noise, not measurements.
9. **Single battery, single tie, four generators.** No hydro, no nuclear, no
   demand response, no electric vehicles, no distributed resources.
10. **Curtailment arises only from congestion** on this capacity mix, and the
    Monte Carlo does not randomise network ratings, so all 100 samples show
    zero curtailment.
11. **Bit-reproducibility is established on one platform and one
    SciPy/HiGHS build.** Cross-platform and cross-BLAS reproducibility is
    untested. `[HYPOTHESIS]`
12. **Economics are stylised.** No market clearing, no bid curves, no
    forward contracts, no capacity payments, no carbon price in the base
    case.

---

## 18. What was NOT demonstrated

This section is mandatory, and it is long on purpose.

**Not demonstrated by this experiment:**

1. **Quantum acceleration of anything.** No QPU, no quantum simulator, no
   hybrid algorithm. `[NOT IMPLEMENTED]`
2. **GPU, HPC or edge execution.** Everything ran on 4 CPU cores. The
   `ComputeSubstrate` abstraction was never invoked. `[NOT IMPLEMENTED]`
3. **Authentication of authorization.** The gate records a claim; it never
   verifies one. Attack E3 succeeded. `[NOT IMPLEMENTED]`
4. **Dual control.** `qradle/core/zones.py` implements it; the
   `DeterministicEngine` authorization path does not use it.
   `[NOT IMPLEMENTED]`
5. **Security-zone enforcement.** Z0–Z3 exist in QRADLE and FLUXA does not
   pass through `ZoneDeterminismEnforcer`. `[NOT IMPLEMENTED]`
6. **A load-bearing QRADLE Merkle proof.** QRADLE's proof is forgeable (H1);
   FLUXA supplied its own. QRADLE's proof was not fixed. `[NOT IMPLEMENTED]`
7. **Reproducible QRADLE chain roots.** They are wall-clock dependent and
   remain so. `[NOT IMPLEMENTED]`
8. **Separate custody of the provenance bundle.** Tamper evidence for log
   *extension* depends on the bundle being held separately; nothing enforces
   that. `[NOT IMPLEMENTED]`
9. **Distributed or multi-party verification.** Single process, single
   machine, no external witness, no third-party attestation.
10. **Any validation against a real power system.** No measured load, no
    measured generation, no historical event, no comparison against a
    production tool (PSS/E, PowerFactory, PyPSA, PLEXOS). **The model has
    never been validated against reality.** `[NOT IMPLEMENTED]`
11. **Cross-platform bit-reproducibility.**
12. **Rollback across process boundaries.** Checkpoints live in an in-memory
    `RollbackManager`; nothing was restored after a restart.
13. **Scalability beyond 6 buses and 100 scenarios.** Nothing here speaks to
    a 10,000-bus network or a 10,000-sample campaign.
14. **That the architecture improves decision quality.** It makes results
    reproducible and auditable. Whether that leads to better operating
    decisions is untested and is a question about organisations, not code.

---

## 19. Scientific falsification

The brief asks ten questions. Answers, with evidence.

**1. Is the simulation actually deterministic?**
**Yes at the FLUXA level, no at the QRADLE level.** 3 × 3 replays gave
bit-identical state hashes, event sequences, Merkle roots and metric hashes.
QRADLE's own chain root differed in all three replays of all three scenarios.
`[OBSERVED]` §7. Three determinism defects had to be fixed first (§9.2–9.4) —
the property was *not* free.

**2. Is the provenance chain actually tamper-evident?**
**Yes, with two qualifications.** Every modification attempted was detected
(§8.1, §15 B1–B4, C1, A3). Qualification one: detecting an *edited* event
requires per-node content recomputation — the chain head does not move, so a
head comparison alone fails. Qualification two: detecting *extension* of a
closed log requires an external witness; the log stays self-consistent
(§8.3). `[OBSERVED]`

**3. Does rollback actually restore state?**
**Yes.** Three experiments, including one mid-disturbance. The restored state
hash matched the recorded one exactly, and 2–5 resumed steps were
bit-identical to the uncorrupted reference. `[OBSERVED]` §10.

**4. Does authorization actually prevent unauthorized operations?**
**It prevents unauthorized operations; it does not prevent self-authorization.**
E1 and E2 were rejected before any physics ran. E3 succeeded. The gate is an
auditability control, not an access control. `[OBSERVED]` §11.3.

**5. Does the system preserve physical constraints?**
**Yes, under adversarial pressure.** Power balance closed to <10⁻⁶ MW at
every one of 288 timesteps in every scenario; the integrated storage balance
closed to ~10⁻¹⁴ MWh; SOC never left its window; no generator exceeded
nameplate; renewable output never exceeded the available resource. Three
attacks designed to force a violation (F1, F2, F4) were clamped and reported
rather than absorbed. An injected 50 MW phantom generation halted the run.
`[OBSERVED]` §16.1, §15.

**6. Does the architecture introduce measurable overhead?**
**Yes, and it is small: 2.49%.** 529 events, 12 verified checkpoints and a
full provenance bundle cost 0.037 s on a 1.474 s run, with bit-identical
physics. `[OBSERVED]` §13.4. Caveat: the LP is 94% of runtime, so the
relative overhead would be larger for a cheaper kernel.

**7. Does the abstraction improve anything versus a conventional simulation?**
**Three things, measurably; one thing, not at all.**
Measurably: (a) bit-exact replay makes a 4-core parallel campaign
trustworthy — 3.80× speed-up with *identical* per-sample identities, which
an uncontrolled simulation cannot assert; (b) tamper-evident provenance
turns "trust the operator's spreadsheet" into a verifiable claim, at 2.49%
cost; (c) an independent physical-invariant check caught a deliberately
corrupted state the solver would have reported as fine. `[OBSERVED]`
Not at all: the architecture does **not** improve the *physics*. The DC
approximation, the LP relaxation and the synthetic profiles are exactly as
accurate as they would be in a bare script. QRADLE makes results
*reproducible and auditable*, not *correct*. A deterministic, fully audited
simulation of a wrong model is a wrong answer with excellent provenance.

**8. Which claimed QRATUM capabilities remain architectural rather than
demonstrated?**
Fourteen, enumerated in §18. The load-bearing ones: authentication, dual
control, zone enforcement, GPU/HPC/QPU execution, distributed verification,
and any validation against a real power system.

**9. What is the strongest evidence against the architecture?**
Three items, in order of force:

- **E3 and H1 together.** Two of QRADLE's named guarantees — "human
  oversight" and "Merkle proof" — do not do what their names claim. The
  oversight invariant records an unverified boolean; the Merkle proof
  verifies by comparing a field it carries to itself. Both are reachable in
  three lines of code. An architecture whose security properties are stated
  as "8 Fatal Invariants" invites the reading that they are enforced; two are
  not, in the specific sense an attacker cares about.
- **The storage-value result (§8.4).** A single documented modelling
  parameter, chosen on plausible-looking grounds, made the base-case dispatch
  1.25% more expensive and suppressed battery cycling by 3×. Determinism and
  provenance gave *perfect* reproducibility of a suboptimal answer. The
  architecture's guarantees are orthogonal to whether the answer is any good,
  and they can make a bad answer look authoritative.
- **No validation against reality.** Every number in this report describes a
  synthetic 6-bus system with assumed parameters. Nothing establishes that
  FLUXA's dispatch resembles a real system's, and the report's provenance
  machinery provides no evidence whatsoever on that question.

**10. What experiment should be performed next?**
In priority order:

1. **Wire `ZoneContext.has_dual_control` into the
   `DeterministicEngine` authorization path and re-run attack E3.** The
   mechanism already exists in QRADLE; the integration does not. This is the
   cheapest available conversion of an architectural claim into a
   demonstrated one.
2. **Make `MerkleProof.verify` recompute a root, and re-run H1.** Then delete
   FLUXA's parallel implementation.
3. **Give `MerkleChain` an injectable clock** (defaulting to
   `datetime.now`) and re-run the determinism experiment against QRADLE's own
   chain root.
4. **Validate the model against a public reference.** Reproduce a published
   test system (IEEE RTS-96 or a PyPSA example) with FLUXA and compare
   dispatch and flows against the reference implementation. Until this is
   done, no FLUXA number should inform any real decision.
5. **Add unit-commitment binaries** (a MILP) and measure the cost of the
   continuous relaxation. This also creates the honest baseline any future
   quantum-optimisation comparison would need.
6. **Introduce forecast error** between horizon windows and re-measure the
   resilience metrics. Current results assume perfect 1-hour foresight and
   are therefore optimistic by an unknown amount.
7. **Run the determinism experiment on a second platform and a second BLAS**
   to test whether bit-reproducibility survives the numerical stack.

---

## 20. Reproducibility

### Exact commands

```bash
# 1. Dependencies (numpy and scipy are required; the rest are optional)
pip install numpy scipy pytest pandas pyarrow psutil

# 2. Inspect the inputs and their hashes
python -m fluxa.cli list
#    expect: config_hash b3fc57aaf4e8aaf02b370953789058dc03ec27b545907f2f015f0a75bd31d8f7
#            model_hash  fac1fd29898a3dbc9547280635762a556435afe641833b817b100e6cf2ea59d9

# 3. The validation suite (297 tests)
python -m pytest fluxa/tests/ -q -o addopts=""

# 4. THE FULL CAMPAIGN — every number in this report
python -m fluxa.experiments.run_all
#    → results/fluxa/  (72 files, ~20 MB, ~361 s on 4 cores)

# 5. Individual experiments
python -m fluxa.cli run --scenario A_BASELINE --steps 288
python -m fluxa.cli run --scenario F_COMPOUND --steps 288 --authorized
python -m fluxa.cli run --scenario F_COMPOUND --steps 288    # refused: InvariantViolation
python -m fluxa.cli rollback
python -m fluxa.cli montecarlo --samples 100 --parallel

# 6. Re-verify an exported ledger from disk, independently of any run
python -m fluxa.cli verify \
  --events results/fluxa/scenarios/primary_A_BASELINE_events.json

# 7. Regression check against the pre-existing suite
#    (run per file, 180 s each, to match the protocol used for the baseline)
for f in tests/*.py qradle/tests/*.py qratum/tests/*.py; do
  timeout 180 python -m pytest "$f" -q -o addopts="" --continue-on-collection-errors
done
#    expect the unchanged baseline, aggregated:
#      2140 passed, 80 failed, 6 collection errors, 4 skipped
```

### Reduced-scale campaign

```bash
python -m fluxa.experiments.run_all \
  --monte-carlo-samples 10 --extended-steps 576 \
  --scaling-counts 1,10 --skip-benchmark      # ~60 s
```

### Artefacts

| Artefact | Contents |
|---|---|
| `results/fluxa/manifest.json` | every stage, timing, hash, artefact path and failure |
| `results/fluxa/primary_scenario_summary.csv` | one row per scenario (§6.1) |
| `results/fluxa/scenarios/primary_<ID>_states.{csv,parquet}` | 288 rows × 89 columns per scenario |
| `results/fluxa/scenarios/primary_<ID>_events.json` | full ledger with per-node and previous hashes |
| `results/fluxa/scenarios/primary_<ID>_{provenance,metrics,summary}.json` | the bundle, all metric families, the summary |
| `results/fluxa/extended/extended_<ID>_*` | the 7-day runs, 2,016 steps each |
| `results/fluxa/determinism.json` | §7 |
| `results/fluxa/provenance.json` | §8.1 |
| `results/fluxa/rollback.json` + `rollback_*_hashes.json` | §10 |
| `results/fluxa/monte_carlo.json`, `monte_carlo_samples.csv` | §14 |
| `results/fluxa/adversarial.json` | §15, with per-attack evidence |
| `results/fluxa/benchmark.json` | §13 |
| `results/fluxa/storage_value_sensitivity.json` | §8.4 |

Every run records its own environment (Python, platform, NumPy, SciPy) and
solver identity (library, method, version, problem size) in its provenance
bundle, so a future divergence can be attributed to the numerical stack
rather than to the model.

### What is committed versus what is regenerable

`results/` is listed in the repository's `.gitignore` (line 106), a
pre-existing convention this work did not change. The campaign's **decisive
evidence** is therefore force-added so every table above is verifiable
directly from the repository (87 files, 1.5 MB): the manifest, all eight
experiment JSONs, the Monte Carlo per-sample CSV, the storage sensitivity,
the scenario summary CSV, and every per-scenario `_summary` / `_metrics` /
`_provenance` document for both the primary and extended runs, plus one
complete event ledger (`primary_A_BASELINE_events.json`, 529 events) so the
off-line verification below can be run without re-executing anything.

The **bulk artefacts are regenerable and not committed**: the per-timestep
state tables (`*_states.csv` and `*_states.parquet`, 288 or 2,016 rows ×
89 columns each) and the remaining full event ledgers, about 18 MB in total.
`python -m fluxa.experiments.run_all` reproduces all of them, and because the
run is bit-reproducible (§7) the regenerated files are identical to the ones
these numbers came from.

An independent off-line check of the committed ledger, with no simulation
run:

```bash
$ python -m fluxa.cli verify \
    --events results/fluxa/scenarios/primary_A_BASELINE_events.json
{
  "n_events": 529,
  "stored_chain_head":     "bb2b34a372eb5ff7cd468d7d356474b06afa338f76b3bd92df82fe3e2c3fd9b8",
  "recomputed_chain_head": "bb2b34a372eb5ff7cd468d7d356474b06afa338f76b3bd92df82fe3e2c3fd9b8",
  "chain_head_matches": true,
  "recomputed_event_tree_root": "094733de8091866d8cd96dffdb2e3b6dc438d3dd9784c01952a6332e663c23ca",
  "problems": [],
  "verified": true
}
```

The recomputed chain head and event-tree root match
`primary_A_BASELINE_provenance.json` exactly, from a tool that rebuilds every
node hash from the stored event bodies. `[OBSERVED]`

---

## 21. Research implications

**What this experiment establishes.**

QRATUM's domain/substrate split is **real and it works**. FLUXA is a genuine
domain implementation — a DC power-flow network, a receding-horizon dispatch
LP, a scenario engine, physical-invariant validation — and it sits on QRADLE
without either layer knowing much about the other. The substrate contributed
deterministic execution, an authorization gate, event emission, Merkle
hashing primitives and working rollback. The vertical contributed physics.
Neither needed to be redesigned for the other. `[OBSERVED]`

The **measurable** benefits are three, and they are not nothing:

1. Bit-exact replay makes a parallel campaign trustworthy. 3.80× speed-up on
   4 cores with provably identical per-sample results. Without determinism,
   "we ran it on more cores and got slightly different numbers" is
   unfalsifiable.
2. Tamper-evident provenance at **2.49%** runtime cost. For any
   simulation whose output informs a regulated decision, that is a very
   cheap audit trail. Every modification attempted was detected.
3. An independent physical-invariant layer caught a corrupted state that the
   solver reported as optimal. Separating "the solver converged" from "the
   answer obeys physics" has measurable value.

**What this experiment does not establish, and the distinction matters most.**

QRADLE makes results **reproducible and auditable**. It does not make them
**correct**. §8.4 is the proof: a deterministic, fully audited, perfectly
reproducible dispatch that is 1.25% more expensive than it needs to be,
because one documented parameter was chosen badly. Determinism reproduced the
suboptimal answer flawlessly. Provenance attested to it cryptographically.
Neither mechanism could notice.

That is the central research implication. An architecture that guarantees
reproducibility and auditability shifts, rather than removes, the burden of
correctness — and it can make a wrong answer *more* persuasive, because the
answer now arrives with hashes, an audit trail and a verification routine
that all pass. The appropriate response is not to abandon the guarantees but
to pair them with the thing this experiment conspicuously lacks:
**validation against an external reference** (§19.10, §19 item 10 of the next
steps). A reproducible simulation that has never been checked against reality
is a reproducible guess.

**On the invariant framing.** QRADLE presents "8 Fatal Invariants" as
immutable and enforced. Six behave as advertised in the paths FLUXA
exercises. Two do not, in the sense an adversary cares about: invariant 1
(human oversight) accepts an unverified self-asserted boolean, and the Merkle
proof that supports invariant 2's auditability story verifies by comparing a
field to itself. Both gaps are three lines of code from being closed, and in
one case the required mechanism is **already in the repository**
(`ZoneContext.has_dual_control`). The gap is integration, not design — but
until it is closed, "enforced invariant" overstates what the code does, and
this report declines to repeat the claim.

**Falsification worked.** Running the experiment adversarially rather than
confirmatorily produced seven defects (§9) and two successful attacks (§15)
that inspection and a confirmatory test suite would both have missed: an
infeasible generator trip, two label leaks into cryptographic identities, a
wall-clock checkpoint id, a metric reporting 3.91 as an efficiency, a
stabilisation deadlock, a ragged export schema, a forgeable substrate proof
and a self-asserted authorization. That is the method's return on investment,
and it is the strongest procedural recommendation this report can make: an
architecture that claims determinism, auditability and safety enforcement
should be tested by someone trying to break those claims, with the failures
published.

---

*Generated by executing `python -m fluxa.experiments.run_all` and
`python -m pytest fluxa/tests/`. Every table in this report is populated from
an artefact in `results/fluxa/`. No number was transcribed by hand from a
prediction, and no result was omitted because it was unflattering.*
