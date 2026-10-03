# FLUXA — the QRATUM energy-system vertical on QRADLE

FLUXA models an electrical energy system (generation, demand, storage,
transmission) and runs it as a **QRADLE domain contract**, so every state
transition is deterministic, authorized, event-logged, Merkle-chained and
recoverable.

```
FLUXA  (this package)
  ├── model.py      typed, frozen, hashable system description
  ├── network.py    linearised (DC) power flow → PTDF
  ├── profiles.py   deterministic, seeded load / solar / wind drivers
  ├── scenarios.py  declarative perturbation engine + safety classification
  ├── dispatch.py   receding-horizon economic dispatch LP (SciPy/HiGHS)
  ├── state.py      machine-readable per-timestep state + classification
  ├── events.py     FLUXA event taxonomy + deterministic Merkle ledger
  ├── provenance.py binary Merkle tree, inclusion proofs, provenance bundle
  ├── engine.py     the QRADLE-integrated run loop
  ├── recovery.py   checkpoint / detect / authorize / restore / resume
  ├── metrics.py    reliability, economic, renewable, resilience metrics
  ├── montecarlo.py seeded stochastic campaign
  ├── benchmark.py  runtime, scaling, parallel, provenance overhead
  └── io.py         JSON / CSV / Parquet artefacts
        │
        ▼
QRADLE  (qradle/, pre-existing)
  DeterministicEngine · FatalInvariants · MerkleNode · RollbackManager
        │
        ▼
Q-Substrate
  CPU only. GPU / HPC / QPU abstractions exist in this repository but are
  NOT exercised by FLUXA. No quantum acceleration is demonstrated.
```

## Name collision

`qratum/verticals/fluxa.py` and `verticals/fluxa.py` in this repository
implement a **supply-chain and logistics** module also called FLUXA. This
package is the **energy-system** FLUXA. They share a name and nothing else;
neither imports the other, and no existing file was modified.

## Quick start

```bash
pip install numpy scipy pytest          # pandas/pyarrow/psutil optional
python -m fluxa.cli list                # system + scenario library with hashes
python -m fluxa.cli run --scenario A_BASELINE --steps 288
python -m fluxa.cli run --scenario F_COMPOUND --steps 288 --authorized
python -m fluxa.cli rollback
python -m fluxa.cli montecarlo --samples 100 --parallel
python -m fluxa.experiments.run_all     # the full campaign → results/fluxa/
pytest fluxa/tests/ -q                  # the validation suite
```

`--authorized` is required for scenarios D and F. Without it QRADLE raises
`InvariantViolation` and the simulation never starts.

## Model summary

A six-bus island system: a 220 MW CCGT, a 90 MW peaking GT, a 180 MW PV
farm, a 150 MW wind farm, a 320 MWh / 80 MW battery, a ±100 MW
interconnection, eight transmission corridors, 420 MW peak demand.

**Power flow** is the linearised DC approximation: fixed unit voltage
magnitudes, negligible branch resistance, small angle differences, no
reactive power. Real-power flows are therefore an exact linear function of
bus injections, which is what lets network limits enter the LP directly. The
cost is that FLUXA cannot speak to voltage collapse, reactive adequacy or
losses. Solved angle spreads are asserted to stay under 0.35 rad so the model
remains consistent with its own linearisation.

**Dispatch** is a receding-horizon LP over a 12-step (1 hour) look-ahead with
perfect foresight inside the window, solved with
`scipy.optimize.linprog(method="highs")`. The objective, every constraint and
every approximation are documented in `fluxa/dispatch.py`'s module docstring.

**Parameter origins** — every number, classified as assumed, conventional or
derived — are in `configs/PARAMETER_SOURCES.md`. No parameter is an
observation of a real power system.

## Determinism

FLUXA is bit-reproducible: replaying a scenario with identical inputs gives
identical per-timestep state hashes, event sequence, Merkle roots and metric
hashes. Three substrate and design defects had to be fixed to achieve this:

1. The run **label** leaked into `FluxaState.state_hash()`, making the state
   tree root depend on what the run was called. Labels are now excluded from
   the content hash.
2. The QRADLE contract id included the run label, and QRADLE folds the
   contract id into `ExecutionResult.output_hash`. The contract id is now
   addressed by *experiment* (system + scenario + run parameters).
3. QRADLE's auto-generated `checkpoint_id` embeds
   `int(datetime.now().timestamp())`, so checkpoint ids — and therefore the
   `CHECKPOINT_CREATED` events and the whole event-chain root — were
   wall-clock dependent. FLUXA supplies explicit deterministic ids.

One substrate non-determinism **remains and is reported, not worked around**:
`qradle.core.merkle.MerkleChain.append` stamps every node with
`datetime.now(timezone.utc)`, so QRADLE's own chain root differs between
identical runs. `qradle_chain_root` is therefore excluded from FLUXA's
provenance identity, and `test_qradle_native_chain_root_is_wall_clock_dependent`
asserts the non-determinism so the claim stays honest if upstream changes.

FLUXA's ledger reuses QRADLE's `MerkleNode` hashing primitive byte for byte
but supplies the **simulation** timestamp, which makes the chain reproducible
with unchanged hashing semantics.

## Provenance

`fluxa/provenance.py` builds a true binary Merkle tree with domain-separated
leaf and internal-node hashes, and inclusion proofs that **recompute the root
from the leaf and sibling path**. This is not redundant with QRADLE:
`qradle.core.merkle.MerkleProof.verify` returns
`self.root_hash == claimed_root` and never recomputes anything, so a proof
with a fabricated `event_hash` and a garbage `proof_path` verifies
successfully. That forgery is executed in the adversarial campaign
(`H1_FORGE_QRADLE_MERKLE_PROOF`) and reported as a substrate defect.

## Safety levels

| Level | FLUXA meaning |
|---|---|
| ROUTINE | ordinary state advancement |
| ELEVATED | major dispatch change, material storage action, or eroded security margin |
| SENSITIVE | a forced generator outage is in effect — **authorization-gated** |
| CRITICAL | load shedding, or ≥2 simultaneous constraint violations |
| EXISTENTIAL | **never emitted.** An ordinary grid contingency is not an architecture-level catastrophe. |

A step that both has a unit out *and* is shedding load is CRITICAL, not
SENSITIVE: the worse condition wins so an outage cannot mask load shedding in
the audit trail.

## Known architectural gap

`RunConfig.authorized` is a boolean asserted by the caller. FLUXA and QRADLE
verify that an authorization flag was *set*, not that a human set it — there
is no signature, credential or second party. Anything able to construct a
`RunConfig` can authorize itself. This is executed as attack
`E3_SELF_ASSERTED_AUTHORIZATION` and **succeeds**. QRADLE's `zones` module
implements dual control (`ZoneContext.has_dual_control`) but the
`DeterministicEngine` authorization path does not use it.

## Artefacts

`python -m fluxa.experiments.run_all` writes to `results/fluxa/`:

```
manifest.json                       every stage, timing, hash and failure
primary_scenario_summary.csv        one row per scenario
scenarios/primary_<ID>_states.csv   288 rows × 89 columns per scenario
scenarios/primary_<ID>_states.parquet
scenarios/primary_<ID>_events.json  full ledger with per-node hashes
scenarios/primary_<ID>_provenance.json
scenarios/primary_<ID>_metrics.json
extended/extended_<ID>_*            the 7-day runs (2016 steps)
determinism.json  provenance.json  rollback.json
monte_carlo.json  monte_carlo_samples.csv
adversarial.json  benchmark.json
```

An exported ledger can be re-verified from disk, independently of any run:

```bash
python -m fluxa.cli verify --events results/fluxa/scenarios/primary_A_BASELINE_events.json
```

## Full report

`FLUXA_QRATUM_SIMULATION_REPORT.md` at the repository root.
