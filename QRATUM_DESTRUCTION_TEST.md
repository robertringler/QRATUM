# QRATUM — Destruction / Dependency Test

**Method:** remove each major component and determine, from the import graph and the test suites, what still works and how much strategic value is lost. Destruction percentages are `[ESTIMATE]` derived from the scorecard in `tools/value_analysis/qratum_value_scorecard.py`; the *dependency findings* are `[OBSERVED]` from the code.

Reproduce the dependency findings:

```bash
grep -rln "import qradle\|from qradle" --include=*.py .     # 57 files
grep -l  "qradle" verticals/*.py                            # no output
grep -rln "class MerkleTree\|class MerkleChain\|previous_hash\|prev_hash" --include=*.py --include=*.rs . | wc -l   # 34
```

---

## 1. RESULTS AT A GLANCE

| Removal | Builds still? | Tests still pass? | Strategic value destroyed `[ESTIMATE]` |
|---|---|---|---|
| **Arbiter + corpus removed** | Yes | Yes (`os/` suite vanishes) | **45–50%** ← largest |
| Deterministic execution removed (`os/` layer) | Yes | No — 163 tests collapse | **35%** |
| Cryptographic provenance removed | Yes | Mostly | **15%** |
| Evidence-discipline apparatus removed | Yes | Yes | **20%** (asymmetric — see §7) |
| **QRADLE removed** | Almost entirely | Yes | **~5%** |
| 14 verticals removed | Yes | Yes | **~3%** |
| Q-Substrate removed | Yes | Yes | **~3%** |
| QRATUM-ASI removed | Yes | Yes | **~2%** (arguably negative — see §5) |

---

## 2. IF QRADLE WERE REMOVED — what remains?

**Almost everything.** This is the most consequential result in the destruction test, because QRADLE is the stated foundation of the architecture.

`[OBSERVED]` 57 Python files reference `qradle`. Breaking them down:

| Consumer group | Files | Breaks? | Is it a working product? |
|---|---:|---|---|
| `qradle/` itself (incl. its own tests) | 12 | n/a | — |
| `qratum_asi/sandbox_platform/*` | ~38 | Yes | **No** — `qratum_asi` is self-declared THEORETICAL `[OBSERVED]` |
| `qratum/platform/` (`api.py`, `reasoning_engine.py`) | 2 | Yes | No — no customers, no deployment |
| `qratum_fullstack_server.py` | 1 | Yes | No — a demo server |
| **The 14 verticals** | **0** | **No** | They import `qratum_platform` + `contracts` instead |
| `os/qratum-os/` (Rust) | 0 | No | Fully independent |
| `quasim/` (67k LOC) | 0 | No | Independent |
| `contracts/` | 0 | No | Independent — and provides the same role |
| `qledger/`, `q-substrate/`, `aion/`, `Aethernet/` | 0 | No | Independent |

**What survives QRADLE's removal:** the arbiter and the entire `os/` subsystem, all 14 verticals, QuASIM in full, `contracts/` (which already does provenance for the verticals), `qledger`, Q-Substrate, AION, Aethernet, HCAL, the claim registry, and the TLA+ specs.

**What dies:** a theoretical ASI sandbox, a demo server, and two platform files.

**Destruction: ~5%.** `[ESTIMATE]` The narrative loses its keystone — every README describes QRADLE as the foundational layer — while the technology loses almost nothing. **QRADLE is a narrative dependency, not a technical one.**

---

## 3. IF THE ARBITER WERE REMOVED — what remains?

`[DERIVED]` This is the costliest removal.

**What survives:** QRADLE's eight invariants, four of which are never invoked or invoked with hardcoded `true` `[TESTED]`; 20+ parallel hash-chain implementations with no shared semantics `[OBSERVED]`; an 8,163-line `no_std` kernel with no distinctive reason to exist, competing against seL4 (machine-checked, free) and QNX (already certified); HCAL, which survives independently; the claim registry, which would then have almost nothing substantive left to certify; 14 TLA+ specs, the central one of which (`arbitration_invariance.tla`) now models nothing.

**What is lost:** the only primitive that survives the substitution test. The only functioning conformance suite in the repository (270 committed scenarios with pinned hashes). The only articulable technical mechanism worth showing to patent counsel. The only credible answer to "what problem does QRATUM solve that nothing else does?"

**Destruction: 45–50%.** `[ESTIMATE]` After this removal, every remaining component has a free, mature substitute that is better (see `QRATUM_COMPETITIVE_SUBSTITUTION.md`). **The repository becomes a large collection of reimplementations of commodity software.**

This is the central evidence for the determination: it is the removal that leaves nothing distinctive behind.

---

## 4. IF THE 14 VERTICALS WERE REMOVED — what remains?

`[OBSERVED]` **Everything.** No core component imports any vertical. The import direction is strictly one-way: verticals → `qratum_platform` + `contracts`.

**Destruction: ~3%.** `[ESTIMATE]` And the verticals contain no domain value to lose: 4,403 LOC across 16 files (~280 each), FLUXA's optimizer is greedy nearest-neighbour carrying the in-source note `"Simplified heuristic - use OR-Tools for production"` `[OBSERVED]`, defaults are hard-coded four-location toy data, and `random.seed(seed)` mutates process-global RNG state — a determinism hazard in a system whose central claim is determinism.

A secondary consequence worth naming: removing the verticals removes the "14 verticals" claim, which is load-bearing for the *breadth* narrative and contributes nothing to the *value* case. §11 of the main report measures QRADLE's leverage across them as exactly zero.

---

## 5. IF QRATUM-ASI WERE REMOVED — what remains?

`[OBSERVED]` **Everything.** Nothing in the repository imports `qratum_asi`. The dependency runs the other way: `qratum_asi/sandbox_platform/*` imports `qradle`.

**Destruction: ~2%, and plausibly negative.** `[ESTIMATE]`

The argument for negative destruction is concrete rather than rhetorical. `qratum_asi/` is 50,325 LOC — the third-largest subsystem — and:

- Its README opens: *"**This is a THEORETICAL ARCHITECTURE.** The components described require fundamental AI breakthroughs that have not yet occurred."* `[OBSERVED]`
- 50 of 129 files (39%) contain placeholder language `[OBSERVED]`.
- Its flagship cryptographic component, `zk_state_verifier.py`, states *"The placeholder accepts any well-formed proof for demonstration"* and registers verifying keys as the literal bytes `b"placeholder_vk_"` `[OBSERVED]`.
- It is framed as "Sovereign Superintelligence Architecture" with pillars named Q-WILL (autonomous intent generation) and Q-EVOLVE (self-improvement).

For the buyers identified in §20 of the main report — a utility's Chief Risk Officer, a defence programme's chief engineer, a medical-device quality lead — a repository containing a self-declared superintelligence layer with placeholder cryptography is a **reason not to proceed**, independent of the arbiter's merits. Removal increases the credibility of what remains with the exact audience the valuable asset must sell to.

---

## 6. IF Q-SUBSTRATE WERE REMOVED — what remains?

`[OBSERVED]` **Everything.** A standalone Rust crate (5,848 LOC) with no inbound dependencies from the Python tree or from `os/`.

**Destruction: ~3%.** `[ESTIMATE]` The one genuinely interesting property lost is the engineering result "<500 KB binary, ≤32 MB footprint, deterministic, air-gap capable". That is real but has not supported a business on its own, and substitutes (`llama.cpp` + Qiskit Aer + `wasmtime`) beat it on every axis except size.

---

## 7. IF CRYPTOGRAPHIC PROVENANCE WERE REMOVED — what remains?

**Execution and arbitration survive intact.** The arbiter's `arbitrate()` is a pure function of `(Intent, LockState, Tick)` and does not depend on hashing. The 270-scenario corpus *does* pin `lock_state_hash` and `audit_tail_hash`, so the corpus degrades to pinning `verdict_codes` only — weaker, but still a conformance suite.

What degrades: auditability becomes unverifiable logging. Nothing detects after-the-fact edits.

**Destruction: ~15%.** `[ESTIMATE]`

Two qualifications matter more than the number:

1. `[OBSERVED]` **"It" is not one thing.** 34 files define their own chain or tree: `qradle/core/merkle.py`, `qradle/merkle.py`, `contracts/provenance.py`, `qledger/chain.py`, `quasim/reproducibility.py`, `quasim/audit/log.py`, `quasim/hcal/audit.py`, `events/log.py`, `qratum/platform/event_chain.py`, `qratum/exascale/network/merkle_nic.py`, `qratum/exascale/compiler/verifier.py`, `temporal_compression/verification.py`, `topological_observer/observer.py`, `qratum_ai_platform/core/audit.py`, `qratum_framework/trace.py`, `qratum_platform/core.py`, `os/.../receipt_merkle.rs`, `q-substrate/src/audit.rs`, `Aethernet/`, and more. Removing "cryptographic provenance" means removing twenty non-interoperable implementations. There is no single primitive to remove — which is itself the finding.

2. `[OBSERVED]` **Much of it is not cryptographic.** The `os/` receipt chain uses **FNV-1a-64**, a non-cryptographic 64-bit hash. QRADLE's chain has no signatures, so whole-chain forgery is undetectable `[TESTED]`. The `crypto/pqc/` modules are explicit placeholders. So a share of this 15% is already notional.

---

## 8. IF DETERMINISTIC EXECUTION WERE REMOVED — what remains?

The answer differs completely by layer, which is itself instructive.

**At the `qradle/` layer: nothing changes.** `[TESTED]` Determinism is not enforced there. `FatalInvariants.enforce_determinism()` exists and is never called by the engine. I ran a deliberately non-deterministic executor twice in one engine: both runs returned `success=True` with different outputs and no error. Removing a guarantee that was never enforced removes nothing.

**At the `os/` layer: it is fatal to the asset.** `[DERIVED]` Determinism is what makes the 270-scenario corpus possible — without replay stability, `lock_state_hash` and `audit_tail_hash` are meaningless and the corpus cannot pin anything. The byte-stable determinism receipt collapses. `lock_state_hash_is_pure_function_of_input` fails by definition. **The arbiter stops being conformance-testable, which is where its value lives.**

**Destruction: ~35%.** `[ESTIMATE]`

The structure here is worth stating explicitly rather than hiding, because it is a legitimate objection to the determination (and §26 of the main report records it as such): **determinism's value in this repository is largely instrumental to arbitration.** Its 35% is not independent of the arbiter's 45–50%; it is the share of the arbiter's value that depends on replay stability. Read the two together, not additively.

---

## 9. IF THE EVIDENCE-DISCIPLINE APPARATUS WERE REMOVED — what remains?

`[OBSERVED]` All the code builds and all the tests pass. `proof_gate::CLAIMS`, `performance_claims_policy.md`, `claim_language_enforcement`, `claim_registry_validation_tests`, the traceability-matrix orphan gate, and `formal/model_limits.md` are additive.

**What is lost is not functionality — it is evaluability.**

**Destruction: ~20%, asymmetrically.** `[ESTIMATE]` Without it, an outsider assessing a 164 MB repository in which 53-66% of commits are bot- or CI-authored at ~2,000 lines per commit `[OBSERVED]` has no mechanism for distinguishing a substantiated claim from a generated one. The apparatus is what makes `[OBSERVED]` claims in this repository checkable at all.

Two honest qualifications:

- `[TESTED]` It currently certifies a tautology: the registered `ahtc-k.scheduler.10x` claim resolves to a benchmark whose ratio is `N/33`, set by a 33-element key space in the workload generator. The published figures `1515` and `3030` are registered verbatim as claim `doc_phrases`.
- `[OBSERVED]` Its enforcement is narrower than its policy asserts: `claim_language_enforcement` scans only the policy file, while the policy claims coverage of "every artifact under this repository"; the claim registry scans only `README.md` and `spec/ahtc_k.md`.

So the 20% is the value of the *mechanism*, not of the current *coverage*. Repairing the coverage is cheap (days) and would raise it.

---

## 10. CONCLUSION

`[DERIVED]` Ranked by strategic value destroyed:

```
  Arbiter + corpus         ████████████████████████  45–50%
  Deterministic execution  ██████████████████        35%   (largely instrumental to the above)
  Evidence discipline      ██████████                20%   (asymmetric: evaluability, not function)
  Cryptographic provenance ███████                   15%   (fragmented across 20+ implementations)
  QRADLE                   ██                         ~5%
  14 verticals             █                          ~3%
  Q-Substrate              █                          ~3%
  QRATUM-ASI               ▌                          ~2%   (plausibly negative)
```

Three findings follow, and each is independent of the percentages:

1. **The removal that destroys the most strategic value is the arbiter**, and after it every remaining component has a free, better substitute. This is the destruction test's contribution to the determination.

2. **QRADLE's removal costs ~5% while its removal from the *narrative* costs everything.** Every README positions it as foundational; the import graph shows the 14 verticals do not touch it, `os/` does not touch it, and QuASIM does not touch it. **The documented architecture is not the implemented one.**

3. **Code volume is anti-correlated with destruction cost.** QRATUM-ASI (50,325 LOC) destroys ~2%; the verticals (4,403 LOC) destroy ~3%; QuASIM (67,376 LOC) is not even in the top eight. The arbiter is 182 lines plus 270 test files. **Roughly 90% of this repository's code carries roughly 20–35% of its value, and removing a large share of it would improve the credibility of the rest.**
