# QRATUM — Competitive Substitution Analysis

**Question for every candidate asset:** *if QRATUM did not exist, what would a sophisticated organization use instead?*

An asset only has value where the realistic substitute is materially worse. Where a free, mature, better-documented substitute exists, the QRATUM component is **cost**, not an asset — regardless of how much code it contains.

Evidence labels: `[OBSERVED]` verified in source · `[TESTED]` demonstrated by execution · `[ESTIMATE]` analytical estimate · `[EXTERNAL]` established by an outside source.

---

## 1. SUMMARY TABLE

| QRATUM asset | Realistic substitute | Substitute cost | Substitute better? | Survives? |
|---|---|---|---|---|
| **Arbiter** (authority arbitration + corpus) | OPA/Cedar; a hand-rolled priority lock; vendor SIS/interlock | $0 – $500k | **No — none occupies the position** | **YES** |
| **HCAL** (policy-gated actuation) | NVML + Ansible + homemade YAML; NVIDIA DCGM | $0 + 2–6 eng-months | Partially | **PARTIALLY** |
| **Evidence discipline** (claim registry) | DOORS/Jama traceability; `criterion`/`codspeed`; model cards | $0 – $200k/yr | **No equivalent exists** | **PARTIALLY** |
| **Contracts / compliance artifacts** | Veeva Vault, MasterControl, LDRA, Jama | $50k – $500k/yr | Yes, considerably | NO |
| **TLA+ suite** | Write your own specs; Alloy; Dafny; Kani | $0 + formal-methods hire | Equal — and QRATUM's are unchecked `[OBSERVED]` | NO |
| **QRADLE** | Postgres append-only + triggers; AWS QLDB; Sigstore/Rekor; Temporal | $0 – low | **Yes, dramatically** | **NO** |
| **AHTC-K** | Request coalescing; memoization; batch dedup; CSE in any compiler | $0 | Yes — decades mature | NO |
| **Kernel** (`no_std` UEFI) | seL4; Zephyr; QNX; VxWorks; FreeRTOS; Linux PREEMPT_RT | $0 – $250k | **Yes, overwhelmingly** | NO |
| **QuASIM** | Qiskit + Aer; Cirq; cuQuantum; ITensor; ANSYS; COMSOL | $0 – $100k/seat | **Yes, overwhelmingly** | NO |
| **Q-Substrate** | `llama.cpp` + Qiskit Aer + `wasmtime`; `candle`; ONNX Runtime | $0 | Yes, except binary size | NO |
| **AION** | MLIR/LLVM; Apache Arrow; Z3; Cranelift | $0 | **Yes, overwhelmingly** | NO |
| **14 verticals** | OR-Tools; Gurobi; Prophet; scikit-learn; domain ISVs | $0 – $50k/seat | **Yes, by 10³–10⁶×** | NO |
| **QRATUM-ASI** | Nothing — it requires unachieved breakthroughs `[OBSERVED]` | n/a | n/a | NO |
| **PQC modules** | `liboqs`; `pqcrypto`; AWS-LC; BoringSSL | $0 | **Yes — QRATUM's are placeholders** `[OBSERVED]` | NO |

**One asset survives outright. Two survive partially. Eleven do not survive.**

---

## 2. THE ASSET THAT SURVIVES — ARBITER

### What a sophisticated organization would do instead

**Option A — a policy engine (OPA/Rego, AWS Cedar, XACML, Casbin).** `[EXTERNAL]`

| Factor | Policy engine | QRATUM arbiter |
|---|---|---|
| Cost | $0, mature, large ecosystem | n/a |
| Engineering effort | 2–4 weeks to integrate | 1 week to write from scratch `[ESTIMATE]` |
| **Holder state** | **None — decisions are stateless per request** | Persistent `LockState` with holder, inflight, override counters `[OBSERVED]` |
| **Human/machine asymmetry** | None — it is an attribute to write a rule about | First-class: dominance window + thrash limiter `[OBSERVED]` |
| **Replay determinism** | Not guaranteed; Rego permits non-deterministic data sources | Pure total function, byte-stable hashes `[TESTED]` |
| **Conformance corpus** | None for this problem | 270 committed scenarios `[OBSERVED]` |
| Bypassability | Userspace library; an application can simply not call it | Can be kernel-resident `[OBSERVED]` |
| Auditability | Decision logs, free-form | 10 stable `u8` verdict codes designed for audit records `[OBSERVED]` |
| Regulatory suitability | No safety qualification | None yet either — but the architecture admits it |

**Verdict:** a policy engine is the wrong shape. You can *encode* arbitration rules in Rego, but you must then build the holder state, the counters, the determinism guarantee, and the corpus yourself — which is the entire asset. Policy engines solve authorization; they do not solve *control contention*.

**Option B — a hand-rolled priority lock.** This is what almost every organization actually does. `[ESTIMATE]`

Cost: 1–2 weeks. Result: undocumented, untested, per-system, with failure modes discovered during incidents and no artifact that answers an auditor. The substitute is *cheap and bad*, and its badness is diffuse (incident time, duplicated effort, regulatory exposure) rather than visible on a budget line. **This is the real competitor, and it is the hardest to displace** — not because it is good, but because nobody has a line item for replacing it.

**Option C — a vendor safety instrumented system / interlock.** `[EXTERNAL]`

| Factor | Vendor SIS | QRATUM arbiter |
|---|---|---|
| Cost | $50k – $500k+ per installation | n/a |
| **Semantics** | **Correct — latching priority, timeout override, override counters are standard** | The same pattern |
| Certification | **IEC 61508 SIL-rated, already qualified** | None `[OBSERVED]` |
| Portability | Hardware-bound, vendor-specific | Portable library |
| Replay corpus | None published | 270 scenarios |
| Applicability to software agents | **None** — designed for physical process loops | Direct |

**Verdict:** this is the most serious substitute and the honest source of prior art. It beats QRATUM on certification — which is the only moat available (§19 of the main report) — and loses on portability and on applicability to software agents. **An organization governing an LLM agent's authority over a production system cannot buy a SIS.** That is the gap.

### Economic comparison

| Dimension | Status quo (bespoke) | Policy engine | Vendor SIS | Arbiter spec + library |
|---|---|---|---|---|
| Cost of substitute | 1–2 eng-weeks per system | $0 + 2–4 weeks | $50k–$500k/install | Spec free; library TBD |
| Engineering effort | Low per system, **repeated N times** | Moderate, plus build the missing half | Low (procurement) | Low |
| Integration effort | Trivial | Moderate | High | Low |
| Performance | Fine | Fine | Fine | Pure function, no allocation |
| **Auditability** | **Poor — logs not designed for the question** | Moderate | Good | Good by construction |
| Reliability | Unknown | Good | **Excellent** | Unproven |
| **Regulatory suitability** | **Poor** | Poor | **Excellent** | **None yet — the gap to close** |
| Switching cost once adopted | n/a | Low | Very high | Low → high once the receipt format enters an audit pipeline |

### Conclusion

**The arbiter survives the substitution test** because no substitute occupies its position: portable, replayable, specified, conformance-testable, kernel-placeable, with an audit taxonomy, and applicable to software agents rather than only physical loops.

Its weakness is equally clear: **the substitute that wins on the dimension buyers care most about — qualified certification — is the vendor SIS**, and closing that gap costs $2M–$20M and 2–4 years `[ESTIMATE]` `[EXTERNAL]`.

---

## 3. THE TWO PARTIAL SURVIVORS

### 3.1 HCAL

**Substitute:** `nvidia-smi` / NVML or ROCm-SMI, driven by Ansible or a homemade Python wrapper, with policy expressed as YAML somebody wrote, plus NVIDIA DCGM for telemetry. `[EXTERNAL]`

| Factor | Substitute | HCAL |
|---|---|---|
| Cost | $0 | n/a |
| Effort to match | 2–6 eng-months to reach parity `[ESTIMATE]` | exists `[OBSERVED]` |
| Dry-run by default | Rarely — scripts usually act | **Yes, by design** `[OBSERVED]` |
| Device allowlist | Hand-maintained | Enforced `[OBSERVED]` |
| Power/clock envelopes | Sometimes | Enforced `[OBSERVED]` |
| Rate limiting | Almost never | Enforced `[OBSERVED]` |
| Approval gates | Ticket systems, out of band | In the actuation path `[OBSERVED]` |
| Tamper-evident audit | No | Yes `[OBSERVED]` |
| Closed-loop calibration | Bespoke | Built in `[OBSERVED]` |
| Multi-vendor | Per-vendor scripts | NVML + ROCm `[OBSERVED]` |

**Verdict: PARTIALLY survives.** Every element is individually commodity, and that is the point: nobody packages them, so every GPU fleet operator rebuilds a worse version. The substitute is free and strictly worse in safety posture. This is a legitimate product — and a weakly defensible one, because the substitute cost is 2–6 engineer-months, not years.

### 3.2 Evidence-discipline apparatus

**Substitutes:** requirements traceability (IBM DOORS, Jama, Polarion); benchmark regression CI (`criterion`, `codspeed`, `bencher`); reproducibility tooling (`snakemake`, MLflow, Weights & Biases); documentation conventions (model cards, datasheets for datasets); prose linters (`vale`, `alex`). `[EXTERNAL]`

| Capability | Nearest substitute | Gap |
|---|---|---|
| Trace *requirements* → tests | DOORS, Jama — mature, expensive | None — substitute wins |
| Detect performance regression | `criterion`, `codspeed` — mature | None — substitute wins |
| **Trace a published documentation or marketing claim → the test that substantiates it, and fail the build on an unregistered claim** | **No equivalent known** `[EXTERNAL]` | **Real gap** |
| **Ban epistemically unsupportable marketing terms in CI** | `vale` enforces style, not epistemics | **Real gap** |
| **Declare "properties NOT proved" alongside a formal model, with a runtime witness per property** | Good practice, rarely written down | **Real gap in practice, not in concept** |

**Verdict: PARTIALLY survives.** The gap is real and I am not aware of a commercial product that fills it. But:

- `[TESTED]` It currently certifies a tautology — the `N/33` benchmark of §6.3 in the main report.
- `[OBSERVED]` Its scope is narrower than its own policy claims: `claim_language_enforcement` scans only the policy file; the claim registry scans only `README.md` and `spec/ahtc_k.md`.
- `[ESTIMATE]` Rebuild cost is 1–2 weeks. The difficulty is organizational willingness to let CI block a press release, not engineering.

It survives as a **credibility asset and a differentiator in regulated sales**, not as a product.

---

## 4. THE ELEVEN THAT DO NOT SURVIVE — THE THREE MOST IMPORTANT

### 4.1 QRADLE — the substitute is dramatically better

This matters most, because QRADLE is the component the repository's narrative is built on.

| Capability | **QRADLE** `[TESTED]` | Postgres append-only | AWS QLDB | Sigstore/Rekor | Temporal |
|---|---|---|---|---|---|
| Append-only log | Yes | Yes | Yes | Yes | Yes |
| Tamper-evident (in place) | **Yes** | Yes (triggers/WAL) | Yes | Yes | Yes |
| **Tamper-evident (whole-chain rewrite)** | **NO** | Yes (WAL + backups) | **Yes (Merkle journal)** | **Yes (witnessed, signed)** | Yes |
| **O(log n) inclusion proof** | **NO — O(n), and the verifier accepts a bogus proof** | n/a | **Yes** | **Yes** | n/a |
| **Replay-stable audit root** | **NO — hashes `datetime.now()`** | n/a | Yes | Yes | **Yes** |
| **Deterministic replay enforced** | **NO — invariant never called** | n/a | n/a | n/a | **Yes — its core feature** |
| **Identity of authorizer** | **NO — a caller-supplied bool** | Yes (DB roles) | Yes (IAM) | **Yes (OIDC, signed)** | Yes |
| Prevents unauthorized execution | **NO — EXISTENTIAL clears on one bool** | Yes (grants) | Yes (IAM) | n/a | Yes |
| **Independent third-party verification** | **NO — no inputs, no code identity, no signature** | Partial | Yes | **Yes — the entire purpose** | Partial |
| Rollback restores state | **PARTIAL — state untouched** | Yes (PITR) | Yes | n/a | **Yes** |
| Cost | "free" + maintenance | $0 | ~$0.10/M reads | $0 | $0 OSS |
| Maturity | 3,496 LOC, 1 failing test | decades | production | production, CNCF | production |

**Verdict:** on nine of eleven capabilities QRADLE is strictly worse than a free substitute. A sophisticated organization needing auditable execution uses **Temporal** for deterministic replay and **Rekor** for verifiable transparency, both free and both better. **QRADLE has no substitution case.**

### 4.2 The 14 verticals — substitutes are better by orders of magnitude

`[OBSERVED]` FLUXA's route optimizer is greedy nearest-neighbour, 399 lines, with the in-source note `"Simplified heuristic - use OR-Tools for production"` — the code tells you to use the substitute.

| Vertical capability | Substitute | Gap |
|---|---|---|
| FLUXA route optimization | Google OR-Tools (free), Gurobi | Free substitute solves VRPs 10³–10⁶× larger, optimally `[EXTERNAL]` |
| FLUXA demand forecasting | Prophet, statsforecast, AutoGluon | Free, validated on real series |
| VITRA (life sciences) | RDKit, OpenMM, Schrödinger | Free-to-$100k, validated |
| JURIS (legal) | Any modern LLM + retrieval | Better and cheaper |
| ECORA / TERAGON / SENTRA / etc. | Established domain ISVs | Validated and supported |

**Verdict:** no substitution case for any vertical. At ~280 lines each they are demonstrations.

### 4.3 The kernel — substitutes are overwhelmingly better

| Factor | **QRATUM OS** | seL4 | Zephyr | QNX | Linux PREEMPT_RT |
|---|---|---|---|---|---|
| LOC | 8,163 `[OBSERVED]` | ~10k verified | large | large | enormous |
| **Formally verified** | **No — 14 unchecked specs** `[OBSERVED]` | **Yes — machine-checked functional correctness** | No | No | No |
| Safety certification | **None** | **Available** | **Available** | **IEC 61508 / ISO 26262 / DO-178C** | Partial |
| Architectures | x86_64 UEFI only | many | many | many | many |
| Drivers / ecosystem | minimal | moderate | large | large | vast |
| **Builds on Linux CI** | **No** `[TESTED]` | Yes | Yes | Yes | Yes |
| Cost | n/a | $0 | $0 | commercial | $0 |

**Verdict:** seL4 is machine-checked, free, and certifiable; QNX is already certified. **There is no substitution case for building a kernel from scratch.** The kernel's only defensible role is as a *host* for the arbiter, and even that is better served by porting the arbiter to seL4 or Zephyr — which, per §25.3 of the main report, is the architectural consequence.

---

## 5. WHAT THE SUBSTITUTION TEST ESTABLISHES

1. **Exactly one asset survives outright: the arbiter.** It wins because it is the only component for which no adequate substitute occupies the same position — not because it is the largest, the most novel, or the most tested.
2. **The survival pattern is inverted against code volume.** `quasim/` (67k LOC), `qratum_asi/` (50k), `qratum/` (41k), `aion/` (12.7k) all fail. The survivor is 182 lines plus 270 test files.
3. **The real competitor for the survivor is the status quo**, not a vendor. Bespoke priority logic is cheap, bad, and universal, and its badness does not appear on any budget line — which is why the sales motion must run through risk and compliance (§20 of the main report), not engineering.
4. **The substitute that beats the survivor does so on certification, not technology.** A vendor SIS is already IEC 61508-rated. That identifies the one investment that converts this asset from a feature into infrastructure, and prices it: $2M–$20M, 2–4 years `[ESTIMATE]`.
5. **QRADLE fails the substitution test more decisively than almost anything else in the repository**, losing on nine of eleven capabilities to free, mature alternatives. Since QRADLE is the stated foundation of the entire architecture, this is the finding with the largest strategic consequence.
