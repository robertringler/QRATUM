# The Physical Operator Computer (POC)

**A Monograph on Architecture, Mathematics, Verification, and Manufacturability**

*QRATUM Research Series — Document POC-MONO-1.0*

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Motivation and Positioning](#2-motivation-and-positioning)
3. [Mathematical Foundations](#3-mathematical-foundations)
4. [Operator Algebra as Instruction Set](#4-operator-algebra-as-instruction-set)
5. [Hardware Architecture](#5-hardware-architecture)
6. [Processor Design](#6-processor-design)
7. [Memory Design](#7-memory-design)
8. [Software Architecture](#8-software-architecture)
9. [Information Geometry](#9-information-geometry)
10. [Dynamical Systems View of Computation](#10-dynamical-systems-view-of-computation)
11. [Verification Framework](#11-verification-framework)
12. [Manufacturing Roadmap](#12-manufacturing-roadmap)
13. [Benchmark Programs](#13-benchmark-programs)
14. [Failure Analysis (FMEA)](#14-failure-analysis-fmea)
15. [Scientific Audit](#15-scientific-audit)
16. [References](#16-references)

Architecture figure: [`figures/poc_architecture.svg`](figures/poc_architecture.svg)

---

## 1. Abstract

The Physical Operator Computer (POC) is a computing architecture in which the
primitive unit of execution is not the Boolean gate acting on bits, but the
**linear operator acting on a physical state**. Where a von Neumann machine
fetches an instruction and mutates a word in memory, the POC configures a
physical medium so that its natural evolution *applies an operator* — a unitary
rotation, a contraction, a projection, a completely positive map — to a state
vector encoded in the medium itself. Programs are words in an operator algebra;
outputs are spectra, fixed points, and expectation values.

This monograph develops the POC from first principles: the C\*-algebraic
foundations; the operator instruction set architecture (O-ISA); processor and
memory organization realized on a programmable photonic interferometer mesh
with analog electronic feedback; a software stack that compiles QRATUM QIL
intents into hash-addressed **operator contracts**; the information-geometric
and dynamical-systems semantics that make analog evolution analyzable; a
verification framework built on tomography budgets and QRATUM's append-only
event log; a four-phase manufacturing roadmap; a benchmark suite; a failure
modes and effects analysis; and a scientific audit that explicitly separates
established results from extrapolation and speculation.

The honest thesis: for a specific class of problems — spectral decomposition,
steady states of open dynamics, natural-gradient optimization — a physical
medium *is* the matrix, and applying an operator costs one optical transit
rather than O(n²) multiply–accumulates. The equally honest antithesis, treated
in §15: analog precision does not scale for free, and the POC is a
domain-specific accelerator, not a general-purpose computer.

---

## 2. Motivation and Positioning

### 2.1 The operator gap

Most workloads that dominate modern compute — training and inference of neural
networks, PDE solvers, eigenproblems, Markov-chain steady states, control — are
**operator-application loops**: apply a linear (or locally linearized) map to a
state, repeat. Digital machines simulate operator application by decomposing it
into scalar multiply–accumulates. The energy cost is dominated not by the
mathematics but by the simulation overhead: data movement between memory and
ALU, at roughly 10–100 pJ per off-chip word moved versus ~1 pJ per MAC.

A physical medium with controllable linear response applies an n×n operator in
one transit, with energy proportional to maintaining the configuration, not to
n². This observation is old (analog computers, optical Fourier processors) and
recently revalidated by programmable nanophotonic processors and analog
crossbar accelerators. The POC's contribution is not the physics but the
**architecture**: treating the operator algebra itself as the ISA, with
composition, adjoint, and spectral access as first-class architectural
operations, under a deterministic, auditable control plane.

### 2.2 Positioning against neighboring architectures

| Architecture | Primitive | State | Determinism model | POC relation |
|---|---|---|---|---|
| von Neumann / GPU | Boolean gate / MAC | bits / floats | exact, replayable | host and verifier of POC |
| Quantum computer | unitary gate on qubits | entangled quantum state | probabilistic, fragile | POC uses *classical* fields; no entanglement claimed |
| Analog computer | integrator on voltages | continuous signals | drift-limited | POC inherits physics, adds algebraic ISA + verification |
| Neuromorphic | spiking neuron | membrane potentials | stochastic | overlapping substrate, different semantics |
| **POC** | **operator on a state vector** | **classical optical/analog field** | **statistically verified, contract-bound** | — |

The POC is explicitly **not** a quantum computer. Its state space is the
complex amplitude vector of a classical field across n modes. It obtains
operator-level parallelism from wave physics, not from entanglement, and
therefore offers no exponential state-space compression. What it offers is
O(n²) → O(n) energy/time scaling for dense operator application at moderate
precision, which is exactly the regime where digital machines burn most of
their power.

### 2.3 Role inside QRATUM

QRATUM's canonical flow — *Intent → Authority → Contract → Time → Execution →
Event → Audit* — was designed for heterogeneous, partially trusted execution
substrates. The POC is the limiting case of an untrusted substrate: an analog
device whose every result must be statistically certified. The POC therefore
plugs into QRATUM as a capability-resolved hardware class (`HARDWARE POC`),
receives **OperatorContracts** issued by Q-Core, and emits hash-chained
measurement events that make analog execution *auditable* — the property
analog computing historically lacked.

---

## 3. Mathematical Foundations

### 3.1 State space

The machine state is a vector ψ in an n-dimensional complex Hilbert space
ℋ ≅ ℂⁿ, physically realized as the complex amplitudes (magnitude and phase) of
a coherent optical field across n waveguide modes. Mixed/uncertain states are
represented by density operators ρ ∈ 𝒟(ℋ) = {ρ ⪰ 0, tr ρ = 1}, realized as
classical ensembles over repeated shots. The inner product ⟨φ|ψ⟩ is physically
accessible via interferometric homodyne detection.

### 3.2 The observable algebra

The set of executable operations is modeled as a C\*-algebra 𝒜 ⊆ B(ℋ) of
bounded operators, closed under:

- **addition** A + B (beam combination),
- **composition** AB (cascaded transit),
- **adjoint** A† (time-reversed/transposed mesh configuration),
- **norm limits** (calibration-converged sequences).

Self-adjoint elements A = A† are observables; their spectra are the
measurable outputs. The C\*-identity ‖A†A‖ = ‖A‖² is what licenses spectral
calculus on the machine: for any self-adjoint A and continuous f, f(A) is
again in 𝒜 and is *physically constructible* (§4.4).

### 3.3 Channels and noise

Real execution is noisy. Every physical instruction is modeled not as an exact
operator but as a **completely positive trace-non-increasing map** (a channel)
Φ: 𝒟(ℋ) → 𝒟(ℋ) with Kraus decomposition

```
Φ(ρ) = Σₖ Mₖ ρ Mₖ†,    Σₖ Mₖ†Mₖ ⪯ I
```

where M₀ ≈ √(1−ε) A is the intended operator and the residual Kraus terms
capture loss, crosstalk, and phase noise. The **instruction fidelity** is

```
F(Φ, A) = tr(Φ(ρ_A) ρ_A) averaged over a state 2-design,
```

and the verification framework (§11) exists to certify F ≥ F_min per
contract. This is the standard open-systems formalism; the POC imports it
wholesale for *classical* fields, where it remains valid because the maps are
linear on the mode space.

### 3.4 Generators and dynamics

Continuous-time execution is generated by a Lindblad-type master equation

```
dρ/dt = ℒ(ρ) = −i[H, ρ] + Σⱼ ( Lⱼ ρ Lⱼ† − ½{Lⱼ†Lⱼ, ρ} )
```

where H (Hermitian) is implemented by programmable phase shifts and
evanescent couplings, and the Lⱼ by engineered loss/gain. For classical
fields, the same equation governs the second-moment (covariance) dynamics. The
semigroup e^{tℒ} is the machine's *native* primitive for relaxation
computations: steady states, PageRank-like fixed points, thermal sampling
(§13, BM-3).

### 3.5 Precision model

The fundamental analog constraint: amplitudes are continuous but measured with
finite signal-to-noise ratio. With shot-noise-limited detection at photon
budget N_ph per mode per shot, single-shot amplitude precision is
δ ≈ 1/√N_ph, and S-shot averaging gives effective precision δ/√S. The POC
therefore exposes precision as a **first-class resource**: every
OperatorContract carries `precision: (bits, confidence)` and the runtime
converts it into shot and photon budgets. Achievable per-transit precision in
demonstrated photonic hardware is ~6–8 bits; the architecture treats anything
beyond ~10 bits as requiring digital cleanup iterations (§8.4).

---

## 4. Operator Algebra as Instruction Set

### 4.1 O-ISA design principle

The instruction set *is* a generating set of the algebra. An O-ISA program is
a word w = g_k g_{k−1} ⋯ g_1 over generators, plus measurement directives.
Compilation is factorization: given target A ∈ 𝒜, find a short word w with
‖w − A‖ ≤ ε. This replaces "instruction selection" with a well-studied
mathematical problem (matrix factorization / synthesis on U(n)).

### 4.2 Generator classes

| Class | Mnemonic | Mathematical object | Physical realization |
|---|---|---|---|
| Phase | `PHS(j, θ)` | diag(…, e^{iθ}, …) | thermo/electro-optic phase shifter on mode j |
| Mix | `MIX(j, k, θ, φ)` | 2×2 unitary on modes j,k | Mach–Zehnder interferometer (MZI) |
| Scale | `SCL(j, g)` | diag(…, g, …), 0 ≤ g ≤ g_max | variable optical attenuator / SOA gain |
| Project | `PRJ(S)` | orthogonal projector onto span S | mode filter + recirculation |
| Adjoint | `ADJ(w)` | w† | reverse-order transit with conjugate phases |
| Evolve | `EVO(H, L, t)` | e^{tℒ} | recirculating loop with engineered loss for time t |
| Measure | `MEAS(B, S)` | POVM in basis B, S shots | homodyne/heterodyne detector bank |

**Completeness.** By Reck/Clements decomposition, {`PHS`, `MIX`} generate all
of U(n) with exactly n(n−1)/2 MZIs; adding `SCL` extends to all of
GL-contractions via singular value decomposition A = UΣV† (two unitary words
and one diagonal scale word). Hence the O-ISA is universal for bounded
operators with ‖A‖ ≤ g_max. This is the load-bearing universality claim, and
it is mathematically exact; only its *precision* is physically limited.

### 4.3 Composition rules and cost model

- `w₂ ∘ w₁`: sequential transit, cost = sum of transit times (~ns each),
  fidelity multiplies: F(w₂w₁) ⪆ F(w₂)·F(w₁).
- `w₁ + w₂` (superposition): parallel paths combined on a balanced coupler;
  costs a 3 dB power penalty per addition — addition is *expensive* in the
  O-ISA, the opposite of digital machines. Compilers prefer multiplicative
  structure.
- `ADJ`: free at compile time (reversed word), but doubles calibration burden
  because the reverse path must be matched.

### 4.4 Spectral calculus as an architectural operation

For self-adjoint A, the POC computes f(A)ψ without diagonalizing digitally:

1. **Power filtering**: repeated transit A^k ψ amplifies dominant
   eigencomponents (power method in hardware; one transit per iteration).
2. **Resolvent access**: a recirculating loop with feedback gain implements
   (I − αA)⁻¹ ψ = Σ αᵏAᵏψ for α‖A‖ < 1, converging in the loop's photon
   lifetime — Neumann series as ringdown physics.
3. **Chebyshev synthesis**: f(A) ≈ Σ cₖ Tₖ(A) with Tₖ generated by the
   three-term recurrence using two loop registers.

Eigenvalue *readout* uses ringdown spectroscopy: the loop's temporal decay
beat spectrum is the spectrum of A, captured on a fast photodiode and FFT'd
digitally. This is the POC's answer to "decode": **the spectrum is a physical
signal, not a computed artifact.**

---

## 5. Hardware Architecture

### 5.1 System overview

The POC node (see `figures/poc_architecture.svg`) consists of five planes:

1. **Intent/compile plane** (digital, host CPU): QIL → operator word →
   mesh configuration. Stateless between jobs.
2. **Control plane** (digital, FPGA): holds the *active configuration*,
   sequences DAC updates, enforces OperatorContract bounds in hardware
   (clamps on gain, power, duration). Q-Core's authority terminates here.
3. **Execution plane** (photonic): the Operator Application Unit (OAU) — a
   programmable interferometer mesh — plus recirculation loops and gain.
4. **Measurement plane** (analog/mixed): detector banks, transimpedance
   amplifiers, ADCs, on-FPGA shot accumulation.
5. **Audit plane** (digital): every configuration write and every measurement
   batch is hashed and appended to the QRATUM event log; replay = re-issue of
   the identical configuration sequence plus statistical comparison.

### 5.2 Reference node specification (Phase M2 target, §12)

| Parameter | Value | Rationale |
|---|---|---|
| Mode count n | 64 | 2,016 MZIs (Clements mesh); within demonstrated SiN scale |
| Transit time | 2 ns | ~40 cm folded waveguide path |
| Reconfiguration | 10 µs (thermal) / 50 ns (carrier) | two-speed: weights vs. activations |
| Insertion loss | ≤ 0.05 dB/MZI → ≤ 3.2 dB/transit | sets loop depth ~ 40 transits |
| Per-transit precision | 7 bits @ 99% confidence | shot budget 10⁴ photons/mode |
| Loop registers | 4 recirculating, Q ≈ 10⁴ | resolvent and Chebyshev units |
| Detector bank | 64 × homodyne, 5 GS/s ADC | full-state readout in one shot batch |
| Power | ≈ 35 W node total | 25 W thermal tuning dominates |

### 5.3 Why photonics for the reference design

Photonic meshes are chosen for the reference design because: (i) the
mode-mixing physics is exactly the linear algebra required, with no
linearization error; (ii) transit latency is set by the speed of light, not RC
constants; (iii) the Reck/Clements result gives a *constructive, exact*
compilation target. Analog electronic crossbars (memristive or
switched-capacitor) are retained as a secondary substrate for high-density,
low-precision operator memory (§7.3), where their O(n²) cell count is
acceptable and their write endurance limits matter less.

---

## 6. Processor Design

### 6.1 The Operator Application Unit (OAU)

The OAU is the ALU analog: a Clements-topology mesh of n(n−1)/2 MZIs with a
column of n output phase shifters and n variable attenuators (the Σ stage).
One pass = one application of a configured operator A = UΣV†ψ:

```
ψ_in ──► [V† mesh] ──► [Σ attenuators] ──► [U mesh] ──► ψ_out
```

Two OAUs are paired per node so that one can be *reconfigured and calibrated*
(10 µs thermal settle) while the other executes — the analog of double
buffering. A word longer than one operator alternates between the pair.

### 6.2 The Spectral Unit (SU)

Four recirculating loop registers, each tapping the OAU output through a
programmable coupler (loop gain α) and re-injecting at the input. Configured
modes:

- **Power mode**: α = 1, k transits → Aᵏψ.
- **Resolvent mode**: loop closed with gain α < 1/‖A‖ → (I − αA)⁻¹ψ in
  steady state, reached in ~Q transit times.
- **Ringdown mode**: impulse in, photodiode beat spectrum out → spec(A).
- **Recurrence mode**: two loops cross-coupled for Chebyshev three-term
  recurrence (§4.4).

### 6.3 The Evolution Unit (EU)

Implements `EVO(H, L, t)`: a loop whose per-transit map is e^{δt·ℒ} via
first-order splitting (Hamiltonian phase pattern + engineered loss pattern),
with t/δt transits. Used for steady-state and relaxation benchmarks (BM-3).

### 6.4 Pipeline and hazards

The POC pipeline is: **CONFIGURE → SETTLE → INJECT → TRANSIT → DETECT →
ACCUMULATE**. The architectural hazards differ from digital pipelines:

- **Thermal hazard**: reconfiguring MZI (i,j) perturbs neighbors; the control
  plane enforces a thermal crosstalk matrix deadtime (structural hazard,
  resolved by the calibration model, not by stalling heuristics).
- **Precision hazard**: a downstream instruction requiring p bits cannot
  consume an upstream result accumulated to fewer than p bits; the contract
  checker rejects such words at compile time (no dynamic interlock needed —
  shot budgets are static per contract).
- **Loop coherence hazard**: spectral-unit programs must complete within the
  loop photon lifetime; the compiler inserts digital re-amplification
  (detect → DAC → re-inject) checkpoints when word depth exceeds ~40.

---

## 7. Memory Design

The POC has three memories with sharply different semantics:

### 7.1 State memory (volatile, transit-lived)

The optical field itself. Lifetime = loop photon lifetime (~µs at Q ≈ 10⁴).
There is no architectural "register file" of states: a state that must
persist beyond a loop lifetime is **measured and digitized** (a lossy,
precision-bounded store) and later re-synthesized by a calibrated modulator
bank (a lossy load). Store/load round-trip costs one shot budget of
precision. This is the deepest departure from von Neumann design and drives
the compiler's structure (§8.3): *minimize state spills, maximize operator
fusion.*

### 7.2 Operator memory (the configuration store)

Operators-as-data live in two tiers:

- **Active tier**: the DAC/heater settings of the two OAUs — exactly 2
  operators resident, swap cost 10 µs.
- **Staged tier**: digital configuration vectors (≈ 2·n² × 16 bit ≈ 16 KB per
  operator at n = 64) in FPGA BRAM, hundreds resident, hash-addressed by
  their OperatorContract ID so that *the same hash always reproduces the same
  physical configuration modulo calibration state* — the property that makes
  the audit plane meaningful.

### 7.3 Associative operator memory (analog crossbar tier, Phase M3)

A memristive crossbar bank stores low-precision (4-bit) operator sketches
with in-memory matrix-vector capability, used for **operator nearest-neighbor
lookup**: given a target A, find the stored configuration minimizing
‖A − Âᵢ‖_F as a warm start for calibration. This converts the crossbar's
weakness (precision) into a role where it is harmless (search), and cuts
cold-configuration calibration time by an order of magnitude in simulation.

---

## 8. Software Architecture

### 8.1 Stack overview

```
QIL intent  (HARDWARE POC, precision/deadline constraints)
   │  Q-Core: authority check, capability resolution
   ▼
OperatorContract        (hash-addressed, immutable, frozen)
   │  poc-compile: factorization + scheduling
   ▼
Operator Word IR  (generator sequence + shot/photon budgets)
   │  poc-lower: calibration-aware lowering
   ▼
Mesh Configuration Program  (DAC sequences, deadtimes, clamps)
   │  FPGA control plane
   ▼
Execution  ──►  Measurement events  ──►  QRATUM event log (hash-chained)
```

### 8.2 The OperatorContract

Extends QRATUM's contract system (`contracts/base.py` pattern: frozen,
SHA-256 hash-addressed, deterministically serialized):

```python
@dataclass(frozen=True)
class OperatorContract(BaseContract):
    operator_spec: OperatorSpec      # target A: dense, structured, or generator-defined
    norm_bound: float                # ‖A‖ ≤ g_max enforced in hardware clamps
    precision_bits: int              # required output precision
    confidence: float                # statistical confidence for precision claim
    shot_budget: int                 # derived; runtime may not exceed
    fidelity_floor: float            # F_min; below this, result is REJECTED not returned
    calibration_epoch: str           # hash of calibration state assumed valid
```

The `calibration_epoch` field is what makes analog execution contract-bound:
a contract is only executable while the device's calibration hash matches;
drift past tolerance invalidates the epoch and forces recalibration before
any further contract executes. Stale-calibration execution is structurally
impossible, not merely discouraged.

### 8.3 Compiler (poc-compile)

Three passes:

1. **Factorize**: target A → UΣV† (digital SVD) → Clements angle program.
   For structured A (sparse, banded, low-rank), specialized factorizations
   produce shorter words (low-rank: rank-r ≪ n uses r MIX columns only).
2. **Fuse and schedule**: minimize state spills (§7.1) by fusing adjacent
   operators into single mesh configurations when the composite still meets
   the precision model; schedule across the dual OAUs to hide
   reconfiguration latency.
3. **Budget**: propagate the precision requirement backward through the word
   using the fidelity-multiplication model (§4.3) to assign per-instruction
   shot budgets; reject the program if total photon budget exceeds the
   contract's energy/deadline bounds. **Compilation can fail on precision
   grounds; this is a feature.**

### 8.4 Hybrid iteration runtime

For precision beyond the analog ceiling, the runtime implements
**digital-cleanup iteration** (the analog of mixed-precision iterative
refinement): solve/apply at 7 bits on the POC, compute the residual digitally
in fp64 on the host, re-inject the residual as a new POC job. Each round
gains ~7 bits; 3 rounds reach fp32-class results with the POC doing the O(n²)
work and the host doing O(n) work. This loop is the *practical* execution
model for linear solves and is benchmarked as BM-2.

---

## 9. Information Geometry

### 9.1 The state manifold

Normalized states live on the complex projective space ℂPⁿ⁻¹; mixed states on
the manifold of density operators. The Fisher–Rao/Bures metric

```
ds² = ½ tr(dρ G),   where dρ = ½(ρG + Gρ)
```

gives the manifold its natural Riemannian structure. The POC's significance:
**homodyne measurement statistics directly sample the quantities from which
the Fisher information matrix is estimated** — the metric is empirically
accessible at the same cost as readout, rather than requiring extra
computation.

### 9.2 Natural gradient as native descent

For optimization intents (QIL `OBJECTIVE` clauses lowered to loss
functionals ℓ(ρ_θ)), the natural gradient step

```
θ_{k+1} = θ_k − η F(θ_k)⁻¹ ∇ℓ(θ_k)
```

requires applying an inverse metric — exactly a resolvent-mode Spectral Unit
operation (§6.2) with A = F(θ_k). The POC thus executes natural-gradient
descent at the per-iteration cost ordinary machines pay for vanilla gradient
descent. This is the architectural justification for the information-geometry
layer: it is not decoration, it is the optimizer's inner loop made physical.
(BM-4 measures this claim.)

### 9.3 Geodesic transport

Parallel transport of a direction vector along the manifold — required by
momentum-based optimizers and by interpolation between operating points — is
implemented as a short EVO word with the metric connection's generator. The
compiler emits it as a fused configuration; cost is one transit per
integration step.

---

## 10. Dynamical Systems View of Computation

### 10.1 Computation as flow

Every POC program is a discrete- or continuous-time dynamical system on the
state manifold. The architectural semantics are:

- **Output = attractor.** A program's result is a fixed point, limit cycle
  spectrum, or ergodic average — never a transient read at an arbitrary time.
- **Convergence is certified, not assumed.** A word is admissible only if the
  compiler can attach a contraction certificate.

### 10.2 Contraction analysis

For a discrete word with composite map T, the compiler verifies a contraction
factor: ‖T(ψ) − T(φ)‖ ≤ c‖ψ − φ‖ with c < 1 in a metric it selects
(weighted ℓ₂ via a diagonal SCL conjugation). For EVO programs it verifies
the Lindbladian's spectral gap λ_gap > 0 (digitally, on the target generator)
and bounds time-to-ε as t ≥ λ_gap⁻¹ ln(1/ε). The certificate (c or λ_gap,
plus the metric) is embedded in the OperatorContract and **rechecked by the
verifier against measured convergence rates** — disagreement between
certified and observed contraction is the single most sensitive drift
detector the machine has (§11.4).

### 10.3 Robustness semantics for analog error

Contraction is also the noise story: a contractive program forgets bounded
injected noise at rate c per step, so steady-state error is bounded by
δ_inj/(1−c). Non-contractive (norm-preserving, unitary-only) words have no
such forgiveness and accumulate phase error linearly in depth — which is why
the O-ISA cost model charges unitary-only deep words a superlinear shot
budget, and why the compiler prefers mildly dissipative formulations when the
mathematics allows. **Dissipation is the POC's error-correction budget.**

---

## 11. Verification Framework

Analog computing died historically for lack of verifiability. The POC treats
verification as co-equal with execution.

### 11.1 Three-level verification hierarchy

| Level | What is certified | Method | Cost | When |
|---|---|---|---|---|
| L1: Configuration | DAC values match contract | digital readback + hash compare | ~µs | every execution |
| L2: Instruction | implemented map ≈ contracted operator | sparse process tomography on a k-design probe set | ~ms | per calibration epoch |
| L3: Program | output statistics match certified attractor | convergence-rate test + cross-validation on digital twin (sampled) | ~s | per contract class, randomized audit |

### 11.2 Tomography budgets

Full process tomography is O(n⁴) measurements and unaffordable. The POC uses
**compressed verification**: the implemented map is known a priori to be near
the target (calibrated hardware), so verification estimates only the
deviation ‖Φ − Φ_target‖⋄ in a low-dimensional perturbation basis (per-MZI
angle errors + global loss + pairwise thermal crosstalk: O(n²) parameters,
estimated from O(n² log n) shots). The result is a confidence interval on the
diamond-norm error, recorded per calibration epoch.

### 11.3 Statistical replay and the event log

Deterministic bitwise replay is impossible for analog hardware; the POC
defines **statistical replay**: re-executing the hash-identical configuration
sequence under the same calibration epoch must produce measurement
distributions whose two-sample test statistic falls inside the contract's
confidence band. Every execution appends to the QRATUM event log:

```
{contract_hash, calibration_epoch, config_sequence_hash,
 shot_count, measurement_digest, verifier_verdict}
```

An auditor with log access can demand statistical replay of any past
contract; a failed replay distinguishes *drift since execution* (calibration
epoch hash differs) from *misreported results* (same epoch, incompatible
statistics). This is the property that makes POC results admissible in
QRATUM's audit chain.

### 11.4 Continuous drift sentinels

Between epochs, three cheap sentinels run interleaved with workload:
(i) a fixed reference word whose output spectrum is monitored (canary
program); (ii) the contraction-rate comparison of §10.2; (iii) detector dark
and gain tracking. Any sentinel excursion past 2σ invalidates the calibration
epoch hash, which by §8.2 structurally halts contract execution.

---

## 12. Manufacturing Roadmap

| Phase | Timeframe | Vehicle | Mode count | Goal | Exit criterion |
|---|---|---|---|---|---|
| M0 | 0–12 mo | COTS evaluation: commercial programmable photonic mesh + bench optics + FPGA | 8–16 | validate O-ISA, calibration loop, L1/L2 verification | BM-1 executed end-to-end under OperatorContract with certified 6-bit precision |
| M1 | 12–24 mo | SiN photonic IC, multi-project-wafer run; hybrid-integrated InP gain; CMOS driver board | 32 | dual-OAU node, Spectral Unit loops, statistical replay | BM-1..3 pass; L3 randomized audit operating; loss ≤ 0.08 dB/MZI |
| M2 | 24–42 mo | dedicated SiN run with custom PDK cells (low-loss crossings, fast carrier shifters); co-packaged detector ASIC | 64 | reference node of §5.2; hybrid iteration runtime | BM-2 reaches fp32-equivalent solves with ≥ 10× energy advantage over GPU baseline at n = 64 dense |
| M3 | 42–60 mo | multi-node: 4 nodes + crossbar associative tier; optical node-to-node state handoff *evaluated* (fallback: digitize-and-forward) | 4 × 64 | block-partitioned operators (n_eff = 256) | BM-5 (blocked operator) within 2× of single-node fidelity model |
| M4 | 60+ mo | volume design only if M2/M3 economics hold | 256+/node | product decision gate | independent replication of M2 results + cost model showing < $/op parity at target workloads |

**Manufacturing risk concentrations** (feeding §14): MZI loss uniformity
across the wafer (yield driver), thermal crosstalk at increased density,
detector ASIC co-packaging, and calibration time scaling (n² parameters —
mitigated by the crossbar warm-start tier in M3).

Each phase exit requires the **Scientific Audit gate** of §15.4: results
reproduced from the event log by a team that did not produce them.

---

## 13. Benchmark Programs

All benchmarks are defined as QIL intents, executed under OperatorContract,
and compared against a tuned GPU baseline (cuBLAS/cuSOLVER on the host) at
equal *certified* precision — the POC is never credited for uncertified bits.

| ID | Program | What it measures | Metric | Honest baseline note |
|---|---|---|---|---|
| BM-1 | Dense operator application, A ∈ ℂ⁶⁴ˣ⁶⁴, 10⁴ sequential applies | core OAU throughput + fidelity decay | applies/s, J/apply, certified bits | GPU wins below n ≈ 32; crossover claim is the test |
| BM-2 | Linear solve Ax = b via resolvent + digital cleanup (§8.4) | hybrid loop end-to-end | time and energy to fp32-residual | includes ALL digitize/re-inject overhead |
| BM-3 | Lindbladian steady state, gapped ℒ, n = 64 | Evolution Unit; the POC's best case | time-to-ε vs. digital expm/eigs | digital baseline allowed sparse structure |
| BM-4 | Natural-gradient descent, 200 iterations, Fisher solve on SU | §9.2 claim | wall-clock and J per iteration at matched final loss | vanilla-SGD digital baseline also reported |
| BM-5 | (M3) blocked 256×256 operator across 4 nodes | partitioning overhead | efficiency vs. single-node model | digitize-and-forward handoff counted fully |
| BM-6 | Adversarial: ill-conditioned solve, κ(A) = 10⁶ | where the POC should lose | rounds of cleanup iteration to converge or DNF | published even when DNF — this is the audit's teeth |

Reporting rule: every benchmark result is published with its contract hash,
calibration epoch, shot budgets, and the event-log digest sufficient for
statistical replay. A benchmark number without a replayable log entry does
not exist.

---

## 14. Failure Analysis (FMEA)

Severity (S), Occurrence (O), Detection difficulty (D) on 1–5 scales;
RPN = S·O·D.

| # | Failure mode | Cause | Effect | S | O | D | RPN | Mitigation |
|---|---|---|---|---|---|---|---|---|
| F1 | Thermal phase drift | ambient/self-heating | silent fidelity loss | 4 | 5 | 3 | 60 | sentinels §11.4; epoch invalidation; carrier shifters for fast ops |
| F2 | MZI angle miscalibration | fab variation + aging | systematic operator bias | 4 | 4 | 3 | 48 | L2 tomography per epoch; per-MZI error basis §11.2 |
| F3 | Loop gain instability (SU) | SOA gain ripple | resolvent divergence (α‖A‖ → 1) | 5 | 3 | 2 | 30 | hardware gain clamp from `norm_bound`; contraction monitor §10.2 |
| F4 | Detector saturation/nonlinearity | power excursion | corrupted statistics passing as data | 5 | 2 | 4 | 40 | power clamps in control plane; dark/gain tracking; range checks pre-accumulation |
| F5 | Calibration-epoch race | contract executed during recalibration | results under unknown configuration | 5 | 2 | 2 | 20 | epoch hash check is atomic with execution start (§8.2) |
| F6 | Crosstalk at reconfiguration | thermal neighbor coupling | transient operator corruption | 3 | 4 | 3 | 36 | crosstalk-matrix deadtimes §6.4; dual-OAU buffering |
| F7 | Shot-budget starvation | compiler underestimate of fidelity decay | precision claim unmet | 4 | 3 | 2 | 24 | verifier rejects (REJECT ≠ wrong answer); budget model updated from L2 data |
| F8 | Digital twin divergence | model error in L3 verifier | false audit failures (availability) or false passes (integrity) | 4 | 3 | 4 | 48 | twin parameters re-fitted from each epoch's L2 estimates; randomized cross-checks |
| F9 | Re-injection synthesis error | DAC/modulator miscalibration in state load | hybrid-iteration loop stalls | 3 | 3 | 3 | 27 | loopback self-test on each cleanup round |
| F10 | Event-log gap | control-plane crash mid-contract | unauditable execution | 5 | 1 | 1 | 5 | write-ahead logging; contract without complete log auto-voided |

Design posture: the top RPN items (F1, F2, F8) are all *drift/model* failures,
not hard faults — consistent with the architecture's bet that the dominant
risk in analog computing is silent degradation, addressed by making
calibration state a first-class, hash-addressed object.

---

## 15. Scientific Audit

This section is the document auditing itself. Claims are sorted into three
bins with the key risks named.

### 15.1 Established (load-bearing, externally validated)

- Reck/Clements universal decomposition of U(n) into MZI meshes — exact
  mathematics, demonstrated in hardware at n up to ~dozens of modes.
- SVD-based extension to contractions; Neumann-series resolvent physics in
  recirculating loops; ringdown spectroscopy.
- CPTP/Lindblad formalism for noisy linear maps; contraction/Lyapunov
  convergence analysis; natural gradient and Fisher–Rao geometry.
- Mixed-precision iterative refinement (the §8.4 loop is its direct analog,
  well-understood numerically).
- QRATUM's contract/event-log machinery (exists in this repository).

### 15.2 Extrapolated (plausible, quantitatively unproven at spec)

- **0.05 dB/MZI loss at n = 64 with full programmability** — best published
  results approach this on test structures; achieving it across 2,016 MZIs
  with yield is an extrapolation. If realized loss is 0.1 dB/MZI, loop depth
  halves and BM-3's advantage shrinks substantially.
- **7 certified bits per transit at the stated photon budgets** — consistent
  with shot-noise math, but certified (not best-case) precision in
  demonstrated systems is typically 4–6 bits. The roadmap's M0 exit
  criterion deliberately tests this first.
- **10× energy advantage at n = 64 (M2 exit)** — depends on the thermal
  tuning power (25 W floor) being amortized across high utilization; at low
  duty cycle the GPU wins on energy. The claim is conditional on workload
  density and is falsifiable by BM-1/BM-2.
- **Calibration scaling**: O(n²) parameters per epoch is manageable at
  n = 64; the warm-start mitigation for larger n is simulated, not
  demonstrated.

### 15.3 Speculative (clearly flagged)

- **Optical node-to-node state handoff (M3)** — coherent state transfer
  between separately calibrated nodes is an open problem; the roadmap
  carries digitize-and-forward as the assumed fallback, and BM-5 is scored
  against that fallback.
- **Associative crossbar operator search at useful scale** — simulation-only.
- **Any general-purpose framing.** The POC accelerates dense/structured
  operator pipelines at moderate precision. Branchy, integer-exact,
  high-precision, or memory-bound workloads are out of scope, permanently,
  by physics. No roadmap phase changes this.

### 15.4 Audit protocol (binding on the roadmap)

1. **Pre-registration**: each phase's exit benchmarks, baselines, and
   analysis code are hash-committed to the event log *before* hardware
   bring-up of that phase.
2. **Independent replication**: phase exit requires reproduction of headline
   numbers from the event log by personnel who did not produce them, using
   statistical replay (§11.3).
3. **Negative-result publication**: BM-6 (designed-to-lose) results and any
   DNF are published with the same prominence as wins.
4. **Kill criteria**: the program terminates at any phase gate where the
   certified-precision or energy-advantage extrapolations of §15.2 are
   falsified with no identified engineering recovery — explicitly: if M1
   cannot certify ≥ 6 bits per transit, or if M2's BM-2 advantage is < 2×,
   the correct scientific outcome is to stop and say so.

The architecture's strongest claim is deliberately modest: **analog operator
execution can be made auditable**. If only that survives contact with
hardware, the contract/verification framework of §8 and §11 remains a
contribution applicable to any analog accelerator.

---

## 16. References

1. M. Reck, A. Zeilinger, H. J. Bernstein, P. Bertani, "Experimental
   realization of any discrete unitary operator," *Phys. Rev. Lett.* 73, 58
   (1994).
2. W. R. Clements, P. C. Humphreys, B. J. Metcalf, W. S. Kolthammer,
   I. A. Walmsley, "Optimal design for universal multiport interferometers,"
   *Optica* 3, 1460 (2016).
3. Y. Shen *et al.*, "Deep learning with coherent nanophotonic circuits,"
   *Nature Photonics* 11, 441 (2017).
4. G. Lindblad, "On the generators of quantum dynamical semigroups,"
   *Commun. Math. Phys.* 48, 119 (1976).
5. S. Amari, "Natural gradient works efficiently in learning," *Neural
   Computation* 10, 251 (1998).
6. W. Lohmiller, J.-J. E. Slotine, "On contraction analysis for non-linear
   systems," *Automatica* 34, 683 (1998).
7. E. Carson, N. J. Higham, "Accelerating the solution of linear systems by
   iterative refinement in three precisions," *SIAM J. Sci. Comput.* 40,
   A817 (2018).
8. A. Sebastian, M. Le Gallo, R. Khaddam-Aljameh, E. Eleftheriou, "Memory
   devices and applications for in-memory computing," *Nature
   Nanotechnology* 15, 529 (2020).
9. QRATUM Canonical Architecture v1.0 — `QRATUM_ARCHITECTURE.md` (this
   repository).
10. N. C. Harris *et al.*, "Linear programmable nanophotonic processors,"
    *Optica* 5, 1623 (2018).

---

*Document POC-MONO-1.0 · QRATUM Research Series*
