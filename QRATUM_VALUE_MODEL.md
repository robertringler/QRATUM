# QRATUM — Value Model

Quantitative backing for `QRATUM_VALUE_ASSET_ANALYSIS.md`. Every figure here is reproduced by:

```bash
python tools/value_analysis/qratum_value_scorecard.py
python tools/value_analysis/qratum_value_scorecard.py --csv
```

**All scores and percentages in this document are `[ESTIMATE]` — analytical estimates, not measurements.** They are anchored to the `[OBSERVED]` and `[TESTED]` findings recorded in the main report, and the methodology is stated explicitly so a reader can disagree with a weight without re-deriving a score.

---

## 1. SCORING MODEL

### 1.1 Dimensions

Fourteen candidate assets are scored 0–10 on ten dimensions.

| Dimension | Definition |
|---|---|
`technical_uniqueness` | How far the mechanism departs from established practice, after prior art is accounted for
`implementation_maturity` | How much is built, runs, and is tested — not how much is written
`cross_domain_leverage` | How many distinct problem domains the asset applies to **outside** this repository
`commercial_value` | Willingness of an identifiable buyer to pay, near and medium term
`defensibility` | Resistance to competent reimplementation (see §4)
`switching_cost` | Cost to an adopter of leaving once adopted
`strategic_importance` | Degree to which other things depend on it, or would come to
`capital_efficiency` | How far the asset advances on $0–$1,000
`competitive_differentiation` | How distinguishable it is from the realistic substitute
`evidence_strength` | **Strength of evidence that the asset does what it claims**

### 1.2 The `evidence_strength` convention — this one matters

`evidence_strength` measures support for the asset's **claims**, not the volume of testing applied to it. An asset tested exhaustively whose claims **failed** scores **low**.

This is why QRADLE scores **2** on evidence despite being the single most heavily probed component in this analysis: eleven capability questions were executed against it and nine returned NO or PARTIAL (`tools/value_analysis/qradle_capability_probe.py`). Scoring it high for "being tested" would invert the meaning of the dimension.

The same convention puts AHTC-K at **2** — its headline claim was tested and found vacuous — and the TLA+ suite at **2**, since 14 specifications exist and zero `.cfg` files exist, so nothing has been machine-checked.

---

## 2. SCORECARD

| Asset | Uniq | Matur | Lever | Comm | Defens | Switch | Strat | Capital | Differ | Evid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ARBITER + 270-corpus | 7 | 7 | 8 | 6 | 3 | 2 | 8 | 9 | 7 | 8 |
| EVIDENCE-DISCIPLINE | 8 | 6 | 9 | 5 | 2 | 2 | 7 | 8 | 7 | 7 |
| HCAL | 5 | 7 | 5 | 7 | 3 | 4 | 5 | 8 | 5 | 6 |
| CONTRACTS / compliance | 3 | 6 | 6 | 5 | 2 | 2 | 5 | 7 | 3 | 5 |
| TLA+ suite | 6 | 2 | 5 | 4 | 3 | 2 | 6 | 7 | 5 | 2 |
| AHTC-K | 5 | 6 | 6 | 3 | 2 | 1 | 4 | 5 | 3 | 2 |
| KERNEL (`no_std` UEFI) | 4 | 5 | 3 | 2 | 4 | 1 | 5 | 2 | 4 | 5 |
| Q-SUBSTRATE | 4 | 5 | 4 | 3 | 2 | 1 | 3 | 6 | 3 | 4 |
| QUASIM | 3 | 5 | 6 | 3 | 2 | 1 | 4 | 4 | 2 | 4 |
| QRADLE | 2 | 4 | 3 | 2 | 1 | 1 | 3 | 6 | 2 | 2 |
| AION | 4 | 2 | 4 | 2 | 2 | 1 | 3 | 3 | 3 | 2 |
| VERTICALS (14) | 1 | 2 | 2 | 2 | 1 | 1 | 2 | 5 | 1 | 2 |
| QRATUM-ASI | 3 | 1 | 5 | 1 | 1 | 1 | 3 | 2 | 2 | 1 |
| PQC | 1 | 1 | 3 | 1 | 1 | 1 | 2 | 3 | 1 | 1 |

### 2.1 Anchors for the most consequential scores

| Score | Anchor |
|---|---|
ARBITER maturity **7** | `[TESTED]` 163/163 tests pass; 270 committed scenario files. Not higher because CI cannot build `[TESTED]` and kernel equivalence is a self-synthesized round-trip `[TESTED]`
ARBITER defensibility **3** | `[ESTIMATE]` 182 LOC; ~1 week to rebuild the core. Not 1, because the corpus + spec + taxonomy take 2–4 months
ARBITER capital efficiency **9** | `[DERIVED]` Publishing a specification and a conformance corpus costs $0 and is the whole wedge
EVIDENCE uniqueness **8** | `[EXTERNAL]` No known commercial product traces documentation claims to substantiating tests with build failure on an unregistered claim
EVIDENCE maturity **6** | `[OBSERVED]` Works, but scans 2 files while its own policy claims repo-wide scope; `[TESTED]` certifies a tautology
HCAL commercial **7** | `[DERIVED]` Highest in the repository: named buyer, existing budget line, 1–3 month cycle, no packaged competitor
TLA+ maturity **2** | `[OBSERVED]` 14 `.tla` files, **0** `.cfg` files repo-wide; documented invocation requires `-config <name>.cfg`; no workflow runs TLC
AHTC-K evidence **2** | `[TESTED]` Ratio is `N/33`, fixed by `COLD_POOL = 32`; the 10× gate cannot fail above N=330
QRADLE evidence **2** | `[TESTED]` 9 of 11 capability questions returned NO or PARTIAL
QRADLE leverage **3** | `[OBSERVED]` Zero of 14 verticals import it. Not 0, because `qratum_asi/sandbox_platform` and `qratum/platform` do
ASI maturity **1** | `[OBSERVED]` Self-declared THEORETICAL ARCHITECTURE; 39% of files carry placeholder language
PQC maturity **1** | `[OBSERVED]` Every scheme labelled "This is a placeholder implementation"
VERTICALS uniqueness **1** | `[OBSERVED]` FLUXA's optimizer is greedy nearest-neighbour with the in-source note "use OR-Tools for production"

---

## 3. WEIGHTINGS AND SENSITIVITY

### 3.1 The five weightings

| Dimension | value-realization (primary) | equal | technologist | acquirer/IP | bootstrapper |
|---|---:|---:|---:|---:|---:|
`technical_uniqueness` | 0.10 | 0.10 | **0.25** | 0.16 | 0.04
`implementation_maturity` | **0.14** | 0.10 | 0.20 | 0.08 | 0.16
`cross_domain_leverage` | 0.08 | 0.10 | 0.15 | 0.08 | 0.04
`commercial_value` | **0.14** | 0.10 | 0.04 | 0.10 | **0.26**
`defensibility` | 0.11 | 0.10 | 0.06 | **0.24** | 0.04
`switching_cost` | 0.04 | 0.10 | 0.02 | 0.12 | 0.04
`strategic_importance` | 0.11 | 0.10 | 0.10 | 0.12 | 0.04
`capital_efficiency` | 0.10 | 0.10 | 0.02 | 0.02 | **0.24**
`competitive_differentiation` | 0.08 | 0.10 | 0.10 | 0.06 | 0.04
`evidence_strength` | 0.10 | 0.10 | 0.06 | 0.02 | 0.10

**Why the primary weighting is shaped this way.** It asks *what is actually worth something to whoever owns this repository today*. `implementation_maturity` and `commercial_value` lead at 0.14 because an unrealized idea in a repository with zero deployments is worth little. `evidence_strength` and `capital_efficiency` sit at 0.10 because an unsubstantiated claim is a liability and an uncapitalized path is a dead end. `switching_cost` is lowest at 0.04 because nothing is deployed, so switching costs cannot yet accrue to anything — weighting them higher would credit assets for stickiness they have not earned.

The other four weightings are deliberately adversarial to the primary one: `technologist` weights commercial value at 0.04; `bootstrapper` weights it at 0.26 (a 6.5× spread). `acquirer/IP` weights defensibility at 0.24 against `bootstrapper`'s 0.04 (a 6× spread). If the conclusion survives both, it is not an artifact of the weights.

### 3.2 Weighted scores

| Asset | value-realization | equal | technologist | acquirer/IP | bootstrapper |
|---|---:|---:|---:|---:|---:|
| **ARBITER** | **6.71** | **6.50** | **6.97** | **5.60** | **7.04** |
| EVIDENCE-DISCIPLINE | 6.19 | 6.10 | 6.89 | 5.26 | 6.28 |
| HCAL | 5.70 | 5.50 | 5.46 | 4.84 | 6.54 |
| CONTRACTS | 4.61 | 4.40 | 4.45 | 3.68 | 5.28 |
| TLA+ suite | 4.21 | 4.20 | 4.39 | 4.08 | 4.32 |
| AHTC-K | 3.88 | 3.70 | 4.53 | 3.46 | 3.98 |
| KERNEL | 3.67 | 3.50 | 4.03 | 3.54 | 3.14 |
| Q-SUBSTRATE | 3.67 | 3.50 | 3.82 | 3.00 | 4.10 |
| QUASIM | 3.56 | 3.40 | 3.83 | 3.02 | 3.66 |
| QRADLE | 2.72 | 2.60 | 2.65 | 2.08 | 3.28 |
| AION | 2.61 | 2.60 | 3.00 | 2.56 | 2.44 |
| VERTICALS | 1.97 | 1.90 | 1.63 | 1.48 | 2.56 |
| QRATUM-ASI | 1.92 | 2.00 | 2.42 | 1.96 | 1.60 |
| PQC | 1.47 | 1.50 | 1.44 | 1.32 | 1.60 |

### 3.3 Rank sensitivity

| Asset | value-real. | equal | technologist | acquirer/IP | bootstrapper | rank spread |
|---|---:|---:|---:|---:|---:|---:|
| **ARBITER** | **1** | **1** | **1** | **1** | **1** | **0** |
| EVIDENCE-DISCIPLINE | 2 | 2 | 2 | 2 | 3 | 1 |
| HCAL | 3 | 3 | 3 | 3 | 2 | 1 |
| CONTRACTS | 4 | 4 | 5 | 5 | 4 | 1 |
| TLA+ suite | 5 | 5 | 6 | 4 | 5 | 2 |
| AHTC-K | 6 | 6 | 4 | 7 | 7 | 3 |
| KERNEL | 7 | 7 | 7 | 6 | 10 | 4 |
| Q-SUBSTRATE | 8 | 8 | 9 | 9 | 6 | 3 |
| QUASIM | 9 | 9 | 8 | 8 | 8 | 1 |
| QRADLE | 10 | 10 | 11 | 11 | 9 | 2 |
| AION | 11 | 11 | 10 | 10 | 12 | 2 |
| VERTICALS | 12 | 13 | 13 | 13 | 11 | 2 |
| QRATUM-ASI | 13 | 12 | 12 | 12 | 13 | 1 |
| PQC | 14 | 14 | 14 | 14 | 14 | 0 |

**The arbiter is rank 1 under all five weightings, with zero rank spread.** `[DERIVED]` The top three are stable in membership under every weighting; only their internal order moves (HCAL rises to 2 under `bootstrapper`, which is exactly what one would expect and is reported as a separate answer in §23 of the main report). QRADLE never ranks above 9th.

**The margin is not large.** Arbiter beats evidence-discipline by 0.52 points under the primary weighting and by 0.08 under `technologist` — well inside the noise of `[ESTIMATE]` scoring. Under `bootstrapper`, HCAL trails by 0.50. **The ranking is robust; the gaps are not.** The main report therefore reports three distinct answers to three distinct questions rather than claiming one asset dominates on all of them.

---

## 4. DEFENSIBILITY MODEL — REBUILD COST

`[ESTIMATE]` Time for a competent, well-resourced competitor given the concept today.

| Layer of the winning asset | Time | What creates the difficulty |
|---|---|---|
| `arbitrate()` core logic | **1 week** | Nothing. 182 lines of clear state-machine code |
| + verdict taxonomy and receipt format | +2–4 weeks | Deciding which denial reasons are genuinely distinct |
| + 270-scenario corpus with pinned hashes | +1–3 months | Judgement about which contention cases matter |
| + written specification | +2–4 weeks | Writing discipline |
| **Subtotal: a credible competing conformance package** | **2–4 months** | Judgement, not difficulty |
| + `no_std` kernel residency, SMP, replay checkpoints | +9–18 months | Kernel engineering is genuinely slow |
| + machine-checked TLA+ with declared bounds and runtime witnesses | +3–6 months | Scarce formal-methods skill |
| + **IEC 61508 / ISO 26262 / DO-178C qualification evidence** | **+2–4 years, $2M–$20M** | **Capital, calendar, auditor relationships, qualified process — not compressible by talent** |
| + third-party adoption of the taxonomy and receipt format | +3–7 years, or never | Network effects — cannot be bought or built |

### The conclusion this forces

```
Rebuild cost of the code            ≈ 1 week          → weakly defensible
Rebuild cost of the package         ≈ 2–4 months      → weakly defensible
Rebuild cost of certification       ≈ 2–4 yr / $2–20M → DEFENSIBLE  ← does not exist yet
Rebuild cost of adoption            ≈ 3–7 yr / never  → DEFENSIBLE  ← does not exist yet
```

**There is no defensible asset in this repository today.** `[DERIVED]` There is one asset whose *path* to defensibility is identifiable and priced. Both moats require resources a $0-start context does not have — the central strategic tension of the whole analysis, and the reason §16 and §24 of the main report recommend publishing the specification free: it is the only move that starts accruing an adoption moat at zero cost.

### Why code volume is not evidence of difficulty here

`[OBSERVED]` 141 commits. 75 (53%) come from accounts explicitly labelled as bots or CI (`copilot-swe-agent[bot]` 33, `github-actions[bot]` 25, `github-actions` 17); including `QuASIM AutoBot` (18) it is 93 of 141 (66%). Together they carry roughly 300,000+ lines — about 2,000+ lines per commit.

`[DERIVED]` A codebase produced substantially by AI coding agents is reproducible by any competitor with equivalent tooling. This collapses the defensibility of every volume-based asset in the repository: QuASIM (67k LOC), QRATUM-ASI (50k), `qratum/` (41k), `qratum_chess/` (26k), `xenon/` (22k). The artifacts that resist it are the ones carrying judgement a generator does not supply — **specifications, adversarial corpora, declared limits, certification evidence, and real deployment history.** Those are exactly the artifacts the determination names.

---

## 5. VALUE CONCENTRATION

### 5.1 Methodology

Each asset's weighted score under the primary weighting is converted to a value share proportional to score raised to a power:

```
share_i  =  score_i^p  /  Σ_j score_j^p
```

`p` encodes convexity — how much more a strong asset is worth than a mediocre one:

- `p = 1` (linear): value is proportional to quality. Too generous to weak assets: it implies a 14-asset portfolio of mediocre components is worth something.
- `p = 2` (squared): moderate convexity.
- `p = 3` (cubed): **used for the headline.** In early-stage deep technology, realisable value is strongly superlinear in asset quality, because a weak asset is not worth a fraction of a strong one — it is usually worth nothing. A greedy VRP heuristic competing with free OR-Tools has no value, not 20% of the arbiter's.

### 5.2 Results

| Convexity | Top 1 | Top 3 | Top 5 | Bottom 9 |
|---|---:|---:|---:|---:|
| Linear (`p=1`) | 12.7% | 35.2% | 51.8% | 48.2% |
| Squared (`p=2`) | 19.3% | 49.6% | 66.3% | 33.7% |
| **Cubed (`p=3`) — headline** | **26.2%** | **62.7%** | **77.7%** | **22.3%** |

**Headline figures `[ESTIMATE]`:**

```
Top 1 capability  (Arbiter)                                              ~26%
Top 3 capabilities (+ Evidence discipline, HCAL)                         ~63%
Top 5 capabilities (+ Contracts/compliance, TLA+ suite)                  ~78%
```

### 5.3 What the concentration actually establishes

Read the sensitivity band, not the point estimate. Under every convexity assumption, the top three hold **35%–63%** of value, and the bottom nine hold **48%–22%**.

The bottom nine are QRADLE, AION, the 14 verticals, QRATUM-ASI, QuASIM, Q-Substrate, the kernel, AHTC-K, and the PQC modules. `[OBSERVED]` Together they are **over 90% of the repository's code**.

**The ratio of code volume to value is inverted across this repository.** That conclusion holds at every convexity level tested and is the robust finding. The exact percentage is not.

### 5.4 Cross-check against the destruction test

The concentration model and the destruction test are independent constructions. They agree:

| Asset | Concentration share (`p=3`) | Destruction cost (`QRATUM_DESTRUCTION_TEST.md`) |
|---|---:|---:|
| Arbiter | ~26% | 45–50% |
| Evidence discipline | ~22% (within top 3) | ~20% |
| QRADLE | ~1.7% | ~5% |
| QRATUM-ASI | ~0.6% | ~2% |
| Verticals | ~0.7% | ~3% |

The arbiter's destruction cost (45–50%) exceeds its concentration share (26%) because destruction captures **dependency** as well as standalone value: removing it leaves nothing distinctive for the evidence apparatus to certify and no reason for the kernel to exist. The two models are measuring different things and both rank the arbiter first. The rank ordering is identical across both for every asset listed. `[DERIVED]`

---

## 6. ECONOMIC MODEL FOR THE TOP THREE

`[ESTIMATE]` Illustrative only. No market research was conducted and these figures should not be relied on for planning.

### Arbiter — Control Authority Specification

| Phase | Capital | Timeline | Revenue model | Realistic outcome |
|---|---|---|---|---|
| Publish spec + corpus | **$0** | 2–4 weeks | None | Reference point established before anyone else defines one |
| Reference library (free, Rust + bindings) | $0 (own time) | 2–3 months | None | Adoption begins; corpus starts receiving real scenarios |
| Paid conformance review | $0 | 3–9 months | $5k–$25k per engagement | First revenue; 1–3 design partners |
| Qualification evidence package | **$2M–$20M** | 2–4 years | $50k–$500k per adopter | **The moat. Requires outside capital** |

**Observation:** the first three phases cost nothing and produce the adoption moat; the fourth produces the certification moat and cannot be bootstrapped. This bifurcation is the single most important planning fact in the analysis.

### HCAL — GPU fleet actuation governance

| Phase | Capital | Timeline | Revenue model |
|---|---|---|---|
| Free dry-run audit CLI | **$0–$1,000** (a GPU to test on) | 4–8 weeks | None — lead generation |
| Paid enforcing agent | $0 | 2–4 months | $5–$50 per managed device per month |
| Site licence | $0 | 6–12 months | $25k–$150k per site per year |

**Fastest path to a first invoice**, with ~80% gross margin and a low regulatory barrier — which is also why it defends poorly (2–6 engineer-months to reproduce).

### Evidence discipline — credibility instrument

Not a revenue line. Extract the claim registry as a standalone open-source tool at $0 cost. Its return is measured in diligence outcomes, not invoices: it is what allows a buyer to believe any other claim about a 164 MB repository in which 53-66% of commits are bot- or CI-authored.

**It must be repaired before it is promoted** `[TESTED]`: widen `claim_language_enforcement` from the policy file to the repository as its own policy already claims; widen the claim registry beyond two files; and replace the AHTC-K workload generator with one whose key space does not determine its own result.

---

## 7. MODEL LIMITATIONS

Stated explicitly, in the spirit of the repository's own `formal/model_limits.md`.

1. **Scores are judgements.** Ten dimensions scored 0–10 by one analyst in one pass. Another analyst would differ by ±2 on many cells. The sensitivity analysis in §3.3 exists because of this, not in spite of it.
2. **Convexity is an assumption, not a measurement.** The `p=3` headline is a modelling choice. §5.2 reports `p=1` and `p=2` so the reader can substitute their own.
3. **No market research.** All commercial figures in §6 are illustrative. No customer, analyst, or buyer was consulted.
4. **Certification costs are external estimates.** The $2M–$20M / 2–4 year range for IEC 61508 / DO-178C qualification is drawn from general industry knowledge `[EXTERNAL]`, not from a quoted engagement.
5. **Corpus quality is unassessed — the largest gap.** The determination's load-bearing claim is that 270 scenarios constitute a conformance suite. I verified they exist and pin hashes `[OBSERVED]`; I did **not** assess their adversarial diversity. If they are near-duplicates and a naive implementation passes all of them, the asset is materially weaker. **This is the first thing an independent reviewer should check.**
6. **Only the `qratum-arbiter` crate was executed.** The `no_std` kernel was not built (PowerShell-only tooling) and was never booted in QEMU during this analysis. Kernel-side claims rest on source inspection alone.
7. **Determinism's share is not independent of the arbiter's.** §8 of the destruction test states this openly: the 35% attributed to deterministic execution is largely the share of the arbiter's value that depends on replay stability. Do not add the two.
