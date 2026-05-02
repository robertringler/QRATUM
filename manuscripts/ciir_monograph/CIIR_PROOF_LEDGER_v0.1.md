# CIIR PROOF LEDGER (DRAFT v0.1)

**Scope:** ch03 → ch08, source-only.
**Source:** `manuscripts/ciir_monograph/chapters/ch03_primitive_definitions.tex`,
`ch04_axiom_system.tex`, `ch05_algebraic_structure.tex`,
`ch06_representation_theory.tex`, `ch07_dynamics.tex`,
`ch08_measurement_theory.tex`.
**Method:** mechanical extraction of every `theorem | lemma | proposition |
corollary | definition | axiom` environment, plus inspection of every
`\begin{proof} … \end{proof}` block. Critical/load-bearing proofs were
read in full (Born rule 8.2, Well-Posedness 7.3, Equilibrium 7.4,
Sufficiency 4.3, Distortion Contraction 8.4, Independence of Axioms 4.1,
CIIR Representation 5.4, Construction of Measurement Operators 8.1).

**Excluded by directive:** no `multi_qubit` references, no
`quasim/ciir/` submodules, no “expected architecture”, no
memory-derived structure. The only inputs are the six chapter files.

---

## 0. Classification key

| Code        | Meaning                                                                              |
|-------------|---------------------------------------------------------------------------------------|
| **PROVED**     | Closed-form mathematical proof in-text, dependencies are theorems already PROVED or are foundational (ZFC, standard functional analysis). |
| **DERIVED**    | Closed-form proof in-text, but conclusion follows from one or more in-system **assumptions** (axioms ch04, or definitions that *encode* what is being proved). Truth is conditional on those assumptions. |
| **NUMERICAL**  | Conclusion supported only by simulation/computation. (None found in ch03–ch08.) |
| **EMPIRICAL**  | Stated as an axiom or postulate, not derived. (All 6 of Chapter 4.) |
| **OPEN**       | No proof, sketch only, scope-restricted relative to claim, or proof contains a logical gap that defeats the asserted independence/derivation. |

Every entry below is one of these five states. “DERIVED” is **not** a
weakness in itself — it is the normal state of an internal theorem —
but DERIVED claims become OPEN when the dependency chain leads back to
an axiom whose strength is exactly the conclusion (circularity).

---

## 1. Claim inventory by chapter

Notation: `[ChX.Y]` is a stable ID assigned by this ledger, ordered by
appearance. Line numbers are first-line of the environment in the
respective chapter file.

### Chapter 3 — Primitive Definitions
File: `ch03_primitive_definitions.tex`. Counts: 22 definitions, 8 lemmas, 7 propositions, 0 theorems, 0 axioms. Proofs in file: 15 (one per lemma/proposition).

| ID    | L#   | Type        | Title                                          | Status   | Proof location | Notes |
|-------|------|-------------|-------------------------------------------------|----------|----------------|-------|
| Ch3.D01 |  44 | Definition | Reality Space                                   | EMPIRICAL | n/a (def)      | Defines $\CR$ as a Polish space; no claim, but introduces unproved structural assumption ($\CR$ exists, is Polish). |
| Ch3.D02 |  78 | Definition | Borel Structure on $\CR$                        | PROVED   | std FA         | Standard. |
| Ch3.L01 |  89 | Lemma      | Standard Borel Property                         | PROVED   | L96–L104       | Cites Kechris; applies standard Polish→standard-Borel theorem. |
| Ch3.P01 | 106 | Proposition| Existence of Non-Trivial Reality Spaces         | PROVED   | L113–L134      | Constructive: $\R^n$ with Lebesgue. |
| Ch3.D03 | 141 | Definition | Smooth Manifold Structure                       | EMPIRICAL | n/a            | |
| Ch3.D04 | 151 | Definition | Riemannian Metric on $\CM$                      | EMPIRICAL | n/a            | |
| Ch3.D05 | 166 | Definition | Levi-Civita Connection                          | PROVED   | std            | Standard ∃! result invoked. |
| Ch3.D06 | 177 | Definition | Riemannian Volume Form                          | PROVED   | std            | |
| Ch3.D07 | 187 | Definition | Constraint Manifold                             | EMPIRICAL | n/a            | Bundles D03–D06 as a structure-defining assumption. |
| Ch3.D08 | 208 | Definition | Constraint Curvature                            | EMPIRICAL | n/a            | |
| Ch3.L02 | 223 | Lemma      | Continuity of Constraint Curvature              | PROVED   | L230–L238      | From smoothness of $g$. |
| Ch3.D09 | 240 | Definition | Constraint Curvature 2-Form                     | EMPIRICAL | n/a            | |
| Ch3.L03 | 253 | Lemma      | Tensoriality of the Constraint Curvature 2-Form | PROVED   | L259–L279      | Standard differential-geometry computation. |
| Ch3.P02 | 281 | Proposition| Bianchi Identity for Constraint Curvature       | PROVED   | L291–L299      | Standard Bianchi for Levi-Civita; cited from std DG. |
| Ch3.D10 | 306 | Definition | Complex Hilbert Space                           | EMPIRICAL | n/a            | |
| Ch3.D11 | 329 | Definition | Cognitive State Space                           | EMPIRICAL | n/a            | $\HC$ separable Hilbert; possibly infinite-dim — used inconsistently with ch07/8 finite-dim assumptions, see Ch7.T03/Ch7.T04 below. |
| Ch3.L04 | 346 | Lemma      | Separability and Dimension                      | PROVED   | L355–L361      | Standard. |
| Ch3.D12 | 363 | Definition | Bounded Operator Algebra                        | EMPIRICAL | n/a            | |
| Ch3.D13 | 377 | Definition | Trace-Class Operators                           | EMPIRICAL | n/a            | |
| Ch3.D14 | 390 | Definition | Density Operator Space                          | EMPIRICAL | n/a            | |
| Ch3.P03 | 405 | Proposition| Topological Properties of $\mathfrak{D}(\HC)$    | PROVED   | L419–L440      | Convexity, weak-* compactness in finite dim. **Note:** infinite-dim case states only weak-* sequential compactness — consistent. |
| Ch3.L05 | 442 | Lemma      | Spectral Decomposition of Density Operators     | PROVED   | L453–L459      | Standard spectral theorem for compact self-adjoint. |
| Ch3.D15 | 468 | Definition | Ontic Operator Algebra                          | EMPIRICAL | n/a            | |
| Ch3.D16 | 487 | Definition | Complete Positivity                             | PROVED   | std            | |
| Ch3.D17 | 500 | Definition | Trace Preservation                              | PROVED   | std            | |
| Ch3.D18 | 508 | Definition | Push-Forward on the Operator Algebra            | EMPIRICAL | n/a            | |
| Ch3.D19 | 525 | Definition | Interface Map                                   | EMPIRICAL | n/a            | Conditions (I1)–(I2) — load-bearing assumption used everywhere downstream. |
| Ch3.P04 | 545 | Proposition| Kraus Representation                            | PROVED   | L557–L566      | Stinespring → Kraus, std. |
| Ch3.L06 | 568 | Lemma      | Interface Maps are Contractive                  | PROVED   | L575–L580      | CPTP ⇒ trace-norm contractive. |
| Ch3.P05 | 582 | Proposition| Stinespring Dilation of Interface Map           | PROVED   | L595–L600      | Std. |
| Ch3.D20 | 607 | Definition | Self-Adjoint Operator                           | PROVED   | std            | |
| Ch3.D21 | 616 | Definition | Observable Algebra                              | EMPIRICAL | n/a            | |
| Ch3.P06 | 629 | Proposition| Algebraic Properties of $\CA(\HC)$               | PROVED   | L643–L658      | $C^*$-algebra structure, std. |
| Ch3.L07 | 660 | Lemma      | Spectral Theorem for Observables                | PROVED   | L671–L678      | Std spectral theorem. |
| Ch3.D22 | 685 | Definition | Constraint-Image Subspace                       | EMPIRICAL | n/a            | $\HC_C := \overline{\Phi(\CA(\CR_{\mathrm{op}}))\,\HC}$. |
| Ch3.P07 | 699 | Proposition| Properties of the Constraint-Image Subspace      | PROVED   | L714–L731      | |
| Ch3.L08 | 733 | Lemma      | Dimension Bound on Constraint-Image Subspace    | DERIVED  | L740–L745      | Depends on Ax 4.4 (used here forwarded). |

**Chapter 3 status totals:** EMPIRICAL = 16 (all definitions encoding structural assumptions); PROVED = 14; DERIVED = 1; OPEN = 0.

---

### Chapter 4 — Axiom System
File: `ch04_axiom_system.tex`. Counts: 6 axioms, 3 theorems, 0 lemmas, 0 propositions. 3 proofs.

| ID    | L#   | Type     | Title                                | Status     | Proof location  | Notes |
|-------|------|----------|---------------------------------------|------------|-----------------|-------|
| Ch4.A1 |  58 | Axiom   | Ontological Priority                  | EMPIRICAL  | n/a             | Postulates non-empty Polish $\CR$ + observer-independence. |
| Ch4.A2 |  89 | Axiom   | Constraint Boundedness                | EMPIRICAL  | n/a             | $\sup_{m\in\CM}\kappa(m) < \infty$. |
| Ch4.A3 | 123 | Axiom   | Interface Completeness                | EMPIRICAL  | n/a             | (IC1) uniqueness up to unitary; (IC2) surjectivity onto sectors. **Strong** — implicitly forces existence of a privileged $\Phi$. |
| Ch4.A4 | 166 | Axiom   | Hilbert Representation                | EMPIRICAL  | n/a             | $\dim\HC \leq \exp(\kappa_{\max}\,\mathrm{vol}(\CM))$. **Used as a hard hypothesis** in Ch8.T04 step 2; quantitative claim is non-trivial. |
| Ch4.A5 | 223 | Axiom   | Measurement Collapse                  | EMPIRICAL  | n/a             | (MC1) Born rule. **OVERLAPS Ch8.T02** (Born derivation) — see logical gap G-L1 below. |
| Ch4.A6 | 281 | Axiom   | Constraint Composition                | EMPIRICAL  | n/a             | Tensor-product composition law. |
| Ch4.T1 | 386 | Theorem | Independence of Axioms                | PROVED (with caveat) | L396–L589 | Constructs six counter-models $\mathfrak M_1,…,\mathfrak M_6$. **Caveat:** $\mathfrak M_1$ defines $\CR=\emptyset$ but then admits "the empty embedding ι: $\CM\to\CR$ is vacuously well-defined since $\CR=\emptyset$ but $\CM$ maps into $\CR$ only if $\CR\neq\emptyset$, which fails" — the bracketed parenthetical concedes the construction is degenerate. Treat as PROVED for axioms 2–6 and **OPEN G-S1** for $\mathfrak M_1$. |
| Ch4.T2 | 596 | Theorem | Consistency of the Axiom System       | PROVED     | L605–L708       | Builds an explicit model on a 1-point manifold + $\HC=\C$. Establishes equiconsistency with ZFC + std functional analysis. |
| Ch4.T3 | 715 | Theorem | Sufficiency of the Axiom System       | **OPEN G-L2**  | L744–L779 marked `\begin{proof}[Proof outline]` | Only an outline. Items (i)–(iv) reference Chapters 5–6 forward. The actual proof of sufficiency is the union of Chapters 5–8 — i.e., this theorem is closed only modulo every later chapter's correctness. Per directive (sufficiency claim, no closed proof) → OPEN. |

**Chapter 4 status totals:** EMPIRICAL = 6 (axioms, expected); PROVED = 2; OPEN = 1 (T3 + caveat on T1 model 1, see G-S1).

---

### Chapter 5 — Algebraic Structure
File: `ch05_algebraic_structure.tex`. Counts: 12 definitions, 8 lemmas, 5 propositions, 4 theorems, 1 corollary. 18 proofs.

| ID      | L#  | Type | Title                                                      | Status  | Proof | Notes |
|---------|-----|------|------------------------------------------------------------|---------|-------|-------|
| Ch5.D01 |  76 | Def  | Constraint Frame                                            | EMPIRICAL | n/a   | |
| Ch5.D02 |  91 | Def  | Constraint Operator Algebra                                 | EMPIRICAL | n/a   | |
| Ch5.D03 | 123 | Def  | Constraint Generators                                       | EMPIRICAL | n/a   | |
| Ch5.L01 | 133 | Lem  | Boundedness of Constraint Generators                        | DERIVED | L145–L159 | Uses Ax 4.2 + Ax 4.4. |
| Ch5.P01 | 161 | Prop | $C^*$-Algebra Structure of $\CK(\HC)$                       | PROVED  | L176–L197 | Closure under norm + adjoint, std. |
| Ch5.L02 | 199 | Lem  | Self-Adjoint Part                                           | PROVED  | L207–L216 | |
| Ch5.D04 | 227 | Def  | Holonomy Tensor                                             | EMPIRICAL | n/a   | Pulled back from Levi-Civita. |
| Ch5.L03 | 243 | Lem  | Antisymmetry of the Holonomy Tensor                         | PROVED  | L248–L255 | Std. |
| Ch5.D05 | 257 | Def  | Effective Planck Constant                                   | EMPIRICAL | n/a   | $\hbar_{\mathrm{eff}}$ is *defined* via curvature-based scaling — load-bearing in Ch5.T1. |
| Ch5.D06 | 285 | Def  | Holonomy Operator                                           | EMPIRICAL | n/a   | |
| Ch5.T1  | 299 | Thm  | Commutation Relations $[\hat C_i,\hat C_j]=i\hbar_{\mathrm{eff}}\hat\Omega_{ij}$ | DERIVED | L325–L389 | Holds *given* the parallel-transport / Levi-Civita ID + Ax 4.4. |
| Ch5.C1  | 391 | Cor  | Lie Algebra Structure                                       | PROVED  | L407–L414 | From T1. |
| Ch5.D07 | 421 | Def  | Constraint Hamiltonian $\hat H_C := \tfrac12\sum_i\hat C_i^2$ | EMPIRICAL | n/a   | |
| Ch5.L04 | 432 | Lem  | Well-Definedness of $\hat H_C$                              | DERIVED | L446–L473 | Uses Ax 4.2 + Ax 4.4. |
| Ch5.P02 | 475 | Prop | Spectral Properties of $\hat H_C$                           | DERIVED | L491–L500 | |
| Ch5.D08 | 502 | Def  | Constraint Vacuum                                           | EMPIRICAL | n/a   | |
| Ch5.L05 | 511 | Lem  | Uniqueness of the Constraint Vacuum                         | DERIVED (cond.) | L519–L534 | **Conditional on irreducibility of $\CK(\HC)$** — irreducibility itself is *not* established as a property of every CIIR system. → flag as G-L3 partial. |
| Ch5.D09 | 543 | Def  | State Functional $\omega_\rho(A):=\Tr(\rho A)$              | EMPIRICAL | n/a   | **Critical:** this definition is what Theorem 8.2 later "derives" the Born rule from. The Born rule's content is encoded *here*, before the Born derivation is stated. See G-L1. |
| Ch5.L06 | 556 | Lem  | Properties of the State Functional                          | PROVED  | L573–L584 | Positivity + normalization, std. |
| Ch5.D10 | 586 | Def  | GNS Triple                                                  | PROVED  | std (GNS) | |
| Ch5.T2  | 606 | Thm  | **CIIR Representation Theorem**                             | PROVED  | L632–L679 | Standard GNS construction. Conditioned on Ch3 + Ax 4.3 (existence of state). |
| Ch5.T3  | 681 | Thm  | Functoriality of CIIR Representations                       | PROVED  | L704–L723 | |
| Ch5.D11 | 730 | Def  | Superselection Sectors                                      | EMPIRICAL | n/a   | |
| Ch5.P03 | 741 | Prop | Decomposition into Superselection Sectors                   | DERIVED | L762–L784 | Uses commutant decomposition, std. |
| Ch5.L07 | 786 | Lem  | Superselection and Density Operators                        | PROVED  | L804–L816 | |
| Ch5.D12 | 825 | Def  | Joint Constraint Operator Algebra                            | EMPIRICAL | n/a   | |
| Ch5.P04 | 841 | Prop | Properties of the Joint Algebra                             | PROVED  | L858–L878 | |
| Ch5.L08 | 880 | Lem  | Entanglement and Constraint Curvature                       | DERIVED (heuristic) | L899–L912 | Bound expressed in terms of curvature scalar; depends on Ch5.D05 ($\hbar_{\mathrm{eff}}$) which is itself defined, not derived. Inequality direction is correct but asserts a *quantitative* link whose tightness is not proved. → G-L4. |
| Ch5.P05 | 914 | Prop | Partial Trace Reduction                                     | PROVED  | L935–L953 | Std. |
| Ch5.T4  | 955 | Thm  | Classical Emergence via Decoherence                         | DERIVED (asymptotic) | L976–L996 | The convergence rate / preferred-basis selection is asserted via einselection; no rate proof. → G-L5. |

**Chapter 5 status totals:** EMPIRICAL = 12 (definitions); PROVED = 10; DERIVED = 7 (some flagged); OPEN-flags = 0 hard / 4 soft (G-L3, G-L4, G-L5 + dependence on G-L1 via D09).

---

### Chapter 6 — Representation Theory & Functorial Structure
File: `ch06_representation_theory.tex`. Counts: 16 definitions, 9 lemmas, 4 propositions, 8 theorems. 21 proofs.

| ID      | L#   | Type | Title                                                | Status | Proof | Notes |
|---------|------|------|------------------------------------------------------|--------|-------|-------|
| Ch6.D01 |  84 | Def  | Cognitive System                                       | EMPIRICAL | n/a   | |
| Ch6.D02 | 101 | Def  | Constraint Morphism                                    | EMPIRICAL | n/a   | |
| Ch6.D03 | 160 | Def  | Composition of Constraint Morphisms                    | EMPIRICAL | n/a   | |
| Ch6.D04 | 172 | Def  | Identity Constraint Morphism                            | EMPIRICAL | n/a   | |
| Ch6.T1  | 180 | Thm  | $\mathbf{CogSys}$ is a Well-Defined Category           | PROVED | L203–L270 | Categorical axioms verified directly. |
| Ch6.L01 | 272 | Lem  | Constraint Morphisms Preserve CPTP Structure           | PROVED | L284–L307 | |
| Ch6.L02 | 309 | Lem  | Hom-sets in $\mathbf{CogSys}$                          | PROVED | L322–L334 | |
| Ch6.D05 | 341 | Def  | CIIR Representation                                    | EMPIRICAL | n/a   | |
| Ch6.D06 | 357 | Def  | Representation Morphism                                 | EMPIRICAL | n/a   | |
| Ch6.D07 | 386 | Def  | Composition of Representation Morphisms                 | EMPIRICAL | n/a   | |
| Ch6.D08 | 393 | Def  | Identity Representation Morphism                        | EMPIRICAL | n/a   | |
| Ch6.T2  | 399 | Thm  | $\mathbf{Hilb_{CIIR}}$ is a Well-Defined Category      | PROVED | L413–L440 | |
| Ch6.L03 | 442 | Lem  | Isomorphisms in $\mathbf{Hilb_{CIIR}}$                 | PROVED | L451–L460 | |
| Ch6.D09 | 467 | Def  | Action of $\mathbf{F}$ on Objects                      | EMPIRICAL | n/a   | |
| Ch6.D10 | 480 | Def  | Action of $\mathbf{F}$ on Morphisms                    | EMPIRICAL | n/a   | |
| Ch6.L04 | 489 | Lem  | $\mathbf{F}(f)$ is a Well-Defined Representation Morphism | PROVED | L496–L522 | |
| Ch6.T3  | 524 | Thm  | $\mathbf{F}$ is a Well-Defined Functor                 | PROVED | L531–L535 | Short. |
| Ch6.T4  | 542 | Thm  | Functoriality of $\mathbf{F}$                          | PROVED | L553–L568 | |
| Ch6.L05 | 570 | Lem  | Faithfulness of $\mathbf{F}$                           | DERIVED | L583–L602 | Faithful **modulo unitary equivalence**, not faithful as a plain functor — language softens the claim correctly. |
| Ch6.D11 | 609 | Def  | Unitary Equivalence of Cognitive Systems               | EMPIRICAL | n/a   | |
| Ch6.T5  | 626 | Thm  | Unitary Equivalence is a Congruence                    | PROVED | L647–L691 | |
| Ch6.P1  | 693 | Prop | CIIR Representation Theorem Revisited                  | DERIVED | L701–L720 | Restatement of Ch5.T2 in functorial language. |
| Ch6.D12 | 727 | Def  | Natural Transformation between Interface Maps          | EMPIRICAL | n/a   | |
| Ch6.D13 | 754 | Def  | Vertical Composition of Natural Transformations        | EMPIRICAL | n/a   | |
| Ch6.D14 | 765 | Def  | Horizontal Composition of Natural Transformations      | EMPIRICAL | n/a   | |
| Ch6.T6  | 781 | Thm  | Natural Transformations form a Category                | PROVED | L790–L824 | |
| Ch6.L06 | 826 | Lem  | Natural Isomorphisms in the Functor Category           | PROVED | L836–L847 | |
| Ch6.P2  | 849 | Prop | Interface Map Deformation                              | DERIVED | L870–L884 | Smooth-deformation existence; uses smoothness of $\Phi$. |
| Ch6.D15 | 894 | Def  | Tensor Product in $\mathbf{CogSys}$                     | EMPIRICAL | n/a   | |
| Ch6.D16 | 934 | Def  | Unit Object in $\mathbf{CogSys}$                        | EMPIRICAL | n/a   | |
| Ch6.T7  | 951 | Thm  | $\mathbf{CogSys}$ and $\mathbf{Hilb_{CIIR}}$ are Symmetric Monoidal Categories | PROVED | L966–L1016 | Direct verification of MacLane coherence. |
| Ch6.L07 |1018 | Lem  | Tensor Product of Morphisms                             | PROVED | L1030–L1046 | |
| Ch6.L08 |1053 | Lem  | Monoidal Coherence Isomorphism                          | PROVED | L1072–L1106 | |
| Ch6.L09 |1108 | Lem  | Unit Coherence Isomorphism                              | PROVED | L1118–L1122 | |
| Ch6.T8  |1124 | Thm  | $\mathbf{F}$ is a Strong Symmetric Monoidal Functor    | PROVED | L1150–L1184 | |
| Ch6.P3  |1186 | Prop | Tensor Product Preserves Probabilistic Interpretation  | DERIVED | L1206–L1220 | Inherits the Ch5.D09 / Ch4.A5 Born content — see G-L1. |
| Ch6.P4  |1222 | Prop | Representation Theorem for Entangled States            | DERIVED | L1244–L1257 | |

**Chapter 6 status totals:** EMPIRICAL = 16; PROVED = 13; DERIVED = 5; OPEN = 0. Chapter 6 is the cleanest chapter — categorical machinery is verified directly.

---

### Chapter 7 — Dynamics, Evolution, and Information Flow
File: `ch07_dynamics.tex`. Counts: 17 definitions, 6 lemmas, 5 propositions, 7 theorems, 1 corollary. 19 proofs.

| ID      | L#   | Type | Title                                              | Status | Proof | Notes |
|---------|------|------|----------------------------------------------------|--------|-------|-------|
| Ch7.D01 |  84 | Def  | Hamiltonian Superoperator $\CL_H$                  | EMPIRICAL | n/a   | |
| Ch7.L01 | 103 | Lem  | Properties of $\CL_H$                              | PROVED | L123–L144 | |
| Ch7.D02 | 156 | Def  | Constraint Lindblad Operators                      | EMPIRICAL | n/a   | Uses constraint generators — load-bearing. |
| Ch7.L02 | 179 | Lem  | Well-Definedness of the Lindblad Operators         | DERIVED | L196–L219 | Conditional on Ax 4.2 + 4.4. |
| Ch7.D03 | 221 | Def  | Constraint Dissipator $\CD_\CM$                    | EMPIRICAL | n/a   | |
| Ch7.L03 | 236 | Lem  | Properties of the Constraint Dissipator           | PROVED | L261–L288 | |
| Ch7.D04 | 295 | Def  | Constraint Liouvillian $\CL = \CL_H + \CD_\CM$     | EMPIRICAL | n/a   | |
| Ch7.T1  | 307 | Thm  | Construction of a Lindblad-form Generator Compatible with CIIR | CONSTRUCTED | L355–L426 | Phase 2.3 closed G-M4 + G-L6 along path (b) of §5.3: Theorem 7.1 was renamed from "Derivation of *the* CIIR master equation" to "Construction of a Lindblad-form generator compatible with CIIR"; (D3) of the construction is now explicitly flagged as a modeling posit (curvature-integral form of $\gamma_k$); (D5) is now stated as constrained uniqueness given (D3) + $V_k \in \CK(\HC)$. Microscopic Lindblad–Davies derivation deferred — see §7.3. |
| Ch7.L04 | 399 | Lem  | GKSL Structure Theorem — Bounded Case              | PROVED | L415–L438 | Standard Lindblad–GKSL theorem, finite/bounded case. |
| Ch7.L05 | 440 | Lem  | Uniqueness of Constraint Lindblad Operators        | DERIVED | L457–L479 | Up to unitary on the noise space — std. |
| Ch7.T2  | 488 | Thm  | CIIR Master Equation — Explicit Form               | DERIVED | L528–L540 | |
| Ch7.D05 | 547 | Def  | Quantum Dynamical Semigroup                        | EMPIRICAL | n/a   | |
| **Ch7.T3** | 567 | Thm  | **Well-Posedness of the CIIR Evolution**           | **OPEN G-S2** | L591–L639 | Hypothesis: `dim H_X < ∞`. Ch3.D11 admits separable infinite-dim $\HC$. Hence theorem is PROVED only on a strict sub-class of CIIR systems and OPEN on the general class. The proof itself uses only Picard–Lindelöf (matrix exponential) and is invalid in the infinite-dim setting (where Hille–Yosida + dissipativity is required). |
| Ch7.D06 | 646 | Def  | Constraint Equilibrium                             | EMPIRICAL | n/a   | |
| Ch7.D07 | 653 | Def  | Effective Temperature                              | EMPIRICAL | n/a   | |
| Ch7.D08 | 667 | Def  | Constraint Gibbs State                             | EMPIRICAL | n/a   | |
| **Ch7.T4** | 678 | Thm  | **Existence and Structure of Equilibrium States**  | **OPEN G-S3** | L699–L754 | Hypothesis: `dim H_X = d < ∞`. Existence proof uses Schauder–Tychonoff on compact $\mathfrak D(\HC)$; infinite-dim $\mathfrak D$ is **not** norm-compact, so the proof method does not extend. Sub-claim (iii) Uniqueness requires irreducibility of $\CK(\HC)$ — itself not guaranteed. → also G-L3. |
| Ch7.P1  | 756 | Prop | Approach to Equilibrium                            | DERIVED | L771–L784 | Uses spectral gap of $\CL$; finite-dim only (inherits G-S2/3). |
| Ch7.D09 | 791 | Def  | Von Neumann Entropy                                | EMPIRICAL | n/a   | |
| Ch7.D10 | 802 | Def  | Constraint Entropy                                 | EMPIRICAL | n/a   | |
| Ch7.L06 | 815 | Lem  | Properties of the Constraint Entropy               | PROVED | L828–L847 | Std. |
| Ch7.D11 | 849 | Def  | Entropy Production Rate                            | EMPIRICAL | n/a   | |
| Ch7.T5  | 858 | Thm  | Entropy Monotonicity — Second Law                  | DERIVED | L874–L915 | Spohn's inequality applied; valid given a faithful invariant state, which is **not** unconditionally available — see G-S3/G-L3. → soft-OPEN. |
| Ch7.C1  | 917 | Cor  | Lyapunov Function                                  | DERIVED | L925–L929 | Inherits T5. |
| Ch7.D12 | 936 | Def  | Trace Distance                                     | PROVED | std    | |
| Ch7.D13 | 947 | Def  | Information Current                                | EMPIRICAL | n/a   | |
| Ch7.D14 | 957 | Def  | Quantum Mutual Information                         | PROVED | std    | |
| Ch7.T6  | 968 | Thm  | Contraction of Distinguishability                  | PROVED | L990–L1014 | Std DPI for trace distance under CPTP. |
| Ch7.T7  |1016 | Thm  | Data-Processing Inequality for CIIR                | PROVED | L1038–L1068 | Std DPI for relative entropy. |
| Ch7.P2  |1070 | Prop | Decoherence under CIIR Evolution                   | DERIVED (asymptotic) | L1084–L1099 | Inherits Ch5.T4 caveats (G-L5). |
| Ch7.P3  |1101 | Prop | Holevo Bound for Constraint-Encoded Information    | PROVED | L1116–L1124 | Std Holevo. |
| Ch7.D15 |1131 | Def  | Evolution Morphism                                 | EMPIRICAL | n/a   | |
| Ch7.D16 |1141 | Def  | Dynamical CogSys Category                          | EMPIRICAL | n/a   | |
| Ch7.P4  |1160 | Prop | Compatibility of Dynamics with the Functor         | PROVED | L1173–L1183 | |
| Ch7.D17 |1185 | Def  | Time-Evolution Natural Transformation              | EMPIRICAL | n/a   | |
| Ch7.P5  |1193 | Prop | Naturality of Time Evolution                       | PROVED | L1206–L1235 | |

**Chapter 7 status totals:** EMPIRICAL = 17; PROVED = 9; DERIVED = 9 (mostly clean); OPEN = 2 (T3, T4). Soft-OPEN propagation: T5, C1, P1, P2 are conditional on the finite-dim restriction.

---

### Chapter 8 — Measurement Theory & Constraint-Induced Distortion
File: `ch08_measurement_theory.tex`. Counts: 22 definitions, 8 lemmas, 6 propositions, 9 theorems, 2 corollaries. 25 proofs.

| ID      | L#   | Type | Title                                                       | Status | Proof | Notes |
|---------|------|------|-------------------------------------------------------------|--------|-------|-------|
| Ch8.D01 | 106 | Def  | Constraint Observable                                       | EMPIRICAL | n/a   | |
| Ch8.D02 | 118 | Def  | Spectral Measure of a Constraint Observable                 | PROVED | std    | |
| Ch8.L01 | 132 | Lem  | PVM Properties from Observable Algebra                      | PROVED | L155–L170 | |
| Ch8.D03 | 172 | Def  | CIIR Projection-Valued Measure (PVM)                        | EMPIRICAL | n/a   | |
| Ch8.D04 | 188 | Def  | CIIR POVM                                                   | EMPIRICAL | n/a   | |
| Ch8.T1  | 206 | Thm  | Construction of Measurement Operators from Constraint Algebra | PROVED | L230–L274 | Naimark dilation, std. |
| Ch8.D05 | 285 | Def  | Measurement Probability Functional                          | EMPIRICAL | n/a   | |
| Ch8.L02 | 297 | Lem  | Probability Measure Properties                              | PROVED | L313–L331 | |
| Ch8.D06 | 333 | Def  | Interface-Induced State                                     | EMPIRICAL | n/a   | |
| **Ch8.T2** | 345 | Thm  | **Derivation of the Born Rule**                             | **OPEN G-L1 (circular)** | L371–L409 | The proof's Step 1 invokes Ch5.D09 ($\omega_\rho(A):=\Tr(\rho A)$) — which **already encodes** $p=\Tr(E\rho)$ for spectral projectors $E$. Steps 2–4 then compute $\omega_\rho(E^{\hat O}(\Delta))$. The genuine independence problem (deriving the trace-form functional uniquely from probabilistic / non-contextuality assumptions on dim≥3) is the content of **Gleason's theorem** and is not invoked. The same content is also stated as **Axiom 4.5 (MC1)**. The closing sentence "No independent probabilistic axiom is required" is therefore false relative to the chapter's own structure: D09 is that axiom, repackaged as a definition. → real logical gap. |
| Ch8.C1  | 411 | Cor  | Born Rule for Discrete Spectra                              | DERIVED | L427–L433 | Inherits G-L1. |
| Ch8.D07 | 443 | Def  | Measurement Instrument                                      | EMPIRICAL | n/a   | |
| Ch8.L03 | 456 | Lem  | Properties of the Measurement Instrument                    | PROVED | L477–L494 | |
| Ch8.T3  | 496 | Thm  | Measurement Update Rule                                     | DERIVED | L520–L550 | Lüders rule; conditional on G-L1. |
| Ch8.D08 | 552 | Def  | Non-Selective Measurement Map                               | EMPIRICAL | n/a   | |
| Ch8.L04 | 562 | Lem  | CPTP Property of Non-Selective Measurement                  | PROVED | L576–L593 | |
| Ch8.D09 | 604 | Def  | Distortion Map                                              | EMPIRICAL | n/a   | |
| Ch8.D10 | 624 | Def  | Distinguishability Metrics                                  | EMPIRICAL | n/a   | |
| Ch8.T4  | 637 | Thm  | Distortion Contraction Bound                                | DERIVED | L664–L747 | (i) Std DPI. (ii) Step 2 invokes Ax 4.4 dimension bound $\dim\HC\leq\exp(\kappa_{\max}\,\mathrm{vol}\CM)$ to size the dilation environment. The exponent is **assumed**; bound is conditional on Ax 4.4. Soft G-L7. |
| Ch8.C2  | 749 | Cor  | Flat Constraint Limit                                       | PROVED | L760–L763 | $K_{\min}=0$ recovers (i). |
| Ch8.D11 | 770 | Def  | Distortion Kernel                                           | EMPIRICAL | n/a   | |
| Ch8.L05 | 792 | Lem  | Dimension Deficit                                           | PROVED | L800–L812 | Linear-algebra. |
| Ch8.T5  | 814 | Thm  | Non-Invertibility of the Distortion Map                     | DERIVED | L835–L868 | Conditional on Ch3.D19 (interface map non-injective), which holds whenever dilation environment $\mathcal K$ is non-trivial; depends on Ax 4.4. |
| Ch8.P1  | 870 | Prop | Quantitative Non-Invertibility                              | DERIVED | L887–L901 | Inherits T5. |
| Ch8.D12 | 912 | Def  | Superposition State                                         | EMPIRICAL | n/a   | |
| Ch8.L06 | 925 | Lem  | Normalisation of Superposition States                       | PROVED | L931–L936 | |
| Ch8.D13 | 938 | Def  | Interference Term                                           | EMPIRICAL | n/a   | |
| Ch8.T6  | 948 | Thm  | Interference from Non-Commutativity                         | DERIVED | L985–L1030 | Conditional on Ch5.T1 (commutation relations) → ultimately on Ax 4.2 + 4.4. |
| Ch8.P2  |1032 | Prop | Interference and Constraint Curvature                       | DERIVED | L1053–L1066 | |
| Ch8.D14 |1077 | Def  | Measurement Context                                         | EMPIRICAL | n/a   | |
| Ch8.D15 |1091 | Def  | Context-Dependent Probability                               | EMPIRICAL | n/a   | |
| Ch8.L07 |1109 | Lem  | Context-Independent Marginals                               | PROVED | L1117–L1130 | |
| Ch8.D16 |1132 | Def  | Joint Outcome Probability                                   | EMPIRICAL | n/a   | |
| Ch8.T7  |1147 | Thm  | Contextuality from Constraint Algebra                       | DERIVED | L1172–L1215 | Uses Ch5.T1; inherits its dependencies. |
| Ch8.P3  |1217 | Prop | Classical Limit: No Contextuality                           | DERIVED | L1231–L1250 | $\hbar_{\mathrm{eff}}\to 0$ limit; depends on Ch5.D05 being a valid limit parameter (G-L4). |
| Ch8.D17 |1257 | Def  | Measurement-Dynamical Composition                           | EMPIRICAL | n/a   | |
| Ch8.T8  |1269 | Thm  | Measurement-Dynamics Compatibility                          | DERIVED | L1300–L1332 | Inherits Ch7 (G-S2/G-S3 if used in infinite dim). |
| Ch8.D18 |1342 | Def  | Measurement-Equipped Cognitive System                       | EMPIRICAL | n/a   | |
| Ch8.D19 |1356 | Def  | Measurement Morphism                                        | EMPIRICAL | n/a   | |
| Ch8.D20 |1372 | Def  | Category $\mathbf{CogSys}_{\mathrm{meas}}$                  | EMPIRICAL | n/a   | |
| Ch8.T9  |1388 | Thm  | Functoriality of Distortion                                 | PROVED | L1411–L1429 | Categorical bookkeeping. |
| Ch8.D21 |1436 | Def  | Generalised Measurement Instrument                          | EMPIRICAL | n/a   | |
| Ch8.L08 |1452 | Lem  | POVM from Interface Map Dilation                            | PROVED | L1474–L1483 | Naimark, std. |
| Ch8.P4  |1485 | Prop | Non-Projective Measurements and Distortion                  | DERIVED | L1499–L1508 | |
| Ch8.D22 |1515 | Def  | Distortion Entropy                                          | EMPIRICAL | n/a   | |
| Ch8.P5  |1527 | Prop | Monotonicity of Distortion Entropy                          | PROVED | L1546–L1565 | |
| Ch8.P6  |1567 | Prop | Distortion Entropy Bound                                    | DERIVED | L1577–L1585 | Uses Ax 4.4 dimension bound. |

**Chapter 8 status totals:** EMPIRICAL = 22; PROVED = 11; DERIVED = 12; OPEN = 1 hard (T2 — Born rule circular). Several DERIVED items inherit the circularity through C1 and T3.

---

## 2. Aggregate counts (ch03–ch08)

| Chapter | Defns | Axioms | Lemmas | Props | Thms | Cors | In-text proofs | Hard-OPEN | Soft gap flags |
|---------|-------|--------|--------|-------|------|------|----------------|-----------|----------------|
| ch03    |   22  |    0   |    8   |   7   |   0  |  0   |       15       |     0     |       0        |
| ch04    |    0  |    6   |    0   |   0   |   3  |  0   |        3       |  1 (T3)   |  1 (T1 model 1)|
| ch05    |   12  |    0   |    8   |   5   |   4  |  1   |       18       |     0     |  3 (L05,L08,T4)|
| ch06    |   16  |    0   |    9   |   4   |   8  |  0   |       21       |     0     |       0        |
| ch07    |   17  |    0   |    6   |   5   |   7  |  1   |       19       | 2 (T3,T4) |  4 propagated  |
| ch08    |   22  |    0   |    8   |   6   |   9  |  2   |       25       |  1 (T2)   |  6 propagated  |
| **Total** | **89** | **6** | **39** | **27** | **31** | **4** | **101** | **4 hard** | **14 soft**   |

Every theorem/lemma/proposition/corollary has an in-text `\begin{proof}`
block (101 proofs for 101 claims). The gap is **not** missing proofs;
the gap is in **proof scope** (T3, T4) and **proof logical content**
(T2, T3-of-ch4).

---

## 3. Dependency graph (textual)

Top-level layering — claim X is "above" claim Y iff X's proof or
hypothesis cites Y:

```
                   ZFC + std functional analysis
                              │
                              ▼
       Ch3 primitives:  Ch3.D01–D22  (Polish $\CR$, $\CM$, $\HC$, $\Phi$, $\CA$)
                              │
                              ▼
        Ch3 lemmas/props:  Ch3.L01–L08, Ch3.P01–P07
                              │
                              ▼
              Ch4 axioms:  Ch4.A1 … Ch4.A6   ◄─── EMPIRICAL anchor
                              │
                              ▼
        Ch4 meta-thms:   Ch4.T1 (Independence), Ch4.T2 (Consistency),
                          Ch4.T3 (Sufficiency)  [outline only]
                              │
                              ▼
   Ch5 algebraic skeleton:  D01–D12, L01–L08, P01–P05, T1–T4, C1
                              │  • Ch5.T1 commutation rel.   [needs Ax 4.2,4.4]
                              │  • Ch5.D09 state functional  ◄── encodes Born
                              │  • Ch5.T2 GNS rep.           [needs Ax 4.3]
                              ▼
  Ch6 categorical lift:  T1–T8, L01–L09, P01–P4
                              │  pure categorical machinery,
                              │  inherits content of Ch5
                              ▼
   Ch7 dynamics:   T1 master eq., T2 explicit, T3 well-posed [dim<∞],
                    T4 equilibrium [dim<∞], T5 second law,
                    T6 trace-DPI, T7 RE-DPI, P1–P5
                              │
                              ▼
   Ch8 measurement: T1 measurement ops, T2 Born rule [G-L1 circular],
                     T3 update rule, T4 distortion bound,
                     T5 non-invertibility, T6 interference,
                     T7 contextuality, T8 measurement-dynamics,
                     T9 distortion functoriality
```

Critical-path dependencies (every ch08 claim depends on at least one
ch04 axiom, transitively):

```
 Ax 4.1 (Ontological Priority)
        ─► Ch3 reality-space facts ─► every later chapter
 Ax 4.2 (Constraint Boundedness)
        ─► Ch5.L01 boundedness
        ─► Ch5.T1 commutation relations
        ─► Ch7.L02 Lindblad well-def
        ─► Ch8.T6 interference
 Ax 4.3 (Interface Completeness)
        ─► Ch3.P05 Stinespring
        ─► Ch5.T2 GNS / CIIR Rep
        ─► Ch6 functor F
        ─► Ch8.T1 measurement-op construction
 Ax 4.4 (Hilbert Representation)        [exponential dim bound]
        ─► Ch5.L01, Ch5.L04
        ─► Ch7.L02
        ─► Ch8.T4 distortion bound (used quantitatively)
        ─► Ch8.P6 distortion entropy bound
 Ax 4.5 (Measurement Collapse / Born)   [redundant w/ Ch8.T2 — circular]
        ─► Ch8.T2 (asserted "derived")
        ─► Ch8.T3 update rule
        ─► Ch8.C1 discrete-spectrum Born
 Ax 4.6 (Constraint Composition)
        ─► Ch5.D12 joint algebra
        ─► Ch6.D15 tensor product in CogSys
        ─► Ch6.T7 / T8 monoidal structure
        ─► Ch8.T7 contextuality (joint algebra)
```

Forward references that are not yet closed:

- Ch4.T3 (Sufficiency) is only an outline; its actual closure is
  the conjunction of Chapters 5–8. This is acceptable structurally
  but means **the axiom system's sufficiency is not proved
  inside ch04** — it is *promised* by chapters 5–8.

---

## 4. Open-gap classification map

Every gap below has a stable ID `G-<class><n>` so closure actions can
be tracked.

### 4.1 Structural gaps (S)

These are gaps caused by *scope mismatch* between a claim's hypothesis
and the surrounding system — not by missing proof steps.

| ID    | Affects     | Failure mode | One-line summary |
|-------|-------------|--------------|------------------|
| **G-S1** | Ch4.T1 model $\mathfrak M_1$ | degenerate counter-model | The model $\mathfrak M_1$ that is supposed to violate Ax 4.1 takes $\CR=\emptyset$, but then admits "the empty embedding $\iota:\CM\to\CR$ is vacuously well-defined since $\CR=\emptyset$ but $\CM$ maps into $\CR$ only if $\CR\neq\emptyset$, which fails". The text concedes the construction is degenerate. Independence of Ax 4.1 from {4.2,…,4.6} is therefore **not strictly demonstrated** by $\mathfrak M_1$ as written. |
| **G-S2** | Ch7.T3, Ch7.P1 | scope restriction | Theorem assumes $\dim\HC<\infty$. Ch3.D11 admits separable infinite-dim $\HC$. Proof method (matrix exponential / Picard–Lindelöf) does not extend; infinite-dim case requires Hille–Yosida / $\CL$ being a *generator*. |
| **G-S3** | Ch7.T4, Ch7.P1, Ch7.T5, Ch7.C1 | scope restriction | Same as G-S2: equilibrium proof uses Schauder–Tychonoff on compact $\mathfrak D(\HC)$, which is not norm-compact in infinite dim. Uniqueness sub-claim further requires irreducibility of $\CK(\HC)$, which is itself unproved. |
| **G-S4** | Ch4 ↔ Ch7/8 | axiom-vs-theorem overlap | Ax 4.5 (Measurement Collapse / Born) is later "re-derived" as Ch8.T2. Either Ax 4.5 is redundant, or T2 is a re-derivation under additional assumptions. The chapters do not state which. |

### 4.2 Logical gaps (L)

These are gaps in the *content* of a proof — the chain of inference is
either incomplete or circular.

| ID    | Affects   | Failure mode | One-line summary |
|-------|-----------|--------------|------------------|
| **G-L1** | Ch8.T2, Ch8.C1, Ch8.T3, Ch6.P3 | circular derivation | Born rule "derived" from `ω_ρ(A):=Tr(ρA)` (Ch5.D09) which already encodes the Born rule. Genuine independence (Gleason's theorem, dim≥3) is not invoked. |
| **G-L2** | Ch4.T3 | proof outline only | Sufficiency theorem has only `\begin{proof}[Proof outline]`. Closure is deferred to chapters 5–8 in prose form; no formal QED. |
| **G-L3** | Ch5.L05, Ch7.T4(iii) | unsupported irreducibility hypothesis | "Uniqueness of constraint vacuum / equilibrium" assumes $\CK(\HC)$ acts irreducibly. Irreducibility is *not* proved as a property of every CIIR system, nor classified as an axiom. |
| **G-L4** | Ch5.L08 | non-quantitative quantitative bound | Entanglement-curvature bound is asserted; proof gives only a one-sided inequality without tightness. The classical-limit invocation of $\hbar_{\mathrm{eff}}\to 0$ in Ch8.P3 then depends on this being a *parameter* of the system, but Ch5.D05 *defines* it. |
| **G-L5** | Ch5.T4, Ch7.P2 | asymptotic without rate | "Classical emergence via decoherence" / "decoherence under CIIR" — asymptotic assertions with no convergence rate, no preferred-basis selection rate. Standard einselection-style argument is invoked but not mechanized. |
| **G-L6** | Ch7.T1 | derivation-by-positing | ~~"Derivation of the CIIR Master Equation" — but the Lindblad operators are *posited* in Ch7.D02 as "constraint Lindblad operators". The theorem verifies consistency, not first-principle derivation.~~ **Closed in Phase 2.3 along path (b) of §5.3:** Ch7.T1 retitled to "Construction of a Lindblad-form Generator Compatible with CIIR"; the construction-vs-derivation distinction is now explicit in the theorem statement (D3 = posit, D5 = constrained uniqueness) and in `Remark~\ref{rem:7.1.scope}`. See §7.3. |
| **G-L7** | Ch8.T4 (ii), Ch8.P6 | bound conditional on Ax 4.4 | The exponential dimension bound from Ax 4.4 is invoked quantitatively. The claim's strength is no greater than that of Ax 4.4 itself. |

### 4.3 Missing proof obligations (M)

These are obligations that the text introduces (by virtue of
definitions or axioms) but never discharges.

| ID    | Source | Obligation never discharged |
|-------|--------|------------------------------|
| **G-M1** | Ax 4.4 | A *constructive* example of a CIIR system with $\dim\HC = \exp(\kappa_{\max}\,\mathrm{vol}\CM)$ — the exponential is asserted as an upper bound but no instance is shown to *saturate* it; without this the axiom is informationally vacuous as a tight quantitative statement. |
| **G-M2** | Ax 4.3 (IC1) | Uniqueness of $\Phi$ "up to unitary on $\HC$" is stated; no example showing the unitary class is non-trivial. |
| **G-M3** | Ch5.D05 | "Effective Planck constant" is introduced as a curvature scaling; no proof that the resulting constant is dimensionally and physically consistent across all CIIR systems (vs. a free parameter). |
| **G-M4** | Ch7.D02 | ~~"Constraint Lindblad operators" are *defined* by a particular formula; no proof that they are the unique microscopically-derived operators (this is also G-L6).~~ **Closed in Phase 2.3 along path (b) of §5.3:** Definition 7.2 is unchanged, but the role it plays in Theorem 7.1 is now explicitly that of a modeling posit (step D3), not a derived identity. The "uniqueness" claim of Lemma 7.5 is now a constrained-uniqueness result (given the D3 posit). See §7.3. |
| **G-M5** | Ch3.D19 | Interface map $\Phi$ is required to be CPTP and to satisfy push-forward over $\CA$. No proof that the constraints (I1)+(I2)+push-forward are jointly satisfiable for *every* triple $(\CR,\CM,\HC)$ admitted by ch03. Existence is reduced to Ax 4.3 — i.e., to an axiom — rather than constructed. |

---

## 5. Closure priority ranking

Per the task's four-tier priority ladder
(axiomatic inconsistencies → undefined operators → unnormalized
probabilistic mappings → recursion stability gaps), mapped onto the
gaps actually present in ch03–ch08:

### Priority 1 — Axiomatic inconsistencies

1. **G-S4 + G-L1 (Born rule double-status).** Ax 4.5 vs. Ch8.T2. Either:
   - declare Ax 4.5 *not* axiomatic and replace it with a Gleason-style
     theorem that derives `Tr(Eρ)` from probabilistic assumptions
     (non-contextuality, σ-additivity, dim≥3) — closing G-L1; **or**
   - demote Ch8.T2 to "consistency check, given Ax 4.5" — closing G-S4
     by accepting Ax 4.5 as the primary statement.
   The current chapters claim *both*, which is logically redundant at
   best and circular at worst.
2. **G-S1 (Ax 4.1 independence model).** Repair $\mathfrak M_1$ in
   Ch4.T1 with a non-empty $\CR$-substitute that fails (OP1) for a
   different reason (e.g., $\CR$ not Polish, or observer-dependent),
   so independence of Ax 4.1 is genuinely demonstrated.

### Priority 2 — Undefined / under-specified operators

3. **G-M4 + G-L6 (Lindblad operator uniqueness).** Ch7.D02 + Ch7.T1.
   Either provide a microscopic derivation (Lindblad–Davies weak
   coupling limit applied to a specific CIIR-environment model) **or**
   restate Ch7.T1 as "Construction of a Lindblad-form generator
   compatible with CIIR" rather than "Derivation of *the* CIIR master
   equation".
4. **G-M3 (Effective Planck constant).** Ch5.D05 needs a normalization
   theorem fixing $\hbar_{\mathrm{eff}}$ as a function of curvature, OR
   an explicit declaration that it is a free parameter of each CIIR
   system.
5. **G-L3 (Irreducibility).** Ch5.L05 + Ch7.T4(iii). Either add an
   axiom of irreducibility (or prove a structural theorem giving
   conditions for it) **or** state both uniqueness claims as
   *conditional* in their headers (currently the conditionality is
   stated in the body of T4 but only obliquely in L05).

### Priority 3 — Probabilistic / measurement mappings

6. **G-L1 (re-listed; primary closure path).** Replace Ch8.T2 step 1's
   appeal to Ch5.D09 with an appeal to a Gleason-type lemma. This is
   the single highest-value logical fix in the ledger because it
   converts the Born rule from an *axiom-restated-as-theorem* into a
   genuine theorem.
7. **G-L7 (Distortion bound conditional on Ax 4.4).** Ch8.T4(ii) and
   Ch8.P6 should explicitly state "Assuming the dimension bound of
   Ax 4.4" in their headers, and the constants should be tracked
   symbolically rather than absorbed into $\kappa_{\max},\mathrm{vol}\CM$.

### Priority 4 — Recursion / dynamical-stability gaps

8. **G-S2, G-S3 (finite-dim restriction in Ch7).** Either:
   - restrict ch03.D11 to finite-dim $\HC$ throughout the monograph, or
   - add a Hille–Yosida-based proof of well-posedness for the bounded
     generator $\CL$ on $\mathcal T_1(\HC)$ for separable $\HC$, and
     replace the Schauder–Tychonoff existence argument in Ch7.T4 with
     a Markov–Kakutani-style argument valid in non-compact convex sets.
9. **G-L5 (Decoherence asymptotics).** Ch5.T4, Ch7.P2. Add a spectral-gap
   argument giving an explicit exponential rate of off-diagonal decay.
10. **G-L4 (Entanglement-curvature inequality).** Tighten Ch5.L08 with
    a saturation example, or weaken its statement to qualitative.
11. **G-M1, G-M2, G-M5 (Existence/saturation obligations).** Lowest
    priority: these become important only after priorities 1–3 are
    closed; before that, they are bookkeeping.

---

## 6. What this ledger does NOT contain (per directive)

- No references to `multi_qubit`, `quasim/ciir/*` submodules,
  `qratum/*` runtime, `tests/*`, or any code path. The ledger's
  scope is the six `.tex` files listed above.
- No invented "expected architecture". Every item above is anchored to
  a concrete line in one of `ch03_primitive_definitions.tex`,
  `ch04_axiom_system.tex`, `ch05_algebraic_structure.tex`,
  `ch06_representation_theory.tex`, `ch07_dynamics.tex`,
  `ch08_measurement_theory.tex`.
- No closure action has been taken — this is Phase 1 (extraction +
  classification) only. Phase 2 (proof-gap closure) requires per-gap
  sign-off on the closure path proposed in §5.

---

*End of `CIIR_PROOF_LEDGER_v0.1.md` (Phase 1, draft).*

---

## 7. Phase-2 closure log (in progress)

This section is appended as Phase-2 closure work begins. Each entry
records: which gap, which closure path was taken (out of the options
in §5), and the precise source-file edits that implement it. No claim
is made about gaps not listed here.

### 7.1 Closure of G-L1 + G-S4 (Born-rule double-status)

- **Status:** closed by the conservative path (option (b) in §5.1).
- **Decision.** Axiom 4.5 (Measurement Collapse) remains the
  *primary* statement of the Born rule in the present version of the
  monograph. Theorem 8.2 is reframed as a
  *compatibility / propagation* result, not an independent derivation.
  A genuine Gleason / Busch-style derivation (option (a) of §5.1) is
  explicitly recorded as deferred future work. This eliminates the
  circularity (G-L1) without committing the monograph to a Gleason
  proof at v0.x, and resolves the axiom-vs-theorem overlap (G-S4) by
  picking one of the two as primary.
- **Source-file edits implementing the closure.**
  - `chapters/ch08_measurement_theory.tex`
    - Chapter intro (was lines 47–56, "central thesis is that
      Axiom 4.5 is *derivable*…") rewritten as a compatibility /
      propagation framing, with forward-pointer to
      `Remark~\ref{rem:8.born-status}`.
    - Bullet item in the chapter-summary list — was "The Born rule is
      *derived* from the trace structure" — replaced by "The Born
      rule of Axiom 4.5(MC1) is shown to *propagate* through the
      trace structure".
    - Subsection heading "Derivation of the Born Rule" → "Compatibility
      of the Born Rule with the Interface Map".
    - Subsection lead-in paragraph rewritten to state explicitly that
      this is *not* a Gleason-style independent derivation.
    - `\begin{theorem}[Derivation of the Born Rule]` →
      `\begin{theorem}[Born-Rule Compatibility of the Interface Map]`.
      Statement clause "Axiom 4.5(MC1) is derived, not assumed"
      replaced by an explicit description of the result as a
      compatibility / propagation theorem; clause (iii) now explicitly
      labels Definition 5.9 as the algebraic restatement of (MC1).
    - Step 4 of the proof: phrase "the Born rule … is therefore a
      *derived* consequence of …" → "the Born expression … is
      therefore the propagation of Axiom 4.5(MC1) along …", and the
      final sentence "No independent probabilistic axiom is required"
      → "This is a compatibility / propagation result; it does not
      eliminate Axiom 4.5".
    - New `\begin{remark}\label{rem:8.born-status}[Status of the Born
      rule in CIIR; closure of ledger gaps G-L1, G-S4]` inserted
      immediately after the proof of Theorem 8.2. The remark spells
      out (a) that the Born rule appears both as Axiom 4.5(MC1) and
      as Theorem 8.2; (b) that Step 1 of the proof invokes
      Definition 5.9 which is itself the algebraic restatement of
      (MC1); (c) that a genuinely independent derivation would
      proceed via Gleason / Busch and is deferred; (d) cross-references
      this ledger.
    - Corollary 8.1 renamed "Born Rule for Discrete Spectra" →
      "Born-Rule Compatibility for Discrete Spectra" (statement and
      proof unchanged — they remain a special case of Theorem 8.2).
    - Chapter-summary "Axiom 4.5 status" entry rewritten: was "now a
      *theorem* … reducing the independent axiom count from 6 to 5";
      now states honestly that Axiom 4.5 remains an axiom in the
      present monograph, that Theorem 8.2 / 8.3 are
      compatibility / consistency statements, and that the axiom
      count is 6 (not 5). Cross-references this ledger and
      `Remark~\ref{rem:8.born-status}`.
    - Dependency-graph row label "Born rule (T 8.2)" → "Born-rule
      compat. (T 8.2)"; dependency list now includes
      `Ax.~\ref{ax:4.5}` alongside `D~\ref{def:5.9}` to make the
      dependency on the axiom explicit.
  - `chapters/ch04_axiom_system.tex`
    - `Remark~\ref{rem:4.5}` (immediately following Axiom 4.5)
      extended with a "Status as primary statement of the Born rule"
      paragraph that points forward to `Remark~\ref{rem:8.born-status}`
      and references this ledger. The axiom statement itself
      (MC1)/(MC2)/(MC3) is unchanged.

- **Untouched.** No other chapter, no other claim, no proof of any
  other theorem, no axiom statement, and no definition was edited
  during this closure. In particular, Definition 5.9 is unchanged.

- **Verification of internal consistency after the edits.**
  - All cross-references resolve to the same labels as before
    (`thm:8.2`, `cor:8.1`, `def:5.9`, `ax:4.5`, `rem:4.5`).
  - The new label `rem:8.born-status` is introduced once in
    `ch08_measurement_theory.tex` and referenced from
    `ch04_axiom_system.tex` and from the chapter-summary block in
    `ch08_measurement_theory.tex`.
  - The mathematical content of Theorem 8.2's proof is unchanged;
    only its surrounding *labelling* and *interpretation* are
    corrected. Therefore no downstream theorem (Theorem 8.3,
    Corollary 8.1, Proposition 6.3) requires its proof to be redone.

- **Remaining Priority-1 work.** G-S1 (Ax 4.1 independence
  counter-model $\mathfrak M_1$) is *not* closed by the present
  edit set. It requires a genuine non-empty replacement of $\CR$
  that fails (OP1) for a different reason; this is recorded as the
  next Phase-2 deliverable after Born-rule closure.

### 7.2 Closure of G-S1 (Ax 4.1 independence countermodel $\mathfrak M_1$)

- **Status:** closed by closure path (c) of §5.1
  (reclassify as model-relative + non-degenerate replacement),
  combined with a structural finding about the typing of
  Axiom 4.1.

- **Reconstruction.** The original $\mathfrak M_1$ in the proof of
  Theorem 4.1 (Ch4 §"Axiom Independence") is
  $\mathfrak M_1 = (\CR, \CM, \iota, \HC, \Phi, \CA(\HC))
  = (\emptyset, \{m_0\}, \text{"empty embedding"}, \mathbb C,
  \mathrm{id}, \mathbb R)$.
  - **Locally OK:** the cognitive piece
    $(\HC=\mathbb C, \Phi=\mathrm{id}, \CA(\HC)=\mathbb R)$
    trivially satisfies the *content* of Ax 4.3–Ax 4.6 read as
    statements about $\HC$ alone; the constraint piece
    $(\CM=\{m_0\}, g\equiv 0, \hatC\equiv 0)$ satisfies (C1) and
    (C3)–(C5) of Def 3.7 in isolation.
  - **Global inconsistency.** Definition 3.7 (C2) requires $\iota$
    to be an injective continuous map $\CM \hookrightarrow \CR$.
    With $\CM = \{m_0\}$ and $\CR = \emptyset$ no such function
    exists: there is no function from a non-empty set to the empty
    set, let alone an injective one. The phrase "vacuously
    well-defined" in the original proof is not vacuously true; it
    is a typing error, and the proof itself flags it on the same
    line. Therefore $\mathfrak M_1$ has no well-typed witness, and
    "Ax 4.2–4.6 hold" cannot be honestly evaluated against it.

- **Failure-type classification.** The defect in $\mathfrak M_1$ is
  **semantic under-specification**, not an independence violation
  and not a hidden dependency:
  - *Not an independence violation.* The intent of $\mathfrak M_1$
    (find a model in which Ax 4.1 fails alone) is sound; the
    construction simply does not produce such a model.
  - *Not a hidden dependency.* No theorem deduces (OP1) or (OP2)
    from Ax 4.2–4.6.
  - *Semantic under-specification.* Clauses (OP1) and (OP2) of
    Axiom 4.1 restate, at the level of an *axiom*, properties that
    Definition 3.1 already imposes at the level of the *type*
    "reality space": Def 3.1(i) requires $\CR \neq \emptyset$ and
    Def 3.1(ii)–(v) requires $(\CR, d_\CR)$ to be a complete
    separable metric space (i.e. Polish). Hence any tuple
    violating (OP1) or (OP2) of Ax 4.1 is, by Def 3.1, *not* a
    reality space at all. The substantive content of Axiom 4.1 is
    therefore concentrated in **(OP3)** (realist independence of
    $\CR$ from the cognitive apparatus).

- **Closure decision.** Of the three options listed in §5.1:
  - (c1) *Strengthen the axiom* (admit empty-$\CR$ models) —
    rejected; would destroy the realist core.
  - (c2) *Derive Ax 4.1 from Ax 4.2–4.6* — rejected; no such
    derivation is available without reading Ax 4.3 / Ax 4.6 as
    covertly entailing a $\Phi$-independent ontic stratum, which
    would itself reintroduce G-S1 elsewhere.
  - (c3) *Reclassify as model-relative + supply a non-degenerate
    countermodel for the substantive (OP3) content* — adopted.

- **Source-file edits implementing the closure.**
  - `chapters/ch04_axiom_system.tex`:
    - New `Remark~\ref{rem:4.1.substantive}` inserted immediately
      after `Remark~\ref{rem:4.1}` in §"Axiom 4.1 — Ontological
      Priority". States the (OP1)/(OP2) type-level redundancy with
      Def 3.1, identifies (OP3) as the substantive content, and
      points forward to the new analysis subsection and to this
      ledger entry.
    - Final paragraph of the proof of Theorem 4.1 amended: the
      conclusion now reads "the six axioms are mutually
      independent (with the caveat for $k=1$ discussed in the
      next subsection and resolved by the non-degenerate
      replacement $\mathfrak M_1'$ of
      Lemma~\ref{lem:4.1.gs1.replacement})". The proof of
      Theorem 4.1 is otherwise unchanged.
    - New subsection §"Countermodel Analysis: $\mathfrak M_1$
      Revisited (Closure of Ledger Gap G-S1)"
      (`\label{sec:ch4:gs1}`) inserted between the proof of
      Theorem 4.1 and §Consistency. Contains:
      - (A) Reconstruction of $\mathfrak M_1$ as written, with
        explicit identification of the locally-OK pieces and the
        global typing failure.
      - (B) Classification of the failure as *semantic
        under-specification*, with the new
        `Remark~\ref{rem:4.1.op1op2.redundant}` recording the
        (OP1)/(OP2) type-level redundancy with Def 3.1 and the
        identification of (OP3) as the substantive content.
      - (C) Closure path (c1)/(c2)/(c3) discussion plus
        `Lemma~\ref{lem:4.1.gs1.replacement}` (existence of a
        non-degenerate $\mathfrak M_1'$ violating only (OP3)) with
        full proof: $\CR := \HC^\star = \mathbb C$ defined as a
        function of a fixed cognitive system $X^\star$;
        $\CM=\{m_0\}$, $\iota(m_0)=0$, $\HC=\mathbb C$,
        $\Phi=\mathrm{id}$. Verifies Def 3.1, Def 3.7, failure of
        (OP3), and Ax 4.2–4.6 individually.
      - `Remark~\ref{rem:4.1.gs1.scope}` recording the scope of
        the closure and the future-revision options
        (drop (OP1)/(OP2) entirely, or retain as type-level
        reminder).
  - `CIIR_PROOF_LEDGER_v0.1.md`: this §7.2 closure entry.

- **Untouched.** No other axiom statement, no other countermodel
  $\mathfrak M_2,\dots,\mathfrak M_6$, no other theorem, no other
  proof, no definition (in particular Def 3.1, Def 3.7, Ax 4.1
  itself) was edited. The replacement $\mathfrak M_1'$ is added as
  a *lemma* in a new subsection rather than substituted in-place
  for $\mathfrak M_1$, so that the proof of Theorem 4.1 continues
  to compile and read as before, and so that downstream references
  to $\mathfrak M_1$ (e.g. the dependency-graph rows
  `Independence (T~\ref{thm:4.1})` later in Ch4) remain valid.

- **Verification of internal consistency after the edits.**
  - All cross-references resolve: existing labels (`def:3.1`,
    `def:3.7`, `ax:4.1`, `ax:4.2`–`ax:4.6`, `rem:4.1`,
    `thm:4.1`, `lem:3.1`) are unchanged; new labels
    (`rem:4.1.substantive`, `sec:ch4:gs1`,
    `rem:4.1.op1op2.redundant`, `lem:4.1.gs1.replacement`,
    `rem:4.1.gs1.scope`) are introduced exactly once each in
    `chapters/ch04_axiom_system.tex` and referenced from the
    forward-pointer `Remark~\ref{rem:4.1.substantive}` and from
    each other.
  - The mathematical content of Theorem 4.1 (statement and proof)
    is unchanged except for the trailing parenthetical
    cross-reference; the lemma $\mathfrak M_1'$ is purely
    *additional*, so no downstream theorem (Theorem 4.2
    Consistency, Theorem 4.3 Sufficiency) requires its proof to
    be redone.

- **Remaining Priority-1 work.** With G-L1, G-S4 (Phase 2.1) and
  G-S1 (Phase 2.2 — this entry) closed, the Priority-1 ladder of
  §5 is exhausted. The next deliverable in the closure log is
  Priority-2 item §5.3 — **G-M4 + G-L6 (Lindblad operator
  uniqueness)** — Ch7.D02 + Ch7.T1 either microscopic
  derivation (Lindblad–Davies weak-coupling limit) or restatement
  of Ch7.T1 as a *construction* rather than a *derivation* of
  "the" CIIR master equation.

### 7.3 Closure of G-M4 + G-L6 (Ch7.T1 — Lindblad operator construction)

- **Status:** closed by closure path (b) of §5.3
  (restate Theorem 7.1 as a *construction* of a Lindblad-form
  generator compatible with CIIR, rather than a *derivation* of
  *the* CIIR master equation).

- **Reconstruction of the gap.** Theorem 7.1 (Ch7
  §"CIIR Master Equation") was titled "Derivation of the CIIR
  Master Equation" and asserted that equation (7.1) is
  "derived, not assumed, from the following chain of deductions
  (D1)–(D5)". Inspection of (D1)–(D5):
  - (D1) = self-adjointness/boundedness of $\hat H_C$ — first-principle, valid.
  - (D2) = CPTP-ness of $\Phi_X$ — first-principle, valid.
  - (D3) = the curvature-integral expression
    $\gamma_k = \tfrac{1}{\hbar_{\mathrm{eff}}} \int_\CM \kappa\,
    |e_k|_g^2\, d\mu_g$ from Definition 7.2. **This is a posit**:
    no system--bath Hamiltonian is fixed, no Born–Markov-secular
    hierarchy is invoked, and no Davies weak-coupling limit is
    taken. The expression is the simplest covariant scalar built
    from $(\kappa, d\mu_g, \hbar_{\mathrm{eff}})$ that vanishes on
    flat constraint manifolds and is orthonormal-frame invariant.
  - (D4) = GKSL structure theorem (bounded case) — first-principle, valid.
  - (D5) = Lemma 7.5 "Uniqueness of Constraint Lindblad Operators".
    **The proof's pivotal sentence** "the dissipation rates *must
    be* proportional to the curvature integral" is a load-bearing
    posit, not a derivation. Once (D3) is granted, Lemma 7.5
    correctly fixes $V_k = \sqrt{\gamma_k}\hat C_k$ up to unitary
    mixing.

  Hence the original "Derivation" claim was overreach: (D3) is a
  modeling choice and (D5) is a constrained-uniqueness result
  conditional on (D3). G-M4 (Ch7.D02 not microscopically derived)
  and G-L6 (Ch7.T1 derivation-by-positing) are the same defect at
  two grain sizes: the Lindblad-operator formula is posited, and
  the master-equation theorem inherits this posit through (D3).

- **Failure-type classification.** The gap is best classified as
  **semantic under-specification** at the level of the theorem
  *statement* (the word "Derivation" claims more than the proof
  delivers), with the Definition-7.2 formula playing the role of
  an unmarked modeling axiom. It is *not* a logical error inside
  the proof — every individual step is correct as a conditional
  statement. Closure is therefore a re-labelling exercise, not a
  re-proof.

- **Closure decision.** Of the two options listed in §5.3:
  - (a) *Microscopic Lindblad–Davies derivation* of the
    curvature-integral form for $\gamma_k$, applied to a specific
    CIIR-environment model (cognitive system $X$ weakly coupled
    to a structured constraint bath, Born–Markov-secular,
    weak-coupling limit). Requires committing to a system--bath
    Hamiltonian, computing bath correlation functions, and
    showing that the Lamb-shifted dissipator reduces to
    Definition 7.2 in some limit. **Rejected for this PR**:
    out of scope for a docs-only edit; tracked as a Phase-3
    open item below.
  - (b) *Restate Ch7.T1 as a construction* of a Lindblad-form
    generator compatible with CIIR, with (D3) explicitly flagged
    as a modeling posit. **Adopted.** This is the surgical,
    minimum-risk closure: it does not change a single equation,
    every downstream reference (Theorem 7.2, Theorem 7.3,
    propositions in Ch7, Theorem 8.8 in Ch8) remains valid by
    construction.

- **Source-file edits implementing the closure.**
  - `chapters/ch07_dynamics.tex`:
    - Theorem 7.1 heading changed
      "Derivation of the CIIR Master Equation"
      → "Construction of a Lindblad-form Generator Compatible
      with CIIR" (label `thm:7.1` unchanged; all
      `\ref{thm:7.1}` cross-references continue to resolve).
    - Theorem statement preamble changed
      "This equation is *derived, not assumed*, from the
      following chain of deductions"
      → "This equation is *constructed*, not derived from
      first principles, by piecing together the following
      compatibility steps".
    - Step (D3) of the theorem statement rewritten to flag the
      curvature-integral form of $\gamma_k$ as an explicit
      modeling posit (with cross-reference to
      `Remark~\ref{rem:7.1.scope}` and to this ledger entry).
    - Step (D5) of the theorem statement rewritten as a
      *constrained uniqueness* claim ("Given the modeling choice
      (D3) and the constraint-algebra restriction
      $V_k \in \CK(\HC)$, the Lindblad operators are determined
      up to a unitary mixing matrix").
    - Proof Step 4 retitled "Identification of Lindblad
      operators" → "Choice of Lindblad operators"; body amended
      to flag the curvature-rate posit explicitly.
    - Lemma 7.5's proof: the load-bearing line "the dissipation
      rates *must be* proportional to the curvature integral"
      replaced with "adopting the modeling choice of
      Definition 7.2 for the dependence of $\gamma_k$ on the
      curvature integral (a posit, not a derivation; see
      Remark~\ref{rem:7.1.scope})".
    - New `Remark~\ref{rem:7.1.scope}` inserted immediately
      after Theorem 7.1, documenting the scope of the
      construction, classifying (D1)/(D2)/(D4) as
      first-principle results, (D3) as a modeling posit, and
      (D5) as constrained uniqueness; pointing forward to the
      Lindblad–Davies derivation as future work; and stating
      that downstream theorems inherit the construction's scope
      unchanged.
    - Summary-table row at the end of Ch7:
      "Master eqn derivation (T~\ref{thm:7.1})"
      → "Master eqn construction (T~\ref{thm:7.1})".
  - `CIIR_PROOF_LEDGER_v0.1.md`:
    - Ch7.T1 row: status `DERIVED` → `CONSTRUCTED`; comment
      updated to record the closure-path (b) decision.
    - Gap rows G-M4 and G-L6: original wording struck through
      and a "Closed in Phase 2.3" note appended pointing to
      this §7.3.
    - This §7.3 closure entry.

- **Untouched (load-bearing invariance).**
  - Equation (7.1) is byte-identical before and after the edits.
  - Definition 7.2 (constraint Lindblad operators), Definition 7.3
    (constraint dissipator), Definition 7.4 (constraint
    Liouvillian) are unchanged.
  - Lemma 7.2 (well-definedness), Lemma 7.4 (GKSL),
    Lemma 7.5 (uniqueness) — only the prose of Lemma 7.5's proof
    changes; its statement and conclusions are unchanged.
  - Theorem 7.2 (explicit form), Theorem 7.3 (well-posedness),
    Theorem 7.4 (equilibrium), Theorem 7.5 (entropy
    monotonicity), and the Ch8 references (Theorem 8.8
    "Measurement-Dynamics Compatibility") all use $\CL$ as
    *constructed* by Theorem 7.1; their proofs do not depend on
    whether $\CL$ is derived or constructed, only on its
    Lindblad form, so they remain valid verbatim.
  - All `\ref{thm:7.1}` and `\ref{eq:7.1}` cross-references in
    Ch7 (lines 53, 607, 1356, 1382, 1387) and elsewhere in the
    monograph continue to resolve to the same theorem.

- **Verification of internal consistency after the edits.**
  - Environment balance in `chapters/ch07_dynamics.tex`:
    `theorem` 7/7, `lemma` 6/6, `remark` 1/1, `proof` 19/19,
    `corollary` 1/1, `definition` 17/17, `enumerate` 14/14,
    `itemize` 4/4, `proposition` 5/5.
  - New label `rem:7.1.scope` defined exactly once and
    referenced 3× from inside Theorem 7.1 / Step 4 of its proof
    / the proof of Lemma 7.5.
  - The label `thm:7.1` is unchanged.

- **Deferred (Phase-3 open item).** Microscopic Lindblad–Davies
  derivation (closure path (a) of §5.3) of the curvature-integral
  form for $\gamma_k$. This requires (i) committing to a specific
  CIIR-environment Hamiltonian $H_{\mathrm{tot}} = H_S + H_B +
  H_{SB}$ where $H_{SB}$ couples the cognitive system to a
  structured bath via the constraint generators $\hat C_k$, (ii)
  computing the bath spectral density and showing it is regular
  enough for the weak-coupling limit, (iii) executing the
  Born–Markov-secular reduction and verifying that the resulting
  Lamb-shifted dissipator reduces to $\sum_k L_k \rho L_k^\dagger
  - \tfrac{1}{2}\{L_k^\dagger L_k, \rho\}$ with $L_k =
  \sqrt{\gamma_k}\hat C_k$ and $\gamma_k$ given by Definition 7.2
  in some natural limit. Tracked as Priority-3 open item in §5.

- **Remaining Priority-2 work.** §5.3 is now closed. Next
  Priority-2 deliverables are (i) §5.4 G-M3 (effective Planck
  constant normalization theorem in Ch5.D05), and (ii) §5.5
  G-L3 (irreducibility of $\CK(\HC)$ in Ch5.L05 + Ch7.T4(iii)).

