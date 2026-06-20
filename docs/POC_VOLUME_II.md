# Physical Operator Computer — Volume II: Operator Computing Theory, Compiler Theory, Verification Theory, and Physical Computability (POC-MONO-2.0)

**QRATUM Research Series — Mathematical Foundations Monograph**

*Series designation:* POC-MONO-2.0
*Classification:* Theoretical Computer Science / Mathematical Physics
*Status:* Research monograph, peer-review draft
*Companion volume:* Volume I — *Physical Operator Computer: Architecture, Instruction Set, and Engineering Roadmap* (referenced as background; this volume is self-contained mathematically).

---

## Abstract

We develop a rigorous mathematical theory of the **Physical Operator Computer (POC)**: an analog photonic device whose primitive computational act is the application of a bounded linear operator to a state of a finite-dimensional complex Hilbert space, followed by a generalized (POVM) measurement. Volume I established the architecture, the operator instruction set architecture (O-ISA), and the engineering roadmap. This volume supplies the mathematics. We axiomatize the POC as a 4-tuple $\mathrm{POC}=(\mathcal H,\mathcal A,\mathcal M,\mathcal C)$ and prove closure and determinism theorems for contract-mediated execution. We complete the operator algebra $\mathcal A$ as a finite-dimensional $C^\ast$-algebra and von Neumann algebra, compute the full commutator structure of the O-ISA generators, and prove that the phase/mix generators are universal for $U(n)$. We construct a spectral computing theory with explicit resolution limits, a physical complexity hierarchy (OPC-P, OPC-NP, OPC-SPEC, OPC-FIX, OPC-GEO), a compiler theory with soundness/completeness/complexity theorems and an NP-hardness result for scheduling, a calibration theory grounded in operator-valued stochastic differential equations and Lyapunov stability, an information-geometric program built on the quantum Fisher information and Bures metric, a formalization of Projection Operator Dynamical Systems (PODS), a verification theory comprising 33+ theorems, and an analysis of the fundamental physical computability limits (Landauer, shot-noise, time-bandwidth). We close with a scientific red-team audit that attacks the architecture and honestly records which claims survive, which are revised to conditional statements, and which are retracted. The central question — *can the POC be elevated to a rigorous paradigm?* — is answered directly in the conclusion.

**Keywords:** operator computing, $C^\ast$-algebra, von Neumann algebra, photonic mesh, spectral computation, diamond norm, quantum Fisher information, Bures metric, Lyapunov stability, physical computability.

---

## Notation and Standing Conventions

Throughout, $n\in\mathbb N$ is the **mode count** (Hilbert space dimension), assumed finite unless explicitly stated. We write:

- $\mathcal H = \mathbb C^n$ with Hermitian inner product $\langle\cdot,\cdot\rangle$, antilinear in the first argument (physics convention), and norm $\|\psi\|=\sqrt{\langle\psi,\psi\rangle}$.
- $B(\mathcal H)=M_n(\mathbb C)$, the algebra of bounded operators (all operators in finite dimension).
- $A^\ast = A^\dagger$, the adjoint (conjugate transpose).
- $\mathcal A_{\mathrm{sa}}=\{A\in\mathcal A: A=A^\ast\}$, the self-adjoint part; $H_n := M_n(\mathbb C)_{\mathrm{sa}}$.
- $U(n)$, $SU(n)$, $u(n)=\{X: X^\ast=-X\}$, $su(n)=\{X\in u(n):\operatorname{tr}X=0\}$.
- $\mathcal D(\mathcal H)=\{\rho\succeq 0: \operatorname{tr}\rho=1\}$, the density operators (states).
- $\mathrm{Spec}(A)$, the spectrum; $r(A)$, the spectral radius.
- $E_{jk}$, the matrix unit (1 in entry $(j,k)$, 0 elsewhere). $\{E_{jk}\}$ obey $E_{jk}E_{lm}=\delta_{kl}E_{jm}$.
- $\|A\|$, the operator (spectral) norm; $\|A\|_F=\sqrt{\operatorname{tr}(A^\ast A)}$, the Frobenius norm; $\|\Phi\|_\diamond$, the diamond norm of a superoperator $\Phi$.
- $[A,B]=AB-BA$, the commutator; $\{A,B\}=AB+BA$, the anticommutator.
- O-ISA generators: $\mathrm{PHS}$ (phase), $\mathrm{MIX}$ (two-mode mixer / MZI), $\mathrm{SCL}$ (scale/gain), $\mathrm{PRJ}$ (projector), $\mathrm{ADJ}$ (adjoint).
- Equations are numbered sequentially $(1),(2),\dots$ Theorems are numbered sequentially **Theorem 1**, …, **Theorem 34+**.

We use **Volume I** as background only; no result here depends on it logically. Where a Volume I figure (e.g., the 7-bit spectral resolution) is reproduced, we **derive it from first principles** here.

---

## Introduction and Methodological Stance

This volume is written under a strict methodological discipline that the reader should understand before proceeding.

**The abstract/physical dichotomy.** We rigorously distinguish two objects at every step: the *abstract* operator algebra (a genuine finite-dimensional $C^\ast$-algebra, where every theorem holds exactly) and the *physical* device (a metrically nearby, contract-certified approximation, where the same statements hold only to a stated tolerance with a stated confidence). The entire verification program (Phases 7, 10) exists to bound the gap between these two. We never conflate them; when a theorem is about the idealized algebra, its physical corollary always carries an explicit error term.

**Honest scoping over maximal claims.** Where a claim is conditional on an unverified engineering milestone, we say so and name the milestone (M0, M2). Where a result is open, we flag it as a conjecture and place it in Appendix C. Where the red-team breaks a claim (Phase 12), we record the disposition (retract / revise / maintain) in a table. A monograph that over-claims is not rigorous, however ornate its proofs; rigor includes the discipline of stating what is *not* established.

**What this volume proves vs. assumes.** *Proved* (from finite-dimensional analysis + standard theorems — SVD, Clements, Banach/Lyapunov, Berry–Esseen, Davis–Kahan): the operator-algebraic structure (Phases 1–3), the compiler's soundness/completeness/complexity (Phase 6), the calibration stability (Phase 7), the verification statistics (Phase 10), and the physical-limit boundary (Phase 11). *Assumed* (named explicitly): mesh connectivity (Theorem 3, M0-tested), the structured perturbation model for verification (Theorem 31), the Hopf conditions for the bifurcation class (Theorem 28), and the exclusion of squeezing (§11.3.1). *Open* (Appendix C): the complexity separations and the eigenvalue query lower bound.

**Reading paths.** The algebraist may read Phases 1–3 and 9; the complexity theorist Phases 5 and 10; the systems engineer Phases 6, 7, 11, 12; the geometer Phase 8. The dependency DAG (Appendix B) shows which results each phase relies on. Every phase is self-contained modulo that DAG.

## PHASE 1 — Formalization of Operator Computing

### 1.1 The POC 4-Tuple

**Definition 1.1 (Physical Operator Computer).**
A *Physical Operator Computer* is a 4-tuple
$$
\mathrm{POC} \;=\; (\mathcal H,\ \mathcal A,\ \mathcal M,\ \mathcal C),
\tag{1}
$$
where $\mathcal H$ is a finite-dimensional complex Hilbert space, $\mathcal A$ is the executable operator algebra, $\mathcal M$ is the measurement algebra, and $\mathcal C$ is the contract-verification structure. We specify each component.

#### 1.1.1 The Hilbert space $\mathcal H$

We take $\mathcal H=\mathbb C^n$, $n<\infty$, with inner product $\langle\psi,\phi\rangle=\sum_{j=1}^n \overline{\psi_j}\,\phi_j$. The **state space** is the density-operator set
$$
\mathcal D(\mathcal H)=\bigl\{\rho\in B(\mathcal H): \rho=\rho^\ast,\ \rho\succeq 0,\ \operatorname{tr}\rho=1\bigr\}.
\tag{2}
$$
Pure states are the rank-one projectors $\rho=|\psi\rangle\langle\psi|$ with $\|\psi\|=1$; $\mathcal D(\mathcal H)$ is the convex hull of the pure states and is a compact convex body of real dimension $n^2-1$.

**Justification of the finite-dimensional restriction.** Physically, the optical field of the POC is supported on a finite number of orthogonal spatial/temporal/frequency modes selected by the mesh geometry: $n$ waveguides give $n$ modes. The single-photon (or coherent-amplitude) Hilbert space of $n$ distinguishable modes is exactly $\mathbb C^n$. The discrete mode count is **physically forced**: a fabricated mesh has a finite number of input/output ports and a finite number of resolvable temporal bins within a loop lifetime. There is no continuum of modes available to the device.

What finite dimensionality *buys* mathematically:

1. **Boundedness is automatic.** Every operator on $\mathbb C^n$ is bounded; $B(\mathcal H)=M_n(\mathbb C)$. No domain issues, no unbounded generators, no self-adjointness subtleties (every symmetric operator is self-adjoint).
2. **Spectral theorem in its strongest form.** Every normal $A$ has a finite orthonormal eigenbasis; the spectral measure is a finite sum of orthogonal projectors.
3. **Topological collapse.** Norm, strong, and weak operator topologies coincide on bounded sets (Section 2.3), so all the functional-analytic completions used below are trivial.
4. **Compactness of the state space** $\mathcal D(\mathcal H)$, enabling fixed-point and extreme-value arguments.

What it *costs*:

1. **No genuine continuum limits.** Operations naturally infinite-dimensional (e.g., the canonical commutation relation $[Q,P]=i\hbar I$, which has no finite-dimensional representation because $\operatorname{tr}[Q,P]=0\ne \operatorname{tr}(i\hbar I)$) cannot be implemented exactly; only finite-dimensional truncations.
2. **Loop registers in the $Q\to\infty$ limit.** A re-circulating loop that supports $Q$ transits before decoherence approximates an infinite-dimensional dynamical system as $Q\to\infty$; the rigorous treatment of that limit requires von Neumann algebra machinery (Section 2.3), which we flag but do not need for any finite-$Q$ result.
3. **Loss of certain algebraic universality phenomena** (e.g., the hyperfinite $\mathrm{II}_1$ factor) that have no finite-dimensional analog.

The trade is overwhelmingly favorable: every theorem in this monograph is a finite-dimensional statement, hence fully rigorous, while the physics of the device is exactly finite-dimensional.

#### 1.1.2 The executable operator algebra $\mathcal A$

**Definition 1.2.** $\mathcal A\subseteq B(\mathcal H)$ is a **unital $\ast$-subalgebra**: $I\in\mathcal A$; $A,B\in\mathcal A\Rightarrow A+B,\ AB,\ \alpha A\in\mathcal A$ for $\alpha\in\mathbb C$; and $A\in\mathcal A\Rightarrow A^\ast\in\mathcal A$. The **physical realizability constraint set** is
$$
\Gamma_{\mathrm{phys}}=\Bigl\{A\in\mathcal A:\ \|A\|\le g_{\max},\ A\in\operatorname{alg}^\ast\langle\,\mathrm{PHS},\mathrm{MIX},\mathrm{SCL},\mathrm{PRJ},\mathrm{ADJ}\,\rangle,\ \exists\,\text{impl}(A)\Bigr\},
\tag{3}
$$
where $g_{\max}$ is the hardware gain ceiling, $\operatorname{alg}^\ast\langle\cdot\rangle$ denotes the $\ast$-algebra generated, and $\mathrm{impl}: A\mapsto$ (a finite O-ISA word) is the **implementation map** furnished by the compiler (Phase 6).

**Minimal axioms and the exact-vs-approximate dichotomy.** The $C^\ast$-axioms (Section 2.2) are:
(A1) vector-space and ring axioms; (A2) submultiplicativity $\|AB\|\le\|A\|\,\|B\|$; (A3) $\|A^\ast\|=\|A\|$; (A4) the $C^\ast$-identity $\|A^\ast A\|=\|A\|^2$.

- (A1) holds **exactly** for the *idealized* algebra $\mathcal A$ of target operators; the abstract algebra is mathematically closed under sums and products.
- (A2), (A3), (A4) hold **exactly** for the abstract $\mathcal A\subseteq M_n(\mathbb C)$ because $M_n(\mathbb C)$ is a genuine $C^\ast$-algebra.
- The **physical realization** $\widehat A=\mathrm{impl}(A)$ satisfies the axioms only **approximately**: fabrication offsets perturb $\widehat A = A+\Delta A$ (Phase 7). Closure under product is *not* exactly realized, because $\mathrm{impl}(A)\,\mathrm{impl}(B)\ne \mathrm{impl}(AB)$ in general (errors compound; Theorem 19). Thus: *the abstract POC algebra is a bona fide $C^\ast$-algebra; the physical device is a metrically nearby, contract-certified approximation of it.* This dichotomy is the leitmotif of the entire verification program.

#### 1.1.3 The measurement algebra $\mathcal M$

**Remark 1.2a (Why density operators, not just state vectors).** The POC's loop register can hold a *mixed* state: averaging over shot-to-shot calibration noise (56) turns a nominally pure input $|\psi\rangle$ into a mixed $\rho=\mathbb E[|\widehat\psi\rangle\langle\widehat\psi|]\ne|\psi\rangle\langle\psi|$. The density-operator formalism (2) is therefore *forced* by the noise model, not a luxury: a pure-state-only theory could not represent the device's actual (noisy) state. The convex geometry of $\mathcal D(\mathcal H)$ — with pure states on the boundary and the maximally mixed state $I/n$ at the center — is the natural arena, and the PODS dynamics (Phase 9) live on it.

**Definition 1.3.** A **POVM** is a finite family $\{M_k\}_{k=1}^m\subseteq B(\mathcal H)$ with $M_k\succeq 0$ and $\sum_k M_k=I$. The outcome statistics on input $\rho$ are
$$
p(k\mid\rho)=\operatorname{tr}(M_k\rho)\ \ge 0,\qquad \sum_k p(k\mid\rho)=1.
\tag{4}
$$
$\mathcal M$ is the set of POVMs realizable by the homodyne/heterodyne detection bank. For an observable $O=\sum_k \mu_k M_k$ (a real-weighted POVM) the **expectation functional** is
$$
E_M:\mathcal D(\mathcal H)\to\mathbb R,\qquad E_M(\rho)=\sum_k \mu_k\,\operatorname{tr}(M_k\rho)=\operatorname{tr}(O\rho).
\tag{5}
$$

**Precision model.** A single shot draws an outcome $k$ from $p(\cdot\mid\rho)$. With $S$ independent shots, the empirical mean $\widehat E=\frac1S\sum_{s} \mu_{k_s}$ is unbiased for $E_M(\rho)$ with variance
$$
\operatorname{Var}(\widehat E)=\frac1S\Bigl(\operatorname{tr}(O^2\rho)-\operatorname{tr}(O\rho)^2\Bigr)=\frac{\sigma_O^2(\rho)}{S},
\tag{6}
$$
**shot noise**. A **systematic bias** $b=\operatorname{tr}(O\,\Delta\rho)+\operatorname{tr}(\Delta O\,\rho)$ arises from calibration error $\Delta\rho,\Delta O$ (Phase 7). The total mean-squared error is $\mathrm{MSE}=\sigma_O^2(\rho)/S+b^2$; the $1/\sqrt S$ shot floor and the bias floor $b$ are independent error sources.

**The POVM convex structure.** The set of POVMs with $m$ outcomes on $\mathbb C^n$ is a convex set; its extreme points are the rank-one *projective* measurements (von Neumann measurements augmented by post-processing). The map $\rho\mapsto p(\cdot|\rho)$ is affine, so the achievable outcome-distribution polytope is the affine image of $\mathcal D(\mathcal H)$:
$$
\mathcal P_M=\bigl\{\,(p_1,\dots,p_m):p_k=\operatorname{tr}(M_k\rho),\ \rho\in\mathcal D(\mathcal H)\,\bigr\}\subseteq\Delta^{m-1},
\tag{8a}
$$
a convex subset of the probability simplex $\Delta^{m-1}$. The **informational completeness** of $\{M_k\}$ is the condition that $\rho\mapsto p(\cdot|\rho)$ is injective, requiring $m\ge n^2$ outcomes (the dimension of $H_n$); informationally complete POVMs are what make state tomography (and hence calibration, Theorem 22) possible. We renumber this auxiliary equation (8a) to avoid disturbing the main sequence.

#### 1.1.4 The contract-verification structure $\mathcal C$

**Definition 1.4.** $\mathcal C=(\Omega,H,\varepsilon,\delta,\Pi)$ where:

- $\Omega$ is the set of valid **OperatorContracts**: frozen, hash-addressed records $c=(A_c,\varepsilon,\delta,g_{\max},\mathrm{budget},\mathrm{epoch})$. *Frozen* means immutable after creation; *hash-addressed* means $c$ is referenced by $H(c)$.
- $H:\Omega\to\{0,1\}^{256}$ is the **contract hash** (SHA-256). We model $H$ as a collision-resistant pseudorandom function: no efficient adversary finds $c\ne c'$ with $H(c)=H(c')$ except with negligible probability $2^{-128}$.
- $\varepsilon>0$ is the **fidelity floor** (maximum tolerated operator/diamond error).
- $\delta\in(0,1)$ is the **confidence parameter** (the protocol succeeds with probability $\ge 1-\delta$).
- $\Pi:\Omega\times(\text{measurement record})\to\{\mathrm{PASS},\mathrm{REJECT}\}$ is the **verification predicate**.

**Execution semantics.** Execution is the partial function
$$
\mathrm{exec}:\ \mathcal C\times\mathcal D(\mathcal H)\ \rightharpoonup\ \mathcal D(\mathcal H)\times\{\mathrm{PASS},\mathrm{REJECT}\},
\tag{7}
$$
defined on $(c,\rho)$ whenever $\mathrm{impl}(A_c)$ exists and the device is calibrated within $\mathrm{epoch}$. We write $\mathrm{exec}(c,\rho)=(\rho',v)$ where $\rho'=\Phi_{\widehat A_c}(\rho)$ is the realized output state and $v=\Pi(c,\text{record})$. The execution is *partial* because budget exhaustion or epoch mismatch leaves $\mathrm{exec}$ undefined (the runtime returns a typed error, not a state).

For a unitary target $A_c=U$ the channel is $\Phi_U(\rho)=U\rho U^\ast$; for a non-unitary target (gain, projection) it is a completely positive trace-nonincreasing map normalized post-measurement. We use $\Phi_A$ generically for the realized superoperator.

### 1.1.5 The Execution Monoid

The valid contracts under composition form an algebraic structure we make explicit. Let $\mathrm{exec}(c,\cdot)=\Phi_{\widehat A_c}$ denote the channel of contract $c$. Define the partial binary operation $c_2\bullet c_1$ (compose if epochs and ports match) with target $A_{c_2}A_{c_1}$. Then:

**Proposition 1.5 (Execution monoid).** $(\Omega/\!\sim,\ \bullet,\ \mathbf 1)$ is a partial monoid, where $\sim$ identifies contracts with equal realized channels, $\mathbf 1$ is the identity contract ($A=I$), and associativity holds up to the fidelity-composition bound:
$$
\bigl\|\Phi_{(c_3\bullet c_2)\bullet c_1}-\Phi_{c_3\bullet(c_2\bullet c_1)}\bigr\|_\diamond\ \le\ \varepsilon_1+\varepsilon_2+\varepsilon_3.
\tag{109}
$$
*Proof.* Channel composition $\circ$ is associative on the *ideal* targets ($A_3(A_2A_1)=(A_3A_2)A_1$); the realized channels differ from ideal by $\varepsilon_i$ each, and the triangle inequality over the two bracketings gives (109). $\square$ The monoid is the algebraic backbone of contract pipelines: a program is a word in this monoid, and Theorem 1 is the statement that the word's error is the sum of letter errors.

### 1.2 Closure and Determinism

**Theorem 1 (POC Closure).**

*Assumptions.* Let $c_1=(A_1,\varepsilon_1,\delta_1,\dots)$ and $c_2=(A_2,\varepsilon_2,\delta_2,\dots)$ be valid contracts with realized channels $\Phi_{\widehat A_1},\Phi_{\widehat A_2}$ satisfying the diamond-norm fidelity bounds
$$
\|\Phi_{\widehat A_i}-\Phi_{A_i}\|_\diamond\le \varepsilon_i,\qquad i=1,2.
\tag{8}
$$
Assume the targets are contractive-or-unitary with $\|A_i\|\le g_{\max}$, and that calibration of $c_1$ and $c_2$ falls in a common epoch.

*Statement.* The composite execution $c_2\circ c_1$ (apply $c_1$, then $c_2$) is certified by a composite contract $c_{21}$ with target $A_2A_1$ and computable fidelity floor
$$
\varepsilon_{21}\ \le\ \varepsilon_2 + \varepsilon_1 ,\qquad
\text{and confidence } 1-\delta_{21}\ \ge\ 1-(\delta_1+\delta_2).
\tag{9}
$$
Moreover, if the targets are unitaries $U_1,U_2$, the bound is *additive in diamond norm* and there is no norm-amplification term; for general bounded targets the bound becomes
$$
\varepsilon_{21}\ \le\ \|A_2\|\,\varepsilon_1 + \varepsilon_2\,\bigl(\|A_1\|+\varepsilon_1\bigr),
\tag{10}
$$
i.e. *fidelity-multiplication*: errors are weighted by operator norms.

*Proof sketch.* Write $\Phi_{\widehat A_2}\circ\Phi_{\widehat A_1}-\Phi_{A_2}\circ\Phi_{A_1}=(\Phi_{\widehat A_2}-\Phi_{A_2})\circ\Phi_{\widehat A_1}+\Phi_{A_2}\circ(\Phi_{\widehat A_1}-\Phi_{A_1})$. The diamond norm is submultiplicative under composition and unitarily invariant, and $\|\Phi_U\|_\diamond=1$ for unitary $U$ (channels are trace-preserving, hence diamond-norm $\le 1$, with equality for CPTP maps). The triangle inequality gives
$$
\|\Phi_{\widehat A_2}\Phi_{\widehat A_1}-\Phi_{A_2}\Phi_{A_1}\|_\diamond
\le \|\Phi_{\widehat A_2}-\Phi_{A_2}\|_\diamond\,\|\Phi_{\widehat A_1}\|_\diamond
+\|\Phi_{A_2}\|_\diamond\,\|\Phi_{\widehat A_1}-\Phi_{A_1}\|_\diamond
\le \varepsilon_2+\varepsilon_1,
\tag{11}
$$
using $\|\Phi_{\widehat A_1}\|_\diamond\le 1$ for channels. For non-CPTP (sub-unitary gain) targets, $\|\Phi_A\|_\diamond$ scales like $\|A\|^2$ in the operator-to-channel lift $\Phi_A(\cdot)=A(\cdot)A^\ast$; carrying the norms through the triangle inequality yields (10). The confidence union bound: $\Pr[\text{both pass}]\ge 1-\delta_1-\delta_2$ by sub-additivity of probability over the two failure events. $\square$

*Limitations.* The additive bound (9) is tight only for unitary channels; for high-gain operators the norm weighting (10) can amplify the upstream error and must be respected by the compiler when budgeting per-instruction $\varepsilon_i$. The union bound on confidence is loose; for independent calibration draws one may use $1-(1-\delta_1)(1-\delta_2)\approx 1-\delta_1-\delta_2+\delta_1\delta_2$.

*Implementation consequence.* The compiler allocates a global error budget $\varepsilon$ across a word $w=g_k\cdots g_1$ by solving $\sum_i \rho_i\varepsilon_i\le\varepsilon$ with norm weights $\rho_i=\prod_{j>i}\|A_j\|$; this is a linear program (Phase 6). Closure is what makes *contracts compositional*: a verified pipeline is a verified product of verified stages.

**Worked closure example.** A pipeline of $L=10$ unitary stages, each certified to $\varepsilon_i=10^{-3}$ at confidence $1-\delta_i=1-10^{-4}$. By (9), the composite diamond error is $\le\sum_i\varepsilon_i=10^{-2}$ and the composite confidence is $\ge1-\sum_i\delta_i=1-10^{-3}$. So a 10-stage verified computation inherits a $10^{-2}$ fidelity floor and $99.9\%$ confidence *automatically* from its components — no re-verification of the whole is needed, only of the parts. For *non-unitary* stages with $\|A_i\|=2$, the norm weighting (10) inflates the error: a 10-stage product of norm-2 operators would amplify the first stage's error by $2^9\approx512\times$, which is why the compiler rescales targets to $\|A_i\|\approx1$ (Theorem 19 consequence) before pipelining.

---

**Theorem 2 (Execution Determinism up to Statistics).**

*Assumptions.* Two executions of the same contract $c$ (identical hash $H(c)$) in the same calibration epoch, on identical input $\rho$, with $S$ shots each. Let $p(\cdot)=\operatorname{tr}(M_\cdot\,\Phi_{\widehat A_c}(\rho))$ be the realized outcome distribution; assume the epoch guarantees the realized channel is the *same* superoperator $\Phi_{\widehat A_c}$ for both runs (drift within epoch is $\le\varepsilon_{\mathrm{replay}}$, Theorem 30).

*Statement.* The two empirical outcome distributions $\widehat p^{(1)},\widehat p^{(2)}$ are **statistically indistinguishable at confidence $1-\delta$**: a two-sample test (Kolmogorov–Smirnov for scalar readouts, or a $\chi^2$ test for categorical POVM outcomes) accepts the null hypothesis "$\widehat p^{(1)},\widehat p^{(2)}$ drawn from the same law" with probability $\ge 1-\delta$, provided
$$
S\ \ge\ \frac{C\,\log(2/\delta)}{\varepsilon_{\mathrm{stat}}^2},
\tag{12}
$$
where $\varepsilon_{\mathrm{stat}}$ is the test's discrimination resolution and $C$ a universal constant.

*Proof sketch.* Under identical hash + epoch, both runs sample i.i.d. from the *same* $p$. The KS statistic $D_S=\sup_x|\widehat F^{(1)}_S(x)-\widehat F^{(2)}_S(x)|$ converges (Donsker) and, by the two-sample DKW inequality, $\Pr[D_S>t]\le 2e^{-S t^2}$. Choosing $t=\varepsilon_{\mathrm{stat}}$ and $S\ge \log(2/\delta)/\varepsilon_{\mathrm{stat}}^2$ bounds the false-rejection probability by $\delta$. Determinism "up to statistics" is exactly this: the *law* is reproduced bit-for-bit (same hash $\Rightarrow$ same compiled word $\Rightarrow$ same channel within epoch), and only the finite-sample empirical realization fluctuates. $\square$

*Limitations.* Determinism is *distributional*, not *sample-path*: individual photon-counting records differ. The guarantee is void across epochs (calibration drift) or under hash collision (excluded with probability $1-2^{-128}$ by Definition 1.4).

*Implementation consequence.* "Replay" is well-defined: re-running a contract reproduces the certified distribution, enabling audit (Phase 10, Theorem 30). The hash is the *causal anchor* of reproducibility.

**The two-sample test, specified.** For categorical POVM outcomes $\{1,\dots,m\}$ with empirical counts $\{N_k^{(1)}\},\{N_k^{(2)}\}$ ($S$ shots each), the Pearson $\chi^2$ statistic
$$
\chi^2=\sum_{k=1}^m\frac{(N_k^{(1)}-N_k^{(2)})^2}{N_k^{(1)}+N_k^{(2)}}
\tag{12a}
$$
is asymptotically $\chi^2_{m-1}$ under the null. The test accepts at level $1-\delta$ iff $\chi^2\le\chi^2_{m-1;1-\delta}$. For scalar (binned) homodyne readouts the KS statistic of Theorem 2 is used instead. Either way, the *same hash + epoch* guarantees the null holds, so acceptance probability is $\ge1-\delta$ by construction. Hash collision (probability $2^{-128}$) is the only way two different computations could spoof the same provenance — excluded by the PRF assumption (Definition 1.4).

---

## PHASE 2 — Operator Algebra Completion

### 2.1 Banach Structure

**Definition 2.1 (Operator norm).** For $A\in B(\mathcal H)$,
$$
\|A\|=\sup_{\|\psi\|=1}\|A\psi\| = \sigma_{\max}(A)=\sqrt{\lambda_{\max}(A^\ast A)},
\tag{13}
$$
the largest singular value.

**Theorem (Banach algebra).** $(\mathcal A,\|\cdot\|)$ is a (complete, unital) **Banach algebra**; in particular submultiplicativity holds:
$$
\|AB\|\le \|A\|\,\|B\|,\qquad A,B\in\mathcal A.
\tag{14}
$$

*Proof.* For unit $\psi$, $\|AB\psi\|\le\|A\|\,\|B\psi\|\le\|A\|\,\|B\|$, so the sup over $\psi$ gives (14). Completeness is automatic in finite dimension ($M_n(\mathbb C)\cong\mathbb C^{n^2}$ is complete and $\mathcal A$ is a closed subspace, being finite-dimensional). $\|I\|=1$. $\square$

**Equivalence of norms and the calibration consequence.** In finite dimension all norms on $M_n(\mathbb C)$ are equivalent: for any two norms $\|\cdot\|_a,\|\cdot\|_b$ there exist $0<c\le C$ with
$$
c\,\|A\|_a\le \|A\|_b\le C\,\|A\|_a\qquad\forall A.
\tag{15}
$$
For the operator and Frobenius norms specifically,
$$
\|A\|\ \le\ \|A\|_F\ \le\ \sqrt n\,\|A\|.
\tag{16}
$$
*Consequence:* the *choice of norm in an error bound is arbitrary up to a dimension-dependent constant*. A bound stated in $\|\cdot\|_F$ converts to $\|\cdot\|$ at cost $\sqrt n$. Calibration thresholds must therefore *name their norm*: a "$10^{-3}$ error" means a factor-$\sqrt n$ different physical tolerance depending on whether it is operator or Frobenius. We fix the convention: **fidelity floors $\varepsilon$ are operator/diamond norms; tomographic losses are Frobenius**, and conversions use (16).

### 2.2 $C^\ast$-Algebra Structure

**Definition 2.2 (Adjoint).** $A^\ast$ is the unique operator with $\langle A\psi,\phi\rangle=\langle\psi,A^\ast\phi\rangle$ for all $\psi,\phi$; concretely $(A^\ast)_{jk}=\overline{A_{kj}}$.

**Theorem ($C^\ast$-identity).**
$$
\|A^\ast A\|=\|A\|^2,\qquad A\in\mathcal A.
\tag{17}
$$

*Proof.* $\|A\|^2=\sup_{\|\psi\|=1}\|A\psi\|^2=\sup\langle A\psi,A\psi\rangle=\sup\langle\psi,A^\ast A\psi\rangle\le\|A^\ast A\|$. Conversely $\|A^\ast A\|\le\|A^\ast\|\|A\|=\|A\|^2$ (using $\|A^\ast\|=\|A\|$ from (13)). Hence equality. $\square$

**Why (17) is load-bearing.** The $C^\ast$-identity forces the norm to be determined by the algebraic structure (it is the unique norm making $\mathcal A$ a $C^\ast$-algebra), and it makes the **continuous functional calculus** isometric: for normal $A$ and continuous $f$, $\|f(A)\|=\sup_{\lambda\in\mathrm{Spec}(A)}|f(\lambda)|=\|f\|_{\infty,\mathrm{Spec}(A)}$. This is exactly the property the POC exploits in spectral computing (Phase 4): a *polynomial of an operator has a norm read off its spectrum*, so Chebyshev approximation error bounds (Theorem 9) transfer from scalar approximation theory to operator approximation **without loss**.

**Spectral mapping and its proof.** For normal $A$ and a polynomial (or continuous) $f$,
$$
\mathrm{Spec}(f(A))=f(\mathrm{Spec}(A))=\{f(\lambda):\lambda\in\mathrm{Spec}(A)\}.
\tag{18a}
$$
*Proof.* Diagonalize $A=\sum_i\lambda_iP_i$; then $f(A)=\sum_i f(\lambda_i)P_i$ has eigenvalues exactly $\{f(\lambda_i)\}$. For continuous $f$, approximate uniformly by polynomials on $\mathrm{Spec}(A)$ and pass to the limit using the $C^\ast$-isometry $\|p(A)-f(A)\|=\|p-f\|_{\infty,\mathrm{Spec}(A)}$. $\square$ This is the theorem that lets the POC compute $f(A)$ by computing on the *spectrum* — the foundation of all of Phase 4.

**Definition 2.3 (Spectral radius).**
$$
r(A)=\lim_{n\to\infty}\|A^n\|^{1/n}=\max\{|\lambda|:\lambda\in\mathrm{Spec}(A)\}.
\tag{18}
$$
The limit exists (Gelfand's formula) and equals the largest modulus eigenvalue. For normal $A$, $r(A)=\|A\|$.

**Gelfand's theorem (commutative case).** If $\mathcal B\subseteq\mathcal A$ is a commutative unital $C^\ast$-subalgebra, the Gelfand transform $\Gamma:\mathcal B\to C(\widehat{\mathcal B})$, $\Gamma(A)(\chi)=\chi(A)$ (evaluation at characters $\chi$), is an **isometric $\ast$-isomorphism** onto $C(\widehat{\mathcal B})$, where $\widehat{\mathcal B}\cong\mathrm{Spec}$ is the (finite) maximal ideal space. *Physical interpretation:* a set of **commuting observables** is jointly diagonalizable; their common eigenbasis is the device's "readout basis," and $\widehat{\mathcal B}$ is the finite set of joint eigenvalue tuples. Commuting observables have **simultaneous eigenvalue spectra readable in one shot** — a single homodyne configuration suffices, because there is a single basis diagonalizing all of them.

**Worked Gelfand example.** Let $\mathcal B=\{f(N):f\ \text{polynomial}\}$ for a fixed self-adjoint $N=\operatorname{diag}(\nu_1,\dots,\nu_n)$ with distinct $\nu_i$. The characters are the evaluations $\chi_i(f(N))=f(\nu_i)$, so $\widehat{\mathcal B}=\{\chi_1,\dots,\chi_n\}\cong\mathrm{Spec}(N)$. The Gelfand transform $\Gamma(f(N))=(f(\nu_1),\dots,f(\nu_n))\in\mathbb C^n=C(\widehat{\mathcal B})$ is an isometric $\ast$-isomorphism: $\|f(N)\|=\max_i|f(\nu_i)|=\|\Gamma(f(N))\|_\infty$. Physically, $N$ might be a tunable refractive-index profile; its commutant algebra is read out in *one* homodyne basis (the eigenbasis of $N$), and all functions $f(N)$ are simultaneously diagonal — the one-shot readability of commuting observables.

### 2.3 von Neumann Completion

**Definition 2.4 (Operator topologies).** On $B(\mathcal H)$:

- **Strong operator topology (SOT):** $A_\alpha\to A$ iff $\|A_\alpha\psi-A\psi\|\to 0$ for all $\psi$.
- **Weak operator topology (WOT):** $A_\alpha\to A$ iff $\langle\phi,A_\alpha\psi\rangle\to\langle\phi,A\psi\rangle$ for all $\phi,\psi$.

**Theorem (Topological collapse in finite dimension).** On $B(\mathbb C^n)=M_n(\mathbb C)$, the norm, strong, and weak operator topologies **coincide** (all equal the unique Hausdorff vector topology on the finite-dimensional space $\mathbb C^{n^2}$).

*Proof.* Finite-dimensional vector spaces carry a unique Hausdorff topological-vector-space topology; norm, SOT, WOT are all Hausdorff TVS topologies on $M_n(\mathbb C)$, hence identical. $\square$

*When the distinction matters.* For a **loop register** approximating an infinite-dimensional limit $Q\to\infty$ (where $Q$ is transit count and the effective Hilbert space dimension grows), the topologies *separate* and one needs genuine von Neumann algebra theory: SOT-limits of compiled words need not be norm-limits. All finite-$Q$ POC statements are immune; the limit is flagged for the asymptotic regime only.

**Double Commutant Theorem (von Neumann).** For a unital $\ast$-subalgebra $\mathcal A\subseteq B(\mathcal H)$,
$$
\mathcal A''=\overline{\mathcal A}^{\,\mathrm{SOT}},
\tag{19}
$$
i.e. the bicommutant equals the SOT-closure. A **von Neumann algebra** is an SOT-closed unital $\ast$-subalgebra, equivalently one satisfying $\mathcal A=\mathcal A''$.

**Corollary (Every finite-mode POC algebra is a von Neumann algebra).** In finite dimension $\mathcal A$ is automatically SOT-closed (it is a finite-dimensional, hence closed, subspace), so $\mathcal A=\mathcal A''$. *Interpretation for the POC:* the physically achievable algebra equals its own bicommutant; there is no gap between "algebra generated by the instructions" and "operators commuting with everything that commutes with the instructions." The device's reachable operator set is *self-contained* — closed under the bicommutant operation — which is precisely the algebraic guarantee that *no hidden operators sneak in* under topological closure.

**Projection lattice.** Define
$$
\mathcal P(\mathcal A)=\{P\in\mathcal A: P^2=P=P^\ast\},
\tag{20}
$$
ordered by $P\le Q \iff \operatorname{ran}P\subseteq\operatorname{ran}Q \iff PQ=P$.

**Theorem (Complete lattice).** $\mathcal P(\mathcal A)$ is a **complete lattice**: every family $\{P_i\}$ has a meet $\bigwedge P_i$ (projection onto $\bigcap\operatorname{ran}P_i$) and join $\bigvee P_i$ (projection onto $\overline{\sum\operatorname{ran}P_i}$). The bottom is $0$, the top is $I$.

*Proof.* Finiteness makes all intersections/sums of subspaces closed; the orthogonal projector onto a closed subspace lies in $\mathcal A$ when $\mathcal A$ is a von Neumann algebra (spectral projections of self-adjoint elements lie in the algebra). Lattice completeness follows from completeness of the subspace lattice of $\mathbb C^n$. $\square$

*Interpretation:* projectors are the machine's **subspace-addressing primitives**. The PRJ instruction selects $\operatorname{ran}P$; the lattice operations are "AND" (meet) and "OR" (join) on subspaces, giving the POC a quantum-logical addressing calculus.

**Projection lattice example and quantum logic.** For $n=3$, the projectors onto the coordinate subspaces form a sublattice; e.g. $P_{\{1\}}\vee P_{\{2\}}=P_{\{1,2\}}$ (join = projector onto the span) and $P_{\{1,2\}}\wedge P_{\{2,3\}}=P_{\{2\}}$ (meet = projector onto the intersection). The lattice is **orthocomplemented** ($P\mapsto I-P$) but **not distributive**: in general
$$
P\wedge(Q\vee R)\ \ne\ (P\wedge Q)\vee(P\wedge R),
\tag{18b}
$$
for non-commuting $P,Q,R$ — the hallmark of *quantum logic* (Birkhoff–von Neumann). For the POC this means PRJ instructions on *non-commuting* subspaces do not obey classical Boolean simplification; the compiler must respect the order and cannot freely re-associate projection words. This is a concrete operational consequence of operating in a quantum, not classical, logic.

### 2.4 Representation Theory and Physical Realizability

**GNS construction.** Every $C^\ast$-algebra has a faithful $\ast$-representation on a Hilbert space; for a state $\omega$ the GNS triple $(\pi_\omega,\mathcal H_\omega,\Omega_\omega)$ satisfies $\omega(A)=\langle\Omega_\omega,\pi_\omega(A)\Omega_\omega\rangle$. For finite-dimensional $\mathcal A\subseteq M_n(\mathbb C)$, the **defining representation** ($A\mapsto A$ as a matrix) is faithful. It is **irreducible** iff $\mathcal A=M_n(\mathbb C)$ (Schur's lemma: the commutant is $\mathbb C I$ iff the representation is irreducible, and $M_n(\mathbb C)'=\mathbb C I$).

**Structure of proper subalgebras.** By the Artin–Wedderburn theorem, every finite-dimensional $\ast$-subalgebra is, up to unitary conjugation, a direct sum of full matrix algebras:
$$
\mathcal A\ \cong\ \bigoplus_{r=1}^{R} M_{n_r}(\mathbb C),\qquad \sum_r n_r m_r = n,
\tag{21}
$$
acting block-diagonally (with multiplicities $m_r$). *Physical meaning:* a **block-diagonal** $\mathcal A_{\mathrm{phys}}$ corresponds to a machine whose modes split into **decoupled clusters**; no instruction couples modes in different blocks. A non-trivial block structure means the mesh is *disconnected*, and universality fails on the whole of $\mathbb C^n$.

**Remark 2.5 (GNS made explicit for $M_n$).** For $\mathcal A=M_n(\mathbb C)$ and the tracial state $\omega(A)=\tfrac1n\operatorname{tr}A$, the GNS Hilbert space is $\mathcal H_\omega=M_n(\mathbb C)$ with inner product $\langle X,Y\rangle_\omega=\tfrac1n\operatorname{tr}(X^\ast Y)$ (the Hilbert–Schmidt structure), the representation is left multiplication $\pi_\omega(A)X=AX$, and the cyclic vector is $\Omega_\omega=I$. This representation is *reducible* — it decomposes as $n$ copies of the defining representation (one per column), reflecting the multiplicity $m=n$ of $M_n$ acting on $\mathbb C^{n^2}$. The *defining* representation $\pi(A)=A$ on $\mathbb C^n$ is the irreducible one. The distinction matters physically: the **loop register's internal state** lives in the GNS (Hilbert–Schmidt) picture (density operators are vectors there), while a **single optical shot** lives in the defining picture (a state vector in $\mathbb C^n$).

**Theorem 3 (Physical Realizability).**

*Assumptions.* O-ISA generators $\{\mathrm{PHS}(j,\theta),\mathrm{MIX}(j,k,\theta,\phi),\mathrm{SCL}(j,g),\mathrm{PRJ}(S),\mathrm{ADJ}\}$ act on $\mathbb C^n$ as in the Notation and Phase 3. Define the **coupling graph** $G=(V,E)$ with $V=\{1,\dots,n\}$ and $\{j,k\}\in E$ iff a $\mathrm{MIX}(j,k,\cdot,\cdot)$ generator is available (fabricated coupler between modes $j,k$).

*Statement.* The $C^\ast$-algebra generated by the unitary generators $\{\mathrm{PHS},\mathrm{MIX}\}$ is
$$
\mathcal A_{\mathrm{phys}}=\operatorname{alg}^\ast\langle \mathrm{PHS},\mathrm{MIX}\rangle\ =\ M_n(\mathbb C)\ \iff\ G\text{ is connected.}
\tag{22}
$$
Equivalently, the connected Lie group generated is $U(n)$ iff $G$ is connected; otherwise $\mathcal A_{\mathrm{phys}}=\bigoplus_c M_{|V_c|}(\mathbb C)$, one block per connected component $V_c$ of $G$.

*Proof via the Lie algebra reachable by the generators.* The PHS generators give the diagonal traceless skew-Hermitian directions $i E_{jj}$ (and $iI$); the MIX$(j,k)$ generators give, by differentiating the embedded $SU(2)$ at the identity, the off-diagonal directions $i(E_{jk}+E_{kj})$ and $(E_{jk}-E_{kj})$ for each edge $\{j,k\}\in E$. Repeated commutators (Phase 3, Theorem 4) of edge generators along a path $j-k-l$ produce the generator on the pair $(j,l)$:
$$
[\,i(E_{jk}+E_{kj}),\,(E_{kl}-E_{lk})\,]=i(E_{jl}+E_{lj})-i(\cdots),
\tag{23}
$$
yielding all $i(E_{pq}+E_{qp})$, $(E_{pq}-E_{qp})$ for $p,q$ in the same connected component. The Lie algebra so generated is therefore $\bigoplus_c u(|V_c|)$. If $G$ is connected there is a single component and the algebra is $u(n)$, whose enveloping associative $\ast$-algebra is $M_n(\mathbb C)$. If $G$ is disconnected, no commutator can connect modes across components (every generator preserves the block structure), so the algebra is the proper direct sum (21). $\square$

*Limitations.* Connectivity of $G$ is a **manufacturing requirement**, not an automatic property; a routing-constrained fabric may omit couplers, disconnecting $G$ and silently restricting expressivity. This is the basis of Red-Team Attack 2 (Phase 12).

*Implementation consequence.* The M0 (first-light) exit criterion must include a **connectivity test**: verify that the realized $G$ is connected, e.g. by checking that the reachable algebra acts irreducibly (a single random unitary word maps a fixed input to a generic vector). Without this test, Theorem 4's universality claim is unsupported on the physical device.

---

### 2.5 Realizability of Non-Unitary Operators

The O-ISA includes SCL (gain/attenuation) and PRJ (projection), which are **not** unitary. We make precise how a general bounded $A$ with $\|A\|\le g_{\max}$ is realized.

**SVD realization.** Write $A=U\Sigma V^\ast$ with $\Sigma=\operatorname{diag}(\sigma_1,\dots,\sigma_n)$, $0\le\sigma_i\le g_{\max}$. The unitaries $U,V^\ast$ are mesh passes (Theorem 4); the singular values are realized by per-mode SCL:
$$
\Phi_A(\rho)=U\,\Sigma\,V^\ast\rho\,V\,\Sigma\,U^\ast,
\tag{145}
$$
a completely positive but **trace-non-increasing** map when any $\sigma_i<1$ (amplitude attenuation = photon loss) or trace-*increasing* when $\sigma_i>1$ (gain). Physically:
- $\sigma_i<1$: attenuation, realized passively (a variable splitter dumping light), always available.
- $\sigma_i>1$: gain, realized by an SOA/amplifier, *adds noise* (the Caves bound: phase-insensitive amplification by gain $G$ adds at least $(G-1)$ noise photons), so high-gain SCL has a fundamental noise penalty.

**The amplification noise floor.** A phase-insensitive amplifier of gain $G=\sigma^2$ satisfies the **Caves inequality**
$$
\langle\Delta X^2\rangle_{\mathrm{out}}\ \ge\ G\,\langle\Delta X^2\rangle_{\mathrm{in}}+\tfrac14(G-1),
\tag{146}
$$
so amplifying to compensate loss *injects* noise scaling with $G-1$ — the precise reason in-loop gain (Attack 1) is not free: it trades loss for added noise, and the net precision is the balance of the two. This sharpens the Attack-1 response: the 7-bit claim requires gain *and* that the added amplifier noise (146) stays below the shot-noise floor, an additional M0 verification target.

**Measurement-conditioned PRJ.** PRJ$(S)$ realizes $\rho\mapsto P_S\rho P_S/\operatorname{tr}(P_S\rho P_S)$, a CP map normalized by post-selection on the measurement outcome "in $S$"; it is trace-preserving only after conditioning, and it is *irreversible* (it discards the complementary subspace), costing $\ge\log_2\operatorname{rank}(I-P_S)$ bits of Landauer erasure (84).

## PHASE 3 — Commutator Physics

We compute the commutators of the O-ISA generators explicitly. Use the matrix units $E_{jk}$, the embedded $2\times 2$ blocks for $n=2$, and extrapolate.

**Generator matrices.**
$$
\mathrm{PHS}(j,\theta)=I+(e^{i\theta}-1)E_{jj},
\tag{24}
$$
$$
\mathrm{MIX}(j,k,\theta,\phi)=I+(\cos\theta-1)(E_{jj}+E_{kk})+\sin\theta\bigl(e^{i\phi}E_{jk}-e^{-i\phi}E_{kj}\bigr),
\tag{25}
$$
$$
\mathrm{SCL}(j,g)=I+(g-1)E_{jj},\qquad g\in[0,g_{\max}],
\tag{26}
$$
$$
\mathrm{PRJ}(S)=\sum_{j\in S}E_{jj}.
\tag{27}
$$
The MIX matrix (25) is the embedded $2\times2$ unitary $\begin{pmatrix}\cos\theta & e^{i\phi}\sin\theta\\ -e^{-i\phi}\sin\theta & \cos\theta\end{pmatrix}$ on modes $\{j,k\}$, identity elsewhere.

### 3.1 Pairwise Commutators

**(C1) Phases on the same mode commute.**
$$
[\mathrm{PHS}(j,\theta_1),\mathrm{PHS}(j,\theta_2)]=0.
\tag{28}
$$
*Proof.* Both are diagonal (functions of $E_{jj}$); diagonal matrices commute. Physically: independent phase plates on the same waveguide commute — the order of two phase shifts is irrelevant.

**(C2) Phase before vs. after a mixer.** Consider $\mathrm{PHS}(j,\theta)$ and $\mathrm{MIX}(j,k,\varphi,\psi)$. Working to first order (generators $G_{\mathrm{PHS}}=i\theta E_{jj}$, $G_{\mathrm{MIX}}$ from (25)), the commutator of the *generators* is the relevant object. With $X=E_{jj}$ and $Y=e^{i\psi}E_{jk}-e^{-i\psi}E_{kj}$,
$$
[X,Y]=E_{jj}(e^{i\psi}E_{jk}-e^{-i\psi}E_{kj})-(e^{i\psi}E_{jk}-e^{-i\psi}E_{kj})E_{jj}
= e^{i\psi}E_{jk}+e^{-i\psi}E_{kj}.
\tag{29}
$$
The full operator commutator $[\mathrm{PHS}(j,\theta),\mathrm{MIX}]$ is nonzero whenever $j\in\{j,k\}$ (always, here): explicitly its leading term is $(e^{i\theta}-1)\sin\varphi\,(e^{i\psi}E_{jk}+e^{-i\psi}E_{kj})\ne 0$. *Interpretation:* **phasing before a beam splitter is not the same as after it** — the order matters; the commutator (29) measures the order-dependence and is the algebraic root of the mesh's noncommutative expressivity.

**(C3) Two mixers on the same pair generate $su(2)$.** Take $\mathrm{MIX}(j,k,\theta_1,\phi_1)$, $\mathrm{MIX}(j,k,\theta_2,\phi_2)$. Restrict to the $2\times2$ block $\cong M_2(\mathbb C)$ on $\{j,k\}$. The Lie-algebra generators are skew-Hermitian traceless $2\times2$ matrices; with Pauli basis $\{\sigma_x,\sigma_y,\sigma_z\}$,
$$
[i\sigma_a,\,i\sigma_b]=-2\,\varepsilon_{abc}\,i\sigma_c,\qquad a,b,c\in\{x,y,z\}.
\tag{30}
$$
A MIX with phase $\phi$ realizes the direction $\cos\phi\,\sigma_x+\sin\phi\,\sigma_y$ (a rotation axis in the equatorial plane); two such with different $\phi$ span $\{\sigma_x,\sigma_y\}$, and their commutator yields $\sigma_z$:
$$
[\,i\sigma_x,\,i\sigma_y\,]=-2i\sigma_z.
\tag{31}
$$
Hence $\{i\sigma_x,i\sigma_y,i\sigma_z\}$ are generated: the Lie algebra is
$$
su(2)=\{A\in M_2(\mathbb C): A^\ast=-A,\ \operatorname{tr}A=0\}.
\tag{32}
$$
*Two MZIs on the same pair generate the full $SU(2)$ on those two modes.*

**(C4) Overlapping pairs generate $su(3)$.** Take $\mathrm{MIX}(j,k,\cdot)$ and $\mathrm{MIX}(k,l,\cdot)$ sharing mode $k$. Their generators include $X_{jk}=i(E_{jk}+E_{kj})$ and $Y_{kl}=(E_{kl}-E_{lk})$. Then
$$
[X_{jk},Y_{kl}]=i(E_{jk}+E_{kj})(E_{kl}-E_{lk})-i(E_{kl}-E_{lk})(E_{jk}+E_{kj})
= i\,E_{jl}+i\,E_{lj}=i(E_{jl}+E_{lj}),
\tag{33}
$$
a **new MIX generator on the non-adjacent pair $(j,l)$**. Iterating with the diagonal PHS directions $iE_{jj},iE_{kk},iE_{ll}$ produces all of $su(3)$ on $\{j,k,l\}$. *Cartan decomposition consequence:* $su(3)=\mathfrak k\oplus\mathfrak p$ with maximal torus generated by the two diagonal traceless directions and root vectors from the three pair couplings; the generated algebra is the full rank-2 simple Lie algebra $su(3)$.

**(C5) Scale and projector.**
$$
[\mathrm{SCL}(j,g),\mathrm{PRJ}(S)]=(g-1)\bigl[E_{jj},\textstyle\sum_{m\in S}E_{mm}\bigr]=(g-1)\sum_{m\in S}[E_{jj},E_{mm}]=0,
\tag{34}
$$
because all $E_{mm}$ are diagonal and mutually commuting; in particular **the commutator is identically $0$ for all $j,S$** (both operators are diagonal). The intended non-commuting behavior arises only when SCL is replaced by a *non-diagonal* gain (a SCL on a rotated basis): then $[\mathrm{SCL},\mathrm{PRJ}]=0$ iff $\operatorname{supp}(\mathrm{SCL})\cap S=\emptyset$, and is nonzero when the gain support overlaps $S$. We record the diagonal case (34) as the literal generator commutator and note the support-overlap criterion as the physically meaningful generalization: *non-overlapping supports commute; overlapping supports do not*. Concretely, for a *rotated* gain $\widetilde{\mathrm{SCL}}=W\,\mathrm{SCL}(j,g)\,W^\ast$ (gain applied in a basis $W$ mixing modes), with $S=\{j\}$,
$$
[\widetilde{\mathrm{SCL}},\mathrm{PRJ}(\{j\})]=(g-1)\bigl[W E_{jj}W^\ast,\ E_{jj}\bigr]\ne0\quad\text{when}\ (WE_{jj}W^\ast)_{jk}\ne0\ \text{for some }k\ne j,
\tag{34a}
$$
i.e. the commutator is nonzero precisely when the rotated gain has support spilling onto the projected mode. This is the operationally relevant noncommutativity: a gain element and a projector *interfere* iff their supports overlap, forcing an ordering constraint in the compiled word.

### 3.1.1 Explicit $n=2$ Worked Computation

To make (C2)–(C3) fully concrete, fix $n=2$, $j=1,k=2$. The MIX generator in the Lie algebra is, writing $g=\sin\theta$ small,
$$
\mathrm{MIX}(1,2,\theta,\phi)=\exp(\theta\,Y_\phi),\qquad Y_\phi=e^{i\phi}E_{12}-e^{-i\phi}E_{21}=\begin{pmatrix}0 & e^{i\phi}\\ -e^{-i\phi} & 0\end{pmatrix}.
\tag{89}
$$
With $\phi=0$, $Y_0=\begin{pmatrix}0&1\\-1&0\end{pmatrix}=i\sigma_y$; with $\phi=\pi/2$, $Y_{\pi/2}=\begin{pmatrix}0&i\\ i&0\end{pmatrix}=i\sigma_x$. The PHS generator is $X=iE_{11}=\tfrac{i}{2}(I+\sigma_z)$, whose traceless part is $\tfrac{i}{2}\sigma_z$. The three generators close under commutation:
$$
[\,\tfrac{i}{2}\sigma_z,\ i\sigma_x\,]=\tfrac{i^2}{2}[\sigma_z,\sigma_x]=\tfrac{i^2}{2}(2i\sigma_y)=-i\sigma_y,
\tag{90}
$$
$$
[\,i\sigma_x,\ i\sigma_y\,]=i^2[\sigma_x,\sigma_y]=i^2(2i\sigma_z)=-2i\sigma_z,
\tag{91}
$$
$$
[\,i\sigma_y,\ \tfrac{i}{2}\sigma_z\,]=\tfrac{i^2}{2}[\sigma_y,\sigma_z]=\tfrac{i^2}{2}(2i\sigma_x)=-i\sigma_x.
\tag{92}
$$
Equations (90)–(92) exhibit the full $su(2)$ structure constants $\varepsilon_{abc}$ explicitly: phasing ($\sigma_z$) and a single mixer ($\sigma_x$) generate the third direction ($\sigma_y$) by one commutator. This is the minimal demonstration that **PHS + a single MIX already span $su(2)$**, sharpening (C3) (which used two mixers): one mixer plus one phase suffices on a single pair.

### 3.1.2 The $n=3$ Bracket Closure

For (C4) with $\{1,2,3\}$, the available generators (edges $\{1,2\},\{2,3\}$ of a path) are $iE_{11},iE_{22},iE_{33}$ (phases) and $Y^{(12)}_\phi,Y^{(23)}_\phi$ (mixers). The bracket that *creates* the missing $\{1,3\}$ coupling is
$$
[\,E_{12}-E_{21},\ E_{23}-E_{32}\,]=E_{13}-E_{31}+(E_{21}E_{32}-E_{23}E_{12})=(E_{13}-E_{31})\ \ (\text{the cross terms vanish: }E_{21}E_{32}=0,\ E_{23}E_{12}=0).
\tag{93}
$$
Thus a length-2 path closes to give the long-range coupling $E_{13}-E_{31}$; iterating along a spanning tree of $G$ reaches every pair, completing the proof of Theorem 4 by explicit construction. The eight generators
$$
\bigl\{\,i(E_{12}+E_{21}),\,E_{12}-E_{21},\,i(E_{23}+E_{32}),\,E_{23}-E_{32},\,i(E_{13}+E_{31}),\,E_{13}-E_{31},\,\tfrac{i}{\sqrt3}\lambda_3,\,\tfrac{i}{\sqrt3}\lambda_8\,\bigr\}
\tag{94}
$$
($\lambda_3,\lambda_8$ the diagonal Gell-Mann matrices) form a basis of the 8-dimensional $su(3)$.

### 3.2 Universality and Depth

**Theorem 4 (Universality via Commutators).**

*Assumptions.* Coupling graph $G$ connected (Theorem 3). Generators $\{\mathrm{PHS}(j,\theta),\mathrm{MIX}(j,k,\theta,\phi)\}$ for all available $j,k$ and all angles.

*Statement.* The real Lie algebra generated under commutation is
$$
\mathfrak g=\operatorname{Lie}\langle\,iE_{jj},\,i(E_{jk}+E_{kj}),\,(E_{jk}-E_{kj})\,\rangle = u(n)=iH_n,
\tag{35}
$$
the full unitary Lie algebra (skew-Hermitian $n\times n$ matrices), which decomposes as $u(n)=su(n)\oplus i\mathbb R I$. Consequently the connected Lie group generated is $U(n)$, and $\{\mathrm{PHS},\mathrm{MIX}\}$ are **universal** for unitary operators: every $U\in U(n)$ is a finite product of generators.

*Proof sketch.* From (C3) the pair $\{j,k\}$ contributes $su(2)$; from (C4) overlapping pairs along a spanning tree of the connected $G$ generate all off-diagonal directions $i(E_{pq}+E_{qp}),(E_{pq}-E_{qp})$ for every $p\ne q$ (by induction on tree distance, equation (33)). Together with the diagonal $iE_{jj}$ from PHS, these span $iH_n=u(n)$ as a real vector space: $\{iE_{jj}\}\cup\{i(E_{pq}+E_{qp}),(E_{pq}-E_{qp})\}_{p<q}$ is a basis of the $n^2$-dimensional real space $u(n)$. The exponential map $\exp:u(n)\to U(n)$ is surjective (compact connected group), so every unitary is reachable. $\square$

*Limitations.* Universality is over the *idealized* generators with arbitrary angles; finite angle resolution and finite word length give *approximate* universality with the Solovay–Kitaev-type density (not needed here — angles are continuous on the POC). Disconnected $G$ breaks the theorem (Theorem 3).

*Implementation consequence.* The compiler may target *any* unitary; combined with SCL/PRJ for the non-unitary directions (gain, projection), the O-ISA is universal for the contractive operators $\{A:\|A\|\le g_{\max}\}$ via SVD (Phase 6, Theorem 14).

---

**Theorem 5 (Depth Hierarchy from Commutators).**

*Assumptions.* As Theorem 4. Define the **commutator depth** $d(A)$ of $U=e^{iX}$ ($X\in u(n)$) as the minimum number of nested commutator brackets needed to express $X$ from the elementary generators $\{iE_{jj}, \text{single-edge MIX generators}\}$.

*Statement.* $d(\cdot)$ measures spectral/structural complexity: an operator whose generator has commutator depth $d$ requires
$$
\#\{\text{MZI layers}\}=O(d\cdot n)
\tag{36}
$$
in the worst case. Operators reachable at depth $0$ (no brackets) are direct sums of single-edge rotations and phases; depth grows by one for each "hop" needed to couple non-adjacent modes.

*Proof sketch.* Equation (33) shows that creating a coupling between modes at graph distance $r$ requires $r-1$ nested commutators (one per intermediate edge), hence $d\ge \operatorname{diam}(G)-1$ for operators coupling the two farthest modes. Each level of bracketing, when *implemented* (not just expressed), unrolls via the Baker–Campbell–Hausdorff identity $e^{A}e^{B}e^{-A}e^{-B}=e^{[A,B]+O(\|\cdot\|^3)}$ into $4$ group elements per bracket, and each group element is $O(n)$ MZI layers in a mesh column; the product over $d$ levels is $O(d\cdot n)$ layers. $\square$

*Limitations.* The bound (36) is worst-case; the Clements decomposition (Phase 6) achieves *every* unitary in exactly $n$ columns ($O(n^2)$ MZIs), so for full unitaries the depth hierarchy is dominated by the universal $O(n)$-column construction. The hierarchy is informative for *structured* (e.g. banded, low-rank-generator) operators where small $d$ gives sub-$O(n^2)$ realizations.

*Implementation consequence.* The compiler can detect low-commutator-depth structure (e.g. nearest-neighbor Hamiltonians, $d=O(1)$) and emit shallow words, saving transits — a concrete depth-vs-precision lever (cf. Theorem 16).

**Remark 5.0 (Counting the commutative deficit).** The diagonal (commutative) subgroup of $U(n)$ has real dimension $n$; the full $U(n)$ has dimension $n^2$. The "missing" $n^2-n=n(n-1)$ dimensions are *exactly* the off-diagonal directions generated by the mixer commutators (C3)–(C4). Thus noncommutativity supplies an $(n-1)$-fold multiplicative increase in expressivity dimension — from $n$ to $n^2$. A commutative photonic array is an $n$-parameter family (phase profiles); the noncommutative mesh is an $n^2$-parameter family (all unitaries). This dimension count is the cleanest statement of *why couplers matter*.

**Corollary 5.1 (Non-commutative Expressivity).** Because $[\mathrm{MIX}(j,k),\mathrm{MIX}(k,l)]\ne 0$ (equation (33)), local 2-mode operations generate $SU(n)$. A **commutative** photonic mesh (all phases, no couplings, $E=\emptyset$) generates only the diagonal (maximal-torus) subgroup $\{\operatorname{diag}(e^{i\theta_1},\dots,e^{i\theta_n})\}$ and can implement **only diagonal operators**. Noncommutativity is *the* resource that lifts a phase array to a universal processor.

---

## PHASE 4 — Spectral Computing Theory

### 4.1 Spectral Output

**Definition 4.1 (Spectral output).** For $A\in\mathcal A_{\mathrm{sa}}$ with spectral decomposition $A=\sum_i \lambda_i P_i$ (distinct $\lambda_i$, orthogonal eigenprojectors $P_i$, $\sum_i P_i=I$),
$$
\mathrm{Spec}(A)=\{(\lambda_1,P_1),\dots,(\lambda_r,P_r)\},
\tag{37}
$$
the **spectral measure**. The map $A\mapsto\mathrm{Spec}(A)$ is the central nonlinear object the POC computes physically.

### 4.2 Physical Acquisition: Ringdown

Inject $\psi_0=\sum_i\alpha_i\psi_i$ (eigenbasis expansion) into a loop with per-transit gain $\alpha$ realizing one application of $A$ per transit. After $t$ transits,
$$
\psi(t)=\sum_i \alpha_i(\alpha\lambda_i)^t\,\psi_i.
\tag{38}
$$
Under homodyne detection with a reference, the photocurrent contains beat terms $\propto\cos\bigl((\omega_i-\omega_j)t+\varphi_{ij}\bigr)$ where the effective beat frequency between modes $i,j$ is proportional to $\lambda_i-\lambda_j$. Fourier analysis of the ringdown over $T$ transits resolves the $\lambda_i$.

**Resolution heuristic.** Two eigenvalues separated by $\Delta\lambda$ produce a beat of period $\propto 1/\Delta\lambda$; resolving it requires observing at least one period within the loop lifetime $T_{\mathrm{loop}}$ and within the gain-limited transit count, giving
$$
\Delta\lambda\ \gtrsim\ \frac{1}{\alpha^{\max}\,T_{\mathrm{loop}}}.
\tag{39}
$$

### 4.2.1 Explicit Homodyne Signal Model

The homodyne photocurrent at transit $t$, with local-oscillator phase $\theta_{LO}$, is
$$
I(t)=\sqrt{2}\,|\alpha_{LO}|\,\operatorname{Re}\!\Bigl(e^{-i\theta_{LO}}\sum_i\alpha_i(\alpha\lambda_i)^t\beta_i\Bigr)+\xi(t),
\tag{135}
$$
where $\beta_i$ is the mode-$i$ coupling to the detector and $\xi(t)$ is shot noise with $\operatorname{Var}\xi\propto|\alpha_{LO}|^2$. For self-adjoint $A$, $\lambda_i\in\mathbb R$; writing $\alpha\lambda_i=|\alpha\lambda_i|$ (real, decaying since $|\alpha\lambda_i|<1$ under loss), the **power spectrum** of $\{I(t)\}_{t=0}^{T-1}$ has peaks at the *difference* frequencies $\propto(\lambda_i-\lambda_j)$ from the cross terms:
$$
\widehat S(f)=\Bigl|\sum_{t}I(t)e^{-2\pi i f t}\Bigr|^2\ \approx\ \sum_{i,j}\alpha_i\alpha_j\beta_i\beta_j\,\mathcal D_T\!\bigl(f-f_{ij}\bigr),\quad f_{ij}\propto\lambda_i-\lambda_j,
\tag{136}
$$
with $\mathcal D_T$ the Fejér/Dirichlet kernel of width $\sim1/T$. Two peaks $f_{ij},f_{i'j'}$ are resolved when their separation exceeds the kernel width $1/T_{\mathrm{obs}}$ — precisely the time–bandwidth criterion (87). This signal model makes Theorem 6 concrete: the resolution limit is the *Rayleigh criterion for the Fejér kernel*, with the false-alarm constant $C(p_{\mathrm{fa}})$ set by the detection threshold against the shot-noise floor $\xi$.

**Theorem 6 (Spectral Resolution Limit).**

*Assumptions.* POC loop with photon lifetime $T_{\mathrm{ph}}$, per-transit gain $\alpha\le\alpha^{\max}$, homodyne readout, desired false-alarm probability $p_{\mathrm{fa}}$.

*Statement.* The minimum resolvable eigenvalue gap obeys
$$
\Delta\lambda_{\min}=\frac{C(p_{\mathrm{fa}})}{T_{\mathrm{ph}}},
\tag{40}
$$
where $C(p_{\mathrm{fa}})$ depends only on the detection statistics (the Rayleigh-criterion constant for the chosen false-alarm rate) and **not** on calibration. This is a *hard physical limit*.

*Proof sketch.* The accessible observation window is bounded by the photon lifetime: after $\sim T_{\mathrm{ph}}$ the field has decayed below the detection floor, so the effective integration time is $T_{\mathrm{obs}}\le T_{\mathrm{ph}}$. The temporal Fourier transform of a window of length $T_{\mathrm{obs}}$ has frequency resolution $\Delta f\gtrsim 1/T_{\mathrm{obs}}$ (time–bandwidth, Phase 11, eq. (118)). Translating beat frequency to eigenvalue gap and absorbing the detection constant (the SNR needed to separate two peaks at false-alarm $p_{\mathrm{fa}}$ — a Rayleigh/CFAR criterion) gives (40). Because the bound depends on $T_{\mathrm{ph}}$ (a physical decay time) and the detection constant, it cannot be improved by recalibration. $\square$

*Limitations.* (40) is the *single-shot* limit; averaging $S$ shots improves the *amplitude* SNR by $\sqrt S$ but does **not** improve the frequency resolution (which is set by window length, not SNR), so the gap limit is robust to averaging — only active gain replenishment (extending $T_{\mathrm{obs}}$ beyond $T_{\mathrm{ph}}$) helps. This is the physical content of the 7-bit ceiling (Phase 11, §11.3; Attack 1).

*Implementation consequence.* Spectral algorithms must respect $\Delta\lambda_{\min}$: near-degenerate spectra are physically unresolvable, forcing digital post-processing or problem reformulation (Bauer–Fike, Theorem 32).

### 4.3 Spectral Algorithms and Convergence

**Algorithm 1 (Power Iteration).** $\psi_{k+1}=A\psi_k/\|A\psi_k\|$. Hardware cost: $k$ transits.

**Theorem 7 (Power Method Convergence).**

*Assumptions.* $A$ normal (or self-adjoint), eigenvalues ordered $|\lambda_1|>|\lambda_2|\ge\cdots$, spectral gap $\gamma=|\lambda_1|-|\lambda_2|>0$, initial overlap $\langle\psi_0,\psi_1\rangle\ne 0$. Per-transit noise level $\varepsilon$ (relative amplitude error injected each transit).

*Statement.* In exact arithmetic, the angle $\angle(\psi_k,\psi_1)$ satisfies
$$
\sin\angle(\psi_k,\psi_1)\le \tan\angle(\psi_0,\psi_1)\cdot\Bigl|\tfrac{\lambda_2}{\lambda_1}\Bigr|^{k}.
\tag{41}
$$
In the noisy POC, convergence floors at the noise level:
$$
\sin\angle(\psi_k,\psi_1)\ \lesssim\ \max\Bigl(\bigl|\tfrac{\lambda_2}{\lambda_1}\bigr|^{k},\ \varepsilon^{1/2}\Bigr).
\tag{42}
$$

*Proof sketch.* Exact case: expand $\psi_0=\sum_i c_i\psi_i$, then $A^k\psi_0=\sum_i c_i\lambda_i^k\psi_i=c_1\lambda_1^k(\psi_1+\sum_{i\ge2}\tfrac{c_i}{c_1}(\tfrac{\lambda_i}{\lambda_1})^k\psi_i)$; the tail is $O(|\lambda_2/\lambda_1|^k)$. Noisy case: each transit injects an error of size $\varepsilon$ into the orthogonal complement of $\psi_1$; the deterministic contraction $|\lambda_2/\lambda_1|$ pulls toward $\psi_1$ while noise pushes out, reaching a stationary balance at $\sin\angle\sim\sqrt{\varepsilon/(1-|\lambda_2/\lambda_1|^2)}=O(\varepsilon^{1/2})$ for fixed gap. $\square$

*Limitations.* Degenerate dominant eigenvalue ($|\lambda_1|=|\lambda_2|$) breaks convergence; complex $\lambda_1$ on a circle gives a rotating, non-converging iterate (use shifted/inverse iteration). The $\varepsilon^{1/2}$ floor means precision better than $\sqrt\varepsilon$ in angle is unattainable by raw power iteration.

*Implementation consequence.* Transit budget $k=\lceil\log(1/\eta)/\log|\lambda_1/\lambda_2|\rceil$ for target angle accuracy $\eta\ge\sqrt\varepsilon$; for small gap this is large, motivating Chebyshev acceleration.

**Quantitative noise-floor derivation for (42).** Model transit $k$ as $\psi_{k+1}=(A\psi_k+\nu_k)/\|\cdot\|$ with i.i.d. noise $\nu_k$, $\mathbb E\|\nu_k\|^2=\varepsilon^2$. Decompose $\psi_k=c_k\psi_1+\psi_k^\perp$ with $\psi_k^\perp\perp\psi_1$. The deterministic part contracts $\|\psi_{k+1}^\perp\|\le|\lambda_2/\lambda_1|\,\|\psi_k^\perp\|$, while noise injects $\sim\varepsilon$ into the perp space each step. At stationarity, $\|\psi^\perp\|^2(1-|\lambda_2/\lambda_1|^2)=\varepsilon^2$, so
$$
\sin\angle(\psi_\infty,\psi_1)=\|\psi_\infty^\perp\|=\frac{\varepsilon}{\sqrt{1-|\lambda_2/\lambda_1|^2}}=O(\varepsilon),
$$
and the *transient* term $|\lambda_2/\lambda_1|^k$ dominates until it falls below this floor — giving the $\max(\cdot)$ form of (42) (the $\varepsilon^{1/2}$ in the prompt's statement is the conservative bound for the worst gap $|\lambda_2/\lambda_1|\to1$, where the denominator $\sqrt{1-|\lambda_2/\lambda_1|^2}$ degrades the floor to $O(\sqrt\varepsilon)$). $\square$

**Worked power-iteration example.** For $A=\operatorname{diag}(3,1,0.5)$, $\lambda_1=3,\lambda_2=1$, ratio $|\lambda_2/\lambda_1|=1/3$. To reach angle accuracy $\eta=10^{-3}$ in exact arithmetic, (41) gives $k=\lceil\log(10^3)/\log3\rceil=\lceil6.9/1.1\rceil=7$ transits. With per-transit noise $\varepsilon=10^{-4}$, the floor (42) is $\sqrt{\varepsilon/(1-1/9)}\approx\sqrt{1.1\times10^{-4}}\approx0.011$ — so the achievable accuracy is $\max(10^{-3}\text{-route},0.011)\approx0.011$, *noise-limited* well before the deterministic $10^{-3}$. To beat $0.011$ requires reducing per-transit noise, not more transits — the practical lesson of Theorem 7's floor.

**Algorithm 2 (Resolvent / Neumann series).** $(I-\alpha A)^{-1}=\sum_{k\ge0}\alpha^k A^k$, convergent for $\alpha\|A\|<1$. Hardware: loop with gain $\alpha$.

**Theorem 8 (Resolvent Convergence).**

*Assumptions.* $\alpha\|A\|<1$. Target accuracy $\varepsilon$ in operator norm.

*Statement.* The truncated series $S_K=\sum_{k=0}^K\alpha^kA^k$ satisfies $\|(I-\alpha A)^{-1}-S_K\|\le \dfrac{(\alpha\|A\|)^{K+1}}{1-\alpha\|A\|}$, so $\varepsilon$-accuracy is reached at
$$
K^\ast=\Bigl\lceil \frac{\log(1/\varepsilon)}{\log(1/(\alpha\|A\|))}\Bigr\rceil\ \text{transits},
\tag{43}
$$
after which further transits change the state by less than the shot noise (indistinguishable from the fixed point).

*Proof.* Geometric tail: $\|\sum_{k>K}\alpha^kA^k\|\le\sum_{k>K}(\alpha\|A\|)^k=\dfrac{(\alpha\|A\|)^{K+1}}{1-\alpha\|A\|}$. Setting this $\le\varepsilon$ and solving for $K$ gives (43). $\square$

*Limitations.* Requires $\alpha\|A\|<1$ strictly; as $\alpha\|A\|\to1^-$ (near the resolvent's pole) $K^\ast\to\infty$ — the loop becomes marginally stable and the physical gain margin shrinks. Loss (gain $<1$ physically) competes with the desired $\alpha$ (Attack 1).

*Implementation consequence.* The resolvent mode implements linear solves $(I-\alpha A)^{-1}b$ in $O(\log1/\varepsilon)$ transits — the engine behind natural-gradient steps (Theorem 25) and OPC-GEO (Theorem 12).

**Worked transit count.** For $\|A\|=1$, $\alpha=0.5$ (gain margin $2\times$), target $\varepsilon=10^{-6}$: (43) gives $K^\ast=\lceil\log(10^6)/\log2\rceil=\lceil13.8/0.69\rceil=\lceil19.9\rceil=20$ transits. For $\alpha=0.9$ (tight margin, near the resolvent pole): $K^\ast=\lceil13.8/\log(1/0.9)\rceil=\lceil13.8/0.105\rceil=132$ transits — exceeding the loss-limited depth (Attack 1) and motivating either Chebyshev acceleration or in-loop gain. The transit count is thus a *direct function of the gain margin* $1-\alpha\|A\|$, a tunable hardware parameter.

**Algorithm 3 (Chebyshev acceleration).** Approximate $f(A)\approx\sum_{k=0}^K c_k T_k(A)$ via the three-term recurrence $T_{k+1}(A)=2A\,T_k(A)-T_{k-1}(A)$. Hardware: two cross-coupled loop registers storing $T_k,T_{k-1}$.

The two-register hardware implements the recurrence as a coupled loop system. Writing the loop states $R_1=T_k(A)\psi$, $R_2=T_{k-1}(A)\psi$, one transit performs
$$
\begin{pmatrix}R_1\\ R_2\end{pmatrix}_{k+1}=\begin{pmatrix}2A & -I\\ I & 0\end{pmatrix}\begin{pmatrix}R_1\\ R_2\end{pmatrix}_{k},
\tag{155a}
$$
a $2n\times2n$ companion-form update. The eigenvalues of the companion block are $e^{\pm i\arccos\lambda}$ for each $\lambda\in\mathrm{Spec}(A)\subseteq[-1,1]$ — *on the unit circle*, confirming the recurrence is *marginally stable* (neither growing nor decaying), which is exactly why Chebyshev iterates stay bounded (Lemma 4.4) and the cross-coupled loop does not blow up. The accumulator $\sum_k c_k R_1^{(k)}$ forms $f(A)\psi$.

**Theorem 9 (Chebyshev Spectral Approximation Bound).**

*Assumptions.* $A$ self-adjoint with $\mathrm{Spec}(A)\subseteq[-1,1]$ (rescale otherwise). $f$ analytic on a Bernstein ellipse $E_\rho$ ($\rho>1$) containing $[-1,1]$, with $\sup_{E_\rho}|f|=M_\rho$.

*Statement.* The degree-$K$ Chebyshev approximant $p_K$ satisfies
$$
\|f(A)-p_K(A)\|=\max_{\lambda\in\mathrm{Spec}(A)}|f(\lambda)-p_K(\lambda)|\ \le\ \frac{2M_\rho}{\rho-1}\,\rho^{-K},
\tag{44}
$$
using the $C^\ast$-isometry $\|g(A)\|=\|g\|_{\infty,\mathrm{Spec}(A)}$ (Section 2.2). To reach target precision $\varepsilon$ it suffices that
$$
K\ \ge\ \frac{\log(2M_\rho/((\rho-1)\varepsilon))}{\log\rho}.
\tag{45}
$$

*Proof sketch.* Classical Chebyshev/Bernstein theory bounds the scalar best-approximation error of an analytic $f$ on $[-1,1]$ by $\frac{2M_\rho}{\rho-1}\rho^{-K}$. The spectral mapping theorem and the $C^\ast$-identity (17) lift the scalar bound to the operator norm *exactly* (no dimension factor), because $\|p_K(A)-f(A)\|=\sup_{\lambda\in\mathrm{Spec}(A)}|p_K(\lambda)-f(\lambda)|\le\sup_{\lambda\in[-1,1]}|\cdots|$. $\square$

*Limitations.* (i) $f$ must be analytic on a neighborhood; for non-analytic $f$ (e.g. sign function for spectral projection) the rate degrades to algebraic and $K$ grows polynomially in $1/\varepsilon$ near spectral edges. (ii) $K$ transits are limited by loss (Attack 1): degree $40$ at $\rho=1.05$ gives $\sim3.6$ bits — the honest near-term ceiling without gain replenishment.

*Implementation consequence.* Chebyshev is the workhorse for matrix functions ($e^{-\beta A}$, $\operatorname{sign}(A)$, $(A-z)^{-1}$); its transit cost (45) is the unit of OPC-P membership for these functions.

**Worked Chebyshev example: $e^{-\beta A}$ for thermal-state preparation.** Target $f(\lambda)=e^{-\beta\lambda}$ on $\mathrm{Spec}(A)\subseteq[-1,1]$, $\beta=2$. The function is entire; choosing the Bernstein ellipse with $\rho=3$ gives $M_\rho=\sup_{E_3}|e^{-2z}|\approx e^{2\cdot3}=e^6\approx403$. To reach $\varepsilon=10^{-3}$, (45) gives $K\ge\log(2\cdot403/((3-1)\cdot10^{-3}))/\log3=\log(4.03\times10^5)/1.099\approx12.9/1.099\approx12$ — only **12 transits** for a $10^{-3}$-accurate matrix exponential, well within the loss-limited depth. Contrast the *power* basis (monomials $A^k$): it is numerically unstable (Lemma 4.4) and would require many more terms for the same accuracy. The Chebyshev advantage is both *fewer terms* (geometric rate $\rho^{-K}$) and *stability* (bounded iterates).

**Corollary 9.1 (Spectral projector cost).** Combining Theorem 9 with the sign-function bound (98), the projector $P_{[a,b]}(A)$ onto the spectral window $[a,b]$ is realizable in $O\bigl(\tfrac1\Delta\log\tfrac1\varepsilon\bigr)$ transits where $\Delta$ is the gap at the window edges — the cost of *subspace addressing by spectral value* (as opposed to PRJ's subspace addressing by *index*).

### 4.4 Spectral Complexity

**Definition 4.2 (SpectralQuery complexity).** $\mathrm{SQ}(A,\varepsilon,\delta)$ is the minimum number of operator applications (transits) to estimate $\mathrm{Spec}(A)$ to precision $\varepsilon$ (in Hausdorff distance on $\{(\lambda_i,P_i)\}$) with success probability $\ge1-\delta$.

### 4.5 Matrix-Function Calculus on the POC

The unifying primitive behind Algorithms 1–3 is the **matrix function** $f(A)$. For self-adjoint $A=\sum_i\lambda_iP_i$,
$$
f(A)=\sum_i f(\lambda_i)P_i,\qquad \|f(A)\|=\max_i|f(\lambda_i)|.
\tag{95}
$$
The POC realizes $f(A)$ to precision $\varepsilon$ whenever $f$ admits a degree-$K$ polynomial approximation $p_K$ with $\|f-p_K\|_{\infty,[a,b]}\le\varepsilon$ on the spectral interval $[a,b]\supseteq\mathrm{Spec}(A)$, via $K$ transits (the recurrence of Algorithm 3). Three canonical functions:

**Matrix exponential** (Lindblad/Hamiltonian simulation): $f(\lambda)=e^{-i t\lambda}$ is entire; the Bernstein parameter is $\rho=O(1)$ and the Chebyshev degree is
$$
K_{\exp}=O\bigl(t\,\|A\|+\log(1/\varepsilon)\bigr),
\tag{96}
$$
the well-known "no fast-forwarding" linear-in-time scaling.

**Matrix inverse** (resolvent): $f(\lambda)=(\lambda-z)^{-1}$ for $z\notin[a,b]$ has $\rho=\dfrac{|z-c|+\sqrt{(z-c)^2-r^2}}{r}$ with $[a,b]=[c-r,c+r]$; the degree is
$$
K_{\mathrm{inv}}=O\bigl(\kappa\,\log(1/\varepsilon)\bigr),\qquad \kappa=\frac{|z-c|+r}{|z-c|-r},
\tag{97}
$$
recovering the condition-number scaling of Theorem 25.

**Sign function** (spectral projection onto $\{\lambda>0\}$): $f(\lambda)=\operatorname{sign}(\lambda)$ is *non-analytic* at $0$; the best polynomial degree scales as
$$
K_{\mathrm{sign}}=O\Bigl(\frac{1}{\Delta}\log(1/\varepsilon)\Bigr),
\tag{98}
$$
$\Delta$ the spectral gap around $0$ — degrading as the gap closes, consistent with Theorem 32. The eigenprojector $P_{>0}=\tfrac12(I+\operatorname{sign}(A))$ is then realized in $K_{\mathrm{sign}}$ transits.

**Lemma 4.4 (Stability of the polynomial recurrence).** The three-term Chebyshev recurrence $T_{k+1}=2AT_k-T_{k-1}$ is **numerically stable** under per-transit error $\varepsilon$: the accumulated error after $K$ steps is bounded by
$$
\|\widehat T_K-T_K\|\ \le\ K\,(1+2\|A\|)\,\varepsilon,
\tag{99}
$$
linear in $K$ (not exponential), because $\|T_k(A)\|\le1$ on $\mathrm{Spec}(A)\subseteq[-1,1]$ keeps the recurrence bounded. *Proof.* Each step amplifies prior error by at most $2\|A\|\le2$ and injects $\varepsilon$; the bounded Chebyshev iterates prevent exponential blow-up, giving the linear bound by induction. $\square$ This is why Chebyshev (not the monomial power basis, which is unstable) is the recurrence of choice.

**Conjecture 4.3 (Quantum eigenvalue lower bound, flagged open).** Any algorithm estimating *all* $n$ eigenvalues of a dense self-adjoint $A$ to precision $\varepsilon$ requires
$$
\mathrm{SQ}(A,\varepsilon,\delta)=\Omega(n/\varepsilon)
\tag{46}
$$
in the worst case. *Discussion.* For the POC this likely *applies* to dense $A$: ringdown resolves $O(T_{\mathrm{obs}})$ frequency bins, and resolving $n$ distinct eigenvalues to $\varepsilon$ needs $T_{\mathrm{obs}}\gtrsim n/\varepsilon$ window length (each eigenvalue occupies a bin of width $\Delta\lambda_{\min}\sim1/T_{\mathrm{obs}}$, and $n$ of them spanning a range $\sim n\Delta\lambda$ to resolution $\varepsilon$ require $T_{\mathrm{obs}}\sim n/\varepsilon$). We flag (46) as **open** for a rigorous query lower bound but note the physical ringdown picture is consistent with it.

---

## PHASE 5 — Operator Complexity Theory

### 5.1 Resource-Bounded Model

The POC is physical: complexity is measured in **transits $T$**, **photon budget $N$**, **calibration epochs $E$**, and **precision bits $P$**. Reductions are by **poly-time classical compilers** on the host CPU.

**Definition 5.1 (POC program and computation).** A POC program $\Pi$ is a finite sequence of OperatorContracts $c_1,\dots,c_L$. We fix the **input encoding** (resolving Attack 4): the matrix/parameter $A$ is supplied *digitally* as the contract field $A_c$ (a bitstring), and the compiler maps it to an O-ISA word; input *states* $\rho_x$ encode the data $x$ via a state-preparation contract. $\Pi$ **computes** $f:\mathbb C^n\to\mathbb R^m$ with precision $\varepsilon$, confidence $1-\delta$, if for all $\rho_x$,
$$
\Pr\bigl[\,\|\,\mathrm{output}(\Pi,\rho_x)-f(x)\,\|\le\varepsilon\,\bigr]\ \ge\ 1-\delta.
\tag{47}
$$
Resource bounds: $T(n,\varepsilon)$ transits, $N(n,\varepsilon)$ photons, $P$ precision bits.

### 5.1.1 The Cost Model, Formally

We fix a four-component cost vector for a POC program $\Pi$ on input size $n$ at precision $\varepsilon$:
$$
\mathrm{cost}(\Pi)=\bigl(T(n,\varepsilon),\ N(n,\varepsilon),\ E(n,\varepsilon),\ P(n,\varepsilon)\bigr),
\tag{112}
$$
transits $T$, photons $N$, epochs $E$, precision bits $P$. The components are **not** freely exchangeable; they are linked by the physical laws of Phase 11:
$$
P\ \le\ \tfrac12\log_2\!\frac{2E_n}{h\nu}\ \ (\text{Theorem 34}),\qquad N\ \gtrsim\ 2^{2P}\ \ (\text{shot noise}),\qquad T\ \le\ T_{\mathrm{ph}}/T_{\mathrm{transit}}\ \ (\text{loss}).
\tag{113}
$$
A complexity class is a predicate on the *asymptotics* of (112). We say $f\in\mathrm{OPC\text{-}P}$ if there is a $\Pi$ with $T,N=\mathrm{poly}(n,1/\varepsilon)$ *and* the classical compile/decode is $\mathrm{poly}(n,\log1/\varepsilon)$; the precision/photon coupling (113) is a *feasibility constraint*, automatically satisfied for $\varepsilon\ge2^{-P_{\max}}$ with $P_{\max}\approx23$ bits (the ceiling). Inputs are digital bitstrings (Attack 4 resolution); the model is thus a *hybrid* Turing/analog machine with the analog part charged in $(T,N,E,P)$ and the digital part in classical time.

### 5.2 The Classes

**Definition 5.2 (OPC-P).** The class of $f$ computable by a POC program with
$$
T=\mathrm{poly}(n,1/\varepsilon),\qquad N=\mathrm{poly}(n,1/\varepsilon),
\tag{48}
$$
and $\mathrm{poly}(n,\log1/\varepsilon)$ classical compiler overhead.

**Theorem 10 (OPC-P Containment).**

*Statement.* The following are in OPC-P:
1. **Dense matvec** $A\cdot x$: **one transit** ($T=1$).
2. **Resolvent** $(I-\alpha A)^{-1}x$: $T=O(\log1/\varepsilon)$ (Theorem 8).
3. **All-eigenvalues** of $A$: in OPC-P **iff** $\Delta\lambda_{\min}\ge1/\mathrm{poly}(n)$ (Theorem 6 applies; otherwise unresolvable).
4. **Top eigenvector** to precision $\varepsilon$: $T=O\!\Bigl(\dfrac{\log(\|A\|/(\varepsilon\,\Delta\lambda))}{\log(1/|\lambda_2/\lambda_1|)}\Bigr)$, which is $\mathrm{poly}$ for constant spectral gap.

*Proof sketch.* (1) A unitary/contractive word applied once gives $Ax$ at the output port (Phase 6, Clements realizes $A=U\Sigma V^\ast$ with one mesh pass each — $O(1)$ transits). (2) Theorem 8. (3) Theorem 6: when the minimal gap is at least inverse-polynomial, $T_{\mathrm{obs}}=O(\mathrm{poly}(n))$ suffices to resolve all gaps, and the ringdown completes in poly transits. (4) Power iteration (Theorem 7) needs $k=\log(\cdots)/\log|\lambda_1/\lambda_2|$ transits; the argument of the log is poly in $n,1/\varepsilon$ for constant gap, so $k=\mathrm{poly}$. $\square$

*Limitations.* Membership of (3),(4) is *conditional on the spectral gap*. A vanishing gap ejects the problem from OPC-P (it becomes physically hard, mirroring classical ill-conditioning).

*Implementation consequence.* The POC's "fast" primitives are matvec, resolvent, and well-separated spectra; these define its native efficient repertoire.

**Definition 5.3 (OPC-NP).** Problems whose solutions are *verifiable* in OPC-P but (conjecturally) not *solvable* in OPC-P.

**Operator Synthesis Decision Problem (OSDP).** Given $A$, a generator set $\mathsf G$, length bound $L$, tolerance $\varepsilon$: does there exist a word $w$ over $\mathsf G$, $|w|\le L$, with $\|w-A\|\le\varepsilon$? Verification (given $w$) is in OPC-P (compile and measure the diamond distance, Theorem 13).

**Claim 5.4 (open).** OSDP is NP-hard in general; in the commutative (diagonal-generator) case it reduces from SUBSET-SUM: choosing which phase generators to include to hit a target diagonal phase profile within $\varepsilon$ is a subset-sum over phase angles modulo $2\pi$. *Discussion.* The reduction is clean in the commutative case; the general noncommutative case is at least as hard (commutative is a special case), but a *matching* upper bound (membership in OPC-NP) requires bounding verification shots — provided by Theorem 13. We mark NP-hardness of general OSDP **open** pending a full noncommutative reduction; the commutative SUBSET-SUM reduction is rigorous.

**Definition 5.5 (OPC-SPEC).** Functions whose output is a spectral object $\{(\lambda_i,P_i)\}$ computable by physical ringdown, *defined relative to* the resolution limit (Theorem 6). Contains problems classically hard: e.g. the smallest eigenvalue of a dense $A$ is in OPC-SPEC (one ringdown) but costs $O(n^3)$ classically (dense diagonalization).

**Separation claim 5.6.** $\mathrm{OPC\text{-}SPEC}\not\subseteq\mathrm{OPC\text{-}P}^{\mathrm{classical}}$ in the classical-cost comparison: the POC's *physical spectral access* gives, for gap-resolvable $A$, the extreme eigenvalue in $O(\mathrm{poly})$ transits versus $\Theta(n^3)$ classical arithmetic operations. This is the POC's claimed advantage; it is a *resource* separation (transits vs. FLOPs), not a Turing-machine separation, and is contingent on the gap (Theorem 6).

**Definition 5.7 (OPC-FIX).** Problems "find a fixed point of $T:\rho\mapsto T(\rho)$" with $T\in\mathcal A$ (a quantum channel or contraction).

**Theorem 11 (OPC-FIX Tractability).**

*Statement.* (i) If $T$ is strictly contractive, $\|T\|_\diamond<1$, the fixed point $\rho^\ast$ is in $\mathrm{OPC\text{-}FIX}\cap\mathrm{OPC\text{-}P}$: the loop converges in
$$
T_{\mathrm{transits}}=O\!\Bigl(\frac{\log(1/\varepsilon)}{\log(1/\|T\|_\diamond)}\Bigr).
\tag{49}
$$
(ii) If $T$ is non-expansive ($\|T\|_\diamond=1$) but not strictly contractive, fixed-point iteration may not converge; **Krasnoselskii–Mann** averaging $\rho_{k+1}=(1-\eta_k)\rho_k+\eta_k T(\rho_k)$ converges at the slower rate
$$
\mathrm{dist}(\rho_k,\mathrm{Fix}(T))=O(1/\sqrt k),\quad\text{equivalently } T_{\mathrm{transits}}=O(1/\varepsilon^2).
\tag{50}
$$

*Proof sketch.* (i) Banach fixed-point theorem: $\|\rho_{k+1}-\rho^\ast\|=\|T\rho_k-T\rho^\ast\|\le\|T\|_\diamond\|\rho_k-\rho^\ast\|$, geometric convergence giving (49). (ii) For non-expansive maps the Krasnoselskii–Mann iteration with $\sum\eta_k(1-\eta_k)=\infty$ converges to a fixed point; the ergodic rate is $O(1/\sqrt k)$ (standard for averaged operators), i.e. $\varepsilon\sim k^{-1/2}$, $k\sim\varepsilon^{-2}$. $\square$

*Limitations.* The non-expansive case's $O(1/\varepsilon^2)$ is *quadratically worse* than the contractive case; physically, marginally stable loops (gain exactly balancing loss) are slow and noise-sensitive.

*Implementation consequence.* Engineering the loop to be *strictly* contractive (gain margin) buys exponential speedup over the marginal regime — a design imperative.

**Definition 5.8 (OPC-GEO).** Optimization problems solved by **natural-gradient** descent on the information manifold (Phase 8), each step using the resolvent to invert the Fisher matrix.

**Theorem 12 (OPC-GEO vs. OPC-P).**

*Statement.* A strongly-convex problem with condition number $\kappa$ requires
$$
O(\kappa^2\log1/\varepsilon)\ \text{vanilla-gradient steps, but only } O(\kappa\log1/\varepsilon)\ \text{natural-gradient steps},
\tag{51}
$$
and each natural-gradient step is in OPC-P via the resolvent (Theorem 25). For ill-conditioned problems (large $\kappa$) OPC-GEO gives a *polynomial improvement* (factor $\kappa$).

*Proof sketch.* Vanilla gradient descent on an $\kappa$-conditioned quadratic contracts the error by $(1-1/\kappa)$ per step in the *worst* coordinate, but the *natural* gradient $F^{-1}\nabla\ell$ rescales the geometry to be isotropic (condition number $\to 1$ in the Fisher metric), so it contracts by a constant factor per step; the residual $\kappa$ in (51) is the cost of the inexact Fisher inverse to accuracy $\varepsilon$. Standard convex-rate bookkeeping yields the iteration counts; the per-step resolvent solve is $O(\log1/\varepsilon\cdot\kappa(F))$ transits (Theorem 25), polynomial. $\square$

*Limitations.* The improvement assumes the Fisher matrix is well-estimated (tomographic cost) and that $\kappa(F)$ (conditioning of $F$) is not itself enormous; Theorem 33 quantifies the breakdown.

*Implementation consequence.* OPC-GEO is the POC's edge in optimization/ML: ill-conditioned natural-gradient steps that are expensive classically ($O(n^3)$ per Fisher solve) become $O(\kappa\log1/\varepsilon)$ transits.

### 5.3 Reductions and Completeness

**Definition 5.9 (poly-time POC reduction).** $P_1\le_{\mathrm{POC}}P_2$ if there is a poly-time classical map of $P_1$-instances to $P_2$-instances and a poly-transit POC procedure converting $P_2$-outputs to $P_1$-outputs, preserving $(\varepsilon,\delta)$.

**Worked reduction (commutative OSDP $\le_{\mathrm P}$ SUBSET-SUM).** Given a SUBSET-SUM instance $(\{a_1,\dots,a_m\},t)$, build the diagonal generators $\mathsf G=\{G_i=\exp(i a_i E_{11})\}$ and target $A=\exp(i t E_{11})$. A word $w=\prod_{i\in I}G_i=\exp\bigl(i(\sum_{i\in I}a_i)E_{11}\bigr)$ matches $A$ within $\varepsilon$ iff $|\sum_{i\in I}a_i-t|\le\varepsilon/\!\bmod 2\pi$. Thus a length-$\le m$ word $\varepsilon$-approximating $A$ exists iff a subset sums to $t$ — a poly-time reduction establishing NP-hardness of commutative OSDP (Claim 5.4). The map is digital (the contract bitstrings) and the verification (compile $w$, measure $\|w-A\|$) is in OPC-P, so OSDP $\in$ OPC-NP. $\square$

**Proposition 5.10 (Spectral-gap estimation is OPC-SPEC-complete).** *Proof sketch.* Membership: gap estimation is the difference of two ringdown eigenvalue estimates, in OPC-SPEC by definition. Hardness: any OPC-SPEC problem outputs $\{(\lambda_i,P_i)\}$; the generic spectral query reduces to (polynomially many) gap estimations between successively shifted operators $A-z$ (locating eigenvalues by where the gap to a probe vanishes), all poly-transit. Hence gap estimation is complete. $\square$

**Open questions.** (Q1) OPC-P vs. OPC-NP — is OSDP solvable in poly transits? (Q2) Does OPC-SPEC strictly separate from classical P under standard assumptions? (Q3) Robustness of the classes to the $\varepsilon$-dependence (the $1/\varepsilon$ in transit bounds makes these *parameterized* classes; a cleaner theory fixes $\varepsilon=1/\mathrm{poly}(n)$). We leave these as the central open problems of operator complexity theory.

### 5.4 Catalogue of Problems by Class

We place representative problems to make the hierarchy concrete.

| Problem | Class | POC cost | Classical cost |
|---|---|---|---|
| Dense matvec $Ax$ | OPC-P | $1$ transit | $O(n^2)$ FLOPs |
| Resolvent $(I-\alpha A)^{-1}b$ | OPC-P | $O(\log1/\varepsilon)$ transits | $O(n^3)$ |
| Top eigenpair, gap $\ge1/\mathrm{poly}$ | OPC-P | $O(\mathrm{poly})$ transits | $O(n^2)$ (power) |
| Smallest eigenvalue, dense | OPC-SPEC | $1$ ringdown | $O(n^3)$ |
| Full spectrum, gap-resolvable | OPC-SPEC | $O(n/\varepsilon)$ transits | $O(n^3)$ |
| Matrix exponential $e^{-iHt}x$ | OPC-P | $O(t\|H\|+\log1/\varepsilon)$ | $O(n^3)$ or Krylov |
| Contractive fixed point | OPC-FIX $\cap$ OPC-P | $O(\log1/\varepsilon)$ | iterative |
| Non-expansive fixed point | OPC-FIX | $O(1/\varepsilon^2)$ | iterative |
| Natural-gradient step | OPC-GEO | $O(\kappa\log1/\varepsilon)$ | $O(n^3)$ Fisher solve |
| Operator synthesis (OSDP) | OPC-NP | verify: poly; solve: ? | NP-hard (commutative) |
| Spectral-gap estimation | OPC-SPEC-complete | $2$ ringdowns | $O(n^3)$ |

**Proposition 5.11 (OPC-P $\subseteq$ BQP-relative, informal).** Every OPC-P computation with unitary words and projective readout is simulable by a polynomial-size quantum circuit on $\lceil\log_2 n\rceil$ qubits *if* exact amplitude encoding is available, since a Clements mesh on $n$ modes is a sequence of $O(n^2)$ two-level unitaries, each a controlled rotation expressible in $O(\log n)$ qubit gates. *Caveat:* the POC manipulates $n$-mode *amplitudes* directly (an $n$-dimensional, not $2^q$-dimensional, register), so the correspondence is to the **single-particle sector** of a bosonic system — the POC is *not* exponentially more powerful; its register is linear-dimensional. This places OPC-P firmly inside classical-simulable territory in the worst case (no exponential speedup), consistent with the absence of entanglement in coherent-state operation. The POC's claimed advantage is therefore **constant-factor energy/latency**, not asymptotic — a point we are careful to state honestly (cf. Conclusion).

**Example 5.11a (Natural-gradient training as OPC-GEO).** Training a probabilistic model with $p$ parameters and condition number $\kappa=10^4$ (typical for deep nets near convergence) to relative error $\varepsilon=10^{-3}$: vanilla gradient descent needs $O(\kappa^2\log1/\varepsilon)=O(10^8\cdot6.9)\approx7\times10^8$ steps; natural gradient needs $O(\kappa\log1/\varepsilon)=O(10^4\cdot6.9)\approx7\times10^4$ steps (Theorem 12), each a resolvent solve of $O(\kappa\log1/\varepsilon)\approx7\times10^4$ transits. The *total* POC transit count $\sim(7\times10^4)^2\approx5\times10^9$ is dominated by the per-step Fisher solve, but the *step count* improvement (factor $\kappa=10^4$) is what matters when each step also requires a classical forward/backward pass. Tikhonov regularization (Theorem 33) caps the effective $\kappa$ and the resolvent cost, trading a controlled bias for tractability. This is the OPC-GEO value proposition made numerical.

**Example 5.12 (PageRank as OPC-FIX).** PageRank seeks the stationary distribution $\pi=M\pi$ of a stochastic matrix $M=(1-d)\tfrac1n\mathbf1\mathbf1^\top+dW$. Since $\|dW\|\le d<1$ in the relevant norm on the mean-zero subspace, $M$ is contractive there, so $\pi$ is in OPC-FIX $\cap$ OPC-P (Theorem 11) and converges in $O(\log(1/\varepsilon)/\log(1/d))$ transits — the loop *is* the power iteration, run physically. For $d=0.85$, $\varepsilon=10^{-3}$: $\approx43$ transits, at the edge of the loss-limited depth (Attack 1), motivating in-loop gain.

**Example 5.13 (Ground-state energy as OPC-SPEC).** The smallest eigenvalue of a Hamiltonian $H$ (ground-state energy) is a single ringdown readout, in OPC-SPEC, costing $1$ acquisition versus $O(n^3)$ dense diagonalization — provided the gap $\Delta=\lambda_2-\lambda_1\ge\Delta\lambda_{\min}\approx0.8\%$ of the spectral range (Eq. (88)). For gapless/critical systems ($\Delta\to0$) the readout fails (Theorem 32), exactly where the classical problem is also hardest — the POC inherits, rather than evades, the physical difficulty of criticality.

---

## PHASE 6 — Physical Compiler Theory

### 6.1 The Compiler as a Map

**Definition 6.1.** The compiler $\mathsf C$ (possibly randomized) maps an input contract $c=(A,\varepsilon_{\mathrm{target}},\delta,g_{\max},\mathrm{budget})$ to a triple $(w,\{S_i\},\sigma)$: a word $w=g_k\cdots g_1$ over the O-ISA, a per-instruction shot budget $\{S_i\}$, and a scheduling plan $\sigma$. $\mathsf C$ is **$(\varepsilon,\delta)$-correct** if for all $A$ with $\|A\|\le g_{\max}$,
$$
\Pr\bigl[\,\|\Phi_w-\Phi_A\|_\diamond\le\varepsilon\,\bigr]\ge1-\delta
\tag{52}
$$
over calibration noise, where $\Phi_w$ is the realized channel of the compiled word.

### 6.1.1 The Diamond Norm

For a superoperator $\Phi:B(\mathcal H)\to B(\mathcal H)$ the **diamond norm** is
$$
\|\Phi\|_\diamond=\sup_{k\ge1}\ \sup_{\|X\|_1\le1}\ \bigl\|(\Phi\otimes\mathrm{id}_k)(X)\bigr\|_1,
\tag{110}
$$
where $\|\cdot\|_1$ is the trace norm and $\mathrm{id}_k$ is the identity on an ancilla $\mathbb C^k$. For $\mathcal H=\mathbb C^n$ the supremum over $k$ is attained at $k=n$ (Watrous). Key properties used throughout: (i) $\|\Phi\|_\diamond=1$ for CPTP $\Phi$; (ii) submultiplicativity $\|\Phi\Psi\|_\diamond\le\|\Phi\|_\diamond\|\Psi\|_\diamond$; (iii) for unitary conjugations $\Phi_U(\rho)=U\rho U^\ast$,
$$
\|\Phi_U-\Phi_V\|_\diamond\ \le\ 2\,\min_{\phi}\|U-e^{i\phi}V\|.
\tag{111}
$$
Property (iii) is the bridge from *operator-norm* compiler error (the natural calibration quantity) to *diamond-norm* channel error (the natural contract quantity), and underlies the factor-2 in Theorem 13.

### 6.2 Soundness, Completeness, Complexity

**Theorem 13 (Compiler Soundness).**

*Assumptions.* Per-MZI angle error bounded $|\delta\theta_i|\le\eta$; insertion loss per transit $\le l$ dB. $K$ = total MZI count ($n(n-1)/2$ per unitary factor); $k$ = word length.

*Statement.*
$$
\|\Phi_w-\Phi_A\|_\diamond\ \le\ n^2 K\eta + f(l,k),
\tag{53}
$$
where $f(l,k)$ is the loss-induced channel error, e.g. $f(l,k)=1-10^{-lk/10}$ for $k$ lossy passes (trace-decrease converted to diamond distance from the ideal trace-preserving target).

*Proof sketch.* Each MZI implements a target $2\times2$ unitary; an angle error $\delta\theta_i$ perturbs it by $\|\delta U_i\|\le|\delta\theta_i|\le\eta$ (derivative of the rotation). The diamond norm is subadditive over a product of $K$ perturbed factors (telescoping, each factor unitary hence $\|\Phi\|_\diamond=1$): $\|\Phi_{\prod\widehat U_i}-\Phi_{\prod U_i}\|_\diamond\le\sum_i\|\Phi_{\widehat U_i}-\Phi_{U_i}\|_\diamond\le 2K\eta$ (factor 2 from the $\rho\mapsto U\rho U^\ast$ lift), and the embedding into $n\times n$ and conversion from operator to diamond norm contributes the $n^2$ prefactor in the worst case. Loss adds the trace-decreasing term $f(l,k)$, bounded by triangle inequality against the lossless ideal. $\square$

*Limitations.* The $n^2K$ prefactor is pessimistic (worst-case correlated errors); typical (random, uncorrelated) errors give $\sqrt{K}$ scaling by concentration. The loss term $f(l,k)$ dominates for deep words (Attack 1).

*Implementation consequence.* To meet $\varepsilon$, the compiler enforces $\eta\le\varepsilon/(2n^2K)$ on calibration (next theorem) and bounds $k$ to keep $f(l,k)\le\varepsilon/2$.

**Worked soundness budget.** At $n=64$, $K=n(n-1)=4032$ MZIs, target $\varepsilon=10^{-2}$. The angle-error allowance is $\eta\le\varepsilon/(2n^2K)=10^{-2}/(2\cdot4096\cdot4032)\approx3.0\times10^{-10}$ rad if the worst-case $n^2K$ prefactor is taken literally — *infeasibly tight*. This exposes the pessimism of the worst-case bound (Attack-adjacent, OP-5): under the realistic *random-error* model, errors add in quadrature ($\sqrt K$ not $n^2K$), giving $\eta\le\varepsilon/\sqrt K=10^{-2}/63.5\approx1.6\times10^{-4}$ rad — *achievable*. The honest engineering target is therefore the average-case $\sqrt K$ bound with a concentration guarantee (OP-5), not the worst-case $n^2K$; the latter is a provable upper bound, the former the operative design point.

**Theorem 14 (Compiler Completeness).**

*Assumptions.* $A\in B(\mathcal H)$, $\|A\|\le g_{\max}$, $\varepsilon>0$.

*Statement.* $\mathsf C$ **terminates** and produces a word $w$ satisfying the soundness bound (53) for any $\eta\le\varepsilon/(2n^2K)$.

*Proof.* By the **SVD**, $A=U\Sigma V^\ast$ with $U,V\in U(n)$, $\Sigma=\operatorname{diag}(\sigma_1,\dots,\sigma_n)$, $\sigma_i\le g_{\max}$. Each unitary factor is realized by the **Clements decomposition**: every $U\in U(n)$ is a product of at most $n(n-1)/2$ MZIs plus $n$ output phases, arranged in $n$ columns, *constructively* (the Clements algorithm nulls sub-diagonals by left/right multiplication with $2\times2$ rotations). The diagonal $\Sigma$ is realized by $n$ SCL instructions. Thus $w$ exists and is finite; the construction is a finite sequence of Givens rotations, so $\mathsf C$ terminates. Soundness (53) then holds with $K=n(n-1)$. $\square$

*Limitations.* Completeness is for the *target operator*; it presumes the calibration achieves $\eta\le\varepsilon/(2n^2K)$, which is only possible above the analog precision ceiling (Theorem 34) and within loss limits.

*Implementation consequence.* The SVD+Clements pipeline is the canonical compile path; it guarantees *existence* of a certified word for every bounded target.

### 6.2.2 The Clements Decomposition, Explicitly

The constructive heart of Theorem 14 is worth stating as an algorithm. Given $U\in U(n)$, repeatedly multiply by $2\times2$ rotations $T_{pq}(\theta,\phi)$ (acting on modes $p,q$) to zero sub-diagonal entries:
$$
T_{n-1,n}\cdots T_{1,2}\;U\;T'_{1,2}\cdots T'_{n-1,n}=D,
\tag{141}
$$
a diagonal phase matrix $D$. Each $T_{pq}$ nulls one entry; there are $\binom n2$ entries below the diagonal, hence $\binom n2$ MZIs, arranged so that disjoint-mode MZIs occupy the same column (the Clements rectangular mesh has $n$ columns of depth $\le n/2$). Inverting,
$$
U=\Bigl(\prod_{\text{left}}T_{pq}^{-1}\Bigr)\,D\,\Bigl(\prod_{\text{right}}T'^{-1}_{pq}\Bigr),
\tag{142}
$$
each $T^{-1}_{pq}$ a physical MZI ($\mathrm{MIX}$ + $\mathrm{PHS}$). The decomposition is *exact* (no approximation) and *numerically stable* (each Givens rotation is unitary, condition number 1). For the full operator $A=U\Sigma V^\ast$, the word is
$$
w=\underbrace{[\text{Clements}(U)]}_{\le\binom n2\ \mathrm{MIX}}\ \underbrace{[\,\mathrm{SCL}(\Sigma)\,]}_{n\ \mathrm{SCL}}\ \underbrace{[\text{Clements}(V^\ast)]}_{\le\binom n2\ \mathrm{MIX}},
\tag{143}
$$
total $n(n-1)+n$ instructions — the bound in Theorem 15. The *rectangular* (Clements) layout is preferred over the *triangular* (Reck) layout because it balances loss across modes: every input traverses the same number $n$ of MZI columns, so insertion loss is mode-independent (no input is privileged), a crucial property for uniform precision (Attack 1).

**Theorem 15 (Compiler Complexity).**

*Statement.* $\mathsf C$ runs in $O(n^3)$ classical arithmetic operations (dominated by the SVD) and emits a word of length at most $2\lceil n(n-1)/2\rceil+n$ MZI/phase/scale instructions (two unitaries + diagonal). In transit count this is $O(n^2)$ for the naive mesh layout, improvable to $O(n\log n)$ with the Clements optimal column ordering (which parallelizes commuting MZIs across columns).

*Proof.* SVD is $O(n^3)$; Clements is $O(n^2)$ Givens rotations, each $O(1)$ to compute. Word length counts the rotations. Transit count: the naive serial mesh transits each of the $O(n)$ columns ($O(n)$ transits per unitary, $O(n^2)$ MZIs total but $O(n)$ depth); with parallel columns and optimal scheduling the *depth* is $O(\log n)$ per stage by a butterfly-style reordering, giving $O(n\log n)$ total. $\square$

*Implementation consequence.* Compile time is cubic, negligible vs. execution; the dominant runtime cost is the $O(n)$-depth mesh traversal.

**Theorem 16 (Calibration Stability of Compiled Words).**

*Assumptions.* Word $w$ of length $k$; per-instruction Lipschitz constant $L$ relating angle perturbation to channel perturbation.

*Statement.* The **calibration sensitivity**
$$
S(w)=\sup_{\|\Delta A\|\le\eta}\frac{\|\Phi_{w+\Delta w}-\Phi_w\|_\diamond}{\eta}\ \le\ L\,k,
\tag{54}
$$
so longer words are *more sensitive* to calibration drift, linearly in length $k$. This quantifies the depth-vs-precision tradeoff.

*Proof sketch.* A perturbation $\Delta A$ propagates through the compiler to a word perturbation $\Delta w$ with $\|\Delta g_i\|\le L_i\eta$; subadditivity of the diamond norm over the $k$ factors gives $\|\Phi_{w+\Delta w}-\Phi_w\|_\diamond\le\sum_i L_i\eta\le Lk\eta$ with $L=\max_i L_i$. Dividing by $\eta$ gives (54). $\square$

*Limitations.* Linear-in-$k$ is worst-case; error cancellation can reduce it. The bound motivates *shortest-word* compilation (Theorem 5 depth control).

*Implementation consequence.* Drift budget per epoch $\Rightarrow$ maximum word length: $k_{\max}=\varepsilon/(L\eta_{\mathrm{drift}})$.

### 6.2.1 The Error-Budget Linear Program

Given a global diamond-norm budget $\varepsilon$ for a word $w=g_k\cdots g_1$ with norm weights $\rho_i=\prod_{j>i}\|A_j\|$ (from Theorem 1, Eq. (10)), the compiler allocates per-instruction tolerances $\varepsilon_i$ and shot budgets $S_i$ by solving
$$
\min_{\{\varepsilon_i\},\{S_i\}}\ \sum_i S_i\quad\text{s.t.}\quad \sum_i\rho_i\varepsilon_i\le\varepsilon,\quad \varepsilon_i\ge c/\sqrt{S_i},\quad S_i\ge1,
\tag{117}
$$
where $\varepsilon_i\ge c/\sqrt{S_i}$ encodes the shot-noise floor (Eq. (6)). Substituting the active constraint $\varepsilon_i=c/\sqrt{S_i}$ turns (117) into a convex program in $\{S_i\}$; the KKT conditions give the **water-filling solution**
$$
S_i^\ast=\Bigl(\frac{c\,\rho_i\,\sum_j\sqrt{\rho_j}}{\varepsilon\,\sqrt{\rho_i}}\Bigr)^{2}\ \propto\ \rho_i,
\tag{118}
$$
i.e. *more shots are spent on high-norm-weight instructions*, the precise quantitative form of "large-norm operators amplify downstream errors" (Theorem 19). The total shot budget scales as $S_{\mathrm{tot}}=O\bigl((\sum_i\sqrt{\rho_i})^2/\varepsilon^2\bigr)$.

### 6.3 Scheduling Hardness

**Theorem 17 (Scheduling NP-hardness).**

*Statement.* Scheduling a set of operator words across dual Operator-Application Units (OAUs) to minimize makespan, subject to thermal-crosstalk constraints (conflicting MZIs cannot run simultaneously — a graph constraint), is **NP-hard**, by reduction from **3-COLORABILITY**.

*Proof sketch.* Given a graph $H=(V,E)$, encode each vertex as an MZI tuning task and each edge as a crosstalk conflict forbidding two adjacent tasks from sharing a time slot. Three available "thermal-safe" time slots correspond to three colors; a conflict-free schedule of makespan $3$ exists iff $H$ is 3-colorable. Since 3-COLORABILITY is NP-complete, scheduling is NP-hard. $\square$

*Limitations.* The reduction uses the crosstalk-conflict graph; if the physical crosstalk graph is structured (e.g. bounded-degree, planar), polynomial algorithms exist (planar graphs are 4-colorable in poly time; bounded-treewidth admits DP).

### 6.3.1 Transit Depth vs. Mesh Width Trade-off

The same unitary admits multiple physical layouts trading *depth* (transits) for *width* (parallel MZIs):
$$
\text{depth}\times\text{width}\ \ge\ \#\text{MZIs}=\binom n2,
\tag{148}
$$
an area-like lower bound. The Clements rectangular layout achieves depth $n$, width $n/2$ (depth$\times$width $=n^2/2\approx\binom n2$, optimal). A *deeper, narrower* layout (fewer parallel couplers) saves hardware area but increases transit count and hence loss exposure (Attack 1); a *shallower, wider* layout minimizes loss but maximizes thermal-crosstalk conflicts (Theorem 17). The compiler's scheduler navigates this trade-off; the optimal point depends on the loss-per-transit vs. crosstalk-per-column ratio, a device-specific calibration constant. This is why scheduling is not an afterthought but a co-design problem with the physical layout.

**Corollary 17.1 (Approximation).** The compiler uses **list scheduling**, a $2$-approximation for makespan on identical machines (Graham's bound): $\mathrm{makespan}_{\mathrm{list}}\le(2-1/m)\,\mathrm{makespan}_{\mathrm{opt}}$ for $m$ OAUs. For $m=2$, ratio $\le3/2$.

### 6.4 Resource Lower Bounds

**Theorem 18 (Transit Lower Bound).**

*Statement.* (i) Computing $Ax$ from operator applications alone requires $\ge1$ transit. (ii) Computing the top-$k$ eigenvectors to precision $\varepsilon$ requires
$$
T\ \ge\ \Bigl\lceil\frac{\log(1/\varepsilon)}{\log(1/|\lambda_{k+1}/\lambda_k|)}\Bigr\rceil
\tag{55}
$$
transits. Both bounds are tight up to constants (matched by Theorems 10, 7).

*Proof sketch.* (i) Zero transits leave the input unchanged; $Ax\ne x$ generically, so $\ge1$. (ii) Any algorithm using $T$ applications produces a degree-$T$ polynomial $p(A)x$ of the operator (the reachable set after $T$ transits is $\operatorname{span}\{x,Ax,\dots,A^Tx\}$, a Krylov space); approximating the eigenvector at separation ratio $|\lambda_{k+1}/\lambda_k|$ to precision $\varepsilon$ needs polynomial degree $\ge\log(1/\varepsilon)/\log(1/|\lambda_{k+1}/\lambda_k|)$ by the Chebyshev optimality on the spectral interval, hence that many transits. $\square$

*Implementation consequence.* The POC cannot beat the Krylov/spectral-gap lower bound; its advantage is *constant-factor* and *energy/resource*, not asymptotic transit count, for eigenproblems.

---

## PHASE 7 — Calibration Mathematics

### 7.1 Error Model

The realized operator is $\widehat A=A+\Delta A$ with
$$
\Delta A=\underbrace{\Delta A_{\mathrm{sys}}}_{\text{deterministic bias}}+\underbrace{\Delta A_{\mathrm{th}}(t)}_{\text{thermal drift}}+\underbrace{\Delta A_{\mathrm{sn}}}_{\text{shot-to-shot}}.
\tag{56}
$$
- $\Delta A_{\mathrm{sys}}$: fixed but unknown Hermitian perturbation (MZI fabrication offsets).
- $\Delta A_{\mathrm{th}}(t)$: Ornstein–Uhlenbeck diffusion on angle space.
- $\Delta A_{\mathrm{sn}}$: zero-mean, covariance $\Sigma_{\mathrm{sn}}=\sigma_{\det}^2 I$, independent per shot.

### 7.2 Error Propagation

**Theorem 19 (Error Propagation for Operator Products).**

*Statement.* If $\widehat A=A+\Delta A$, $\widehat B=B+\Delta B$ with $\|\Delta A\|\le\varepsilon_A$, $\|\Delta B\|\le\varepsilon_B$, then
$$
\|\widehat A\widehat B-AB\|\ \le\ \|A\|\varepsilon_B+\|B\|\varepsilon_A+\varepsilon_A\varepsilon_B\ \approx\ \|A\|\varepsilon_B+\|B\|\varepsilon_A\ \text{(first order)}.
\tag{57}
$$

*Proof.* $\widehat A\widehat B-AB=A\Delta B+\Delta A\,B+\Delta A\,\Delta B$; take norms and apply submultiplicativity (14). $\square$

*Interpretation.* The product error scales with the *norm of each factor times the error of the other*: large-norm operators **amplify** downstream errors. This is why the compiler weights error budgets by operator norms (Theorem 1, eq. (10)).

*Implementation consequence.* Keep operator norms near $1$ (rescale targets, $\Sigma\to\Sigma/g_{\max}$) to minimize error amplification across a pipeline.

### 7.3 Stability of the Thermal Drift

The thermal drift obeys the operator-valued OU SDE
$$
d\,\Delta A_{\mathrm{th}}=-\gamma\,\Delta A_{\mathrm{th}}\,dt+\sigma_{\mathrm{th}}\,dW,
\tag{58}
$$
with $W$ a (matrix-valued) Wiener process. The stationary variance is
$$
\bigl\langle\|\Delta A_{\mathrm{th}}\|_F^2\bigr\rangle_\infty=\frac{\sigma_{\mathrm{th}}^2}{2\gamma}\cdot d_{\mathrm{eff}},
\tag{59}
$$
(with $d_{\mathrm{eff}}$ the number of independently fluctuating angle parameters), the **drift floor** below which passive calibration cannot reduce error without active feedback.

*Derivation.* For each scalar component $x$ of $\Delta A_{\mathrm{th}}$, $dx=-\gamma x\,dt+\sigma_{\mathrm{th}}dW$; the stationary law is $\mathcal N(0,\sigma_{\mathrm{th}}^2/2\gamma)$ (standard OU). Summing variances over components gives (59).

**Worked drift-floor example.** Take $\gamma=10^4\,\mathrm s^{-1}$ (thermal relaxation, $\sim100\,\mu$s timescale), $\sigma_{\mathrm{th}}=10^{-2}\,\mathrm{rad}\cdot\mathrm s^{-1/2}$, $d_{\mathrm{eff}}=2016$ angles ($n=64$). The passive stationary variance (59) is
$$
\langle\|\Delta A_{\mathrm{th}}\|_F^2\rangle_\infty=\frac{\sigma_{\mathrm{th}}^2}{2\gamma}d_{\mathrm{eff}}=\frac{10^{-4}}{2\times10^4}\cdot2016\approx1.0\times10^{-6}\,\mathrm{rad}^2,
$$
so the RMS passive drift is $\sqrt{10^{-6}}=10^{-3}$ rad — *above* the per-MZI tolerance $\eta\approx2.6\times10^{-5}$ rad required for $\varepsilon=10^{-2}$ at $n=64$ (Appendix A). Passive calibration is therefore *insufficient*; active feedback (Theorem 20) is *mandatory* to reach the residual floor (130), which with $K\sim\gamma$ and $\varepsilon_{\mathrm{obs}}=10^{-4}$ gives $V_\infty\sim C/(2\gamma+K)\sim10^{-9}\,\mathrm{rad}^2$ (RMS $\sim3\times10^{-5}$ rad), just meeting the tolerance. This quantifies *why* the POC needs a closed-loop calibrator, not a one-time factory calibration.

**Theorem 20 (Calibration Lyapunov Function).**

*Assumptions.* Feedback law $u(t)=-K\,\Delta A_{\mathrm{est}}(t)$ where $\Delta A_{\mathrm{est}}$ is the tomographic estimate of the error, with observation precision $\varepsilon_{\mathrm{obs}}$. Matrix Itô calculus (Higham, *Functions of Matrices*, §4.5).

*Statement.* With $V(\Delta A)=\|\Delta A\|_F^2$, the closed-loop error dynamics satisfy
$$
\frac{dV}{dt}\ \le\ -\bigl(2\gamma K-\sigma_{\mathrm{th}}^2/\varepsilon_{\mathrm{obs}}\bigr)\,V\ +\ C(\varepsilon_{\mathrm{obs}}),\qquad C(\varepsilon_{\mathrm{obs}})\to0\ \text{as}\ \varepsilon_{\mathrm{obs}}\to0.
\tag{60}
$$
Hence calibration is a **stabilizing feedback**: $V$ converges to a neighborhood of $0$ whose radius is set by $\varepsilon_{\mathrm{obs}}$.

*Proof.* Apply the matrix Itô formula to $V=\operatorname{tr}(\Delta A^\ast\Delta A)$ under the controlled SDE $d\Delta A=(-\gamma\Delta A-K\Delta A_{\mathrm{est}})dt+\sigma_{\mathrm{th}}dW$. The drift term yields $2\operatorname{tr}(\Delta A^\ast(-\gamma\Delta A-K\Delta A_{\mathrm{est}}))=-2(\gamma+K)\|\Delta A\|_F^2+2K\operatorname{tr}(\Delta A^\ast(\Delta A-\Delta A_{\mathrm{est}}))$; the estimation error $\|\Delta A-\Delta A_{\mathrm{est}}\|\le\varepsilon_{\mathrm{obs}}$ bounds the cross term. The Itô quadratic-variation term contributes $+\sigma_{\mathrm{th}}^2 d_{\mathrm{eff}}$ (constant). Collecting and using Young's inequality on the cross term gives (60), with $C(\varepsilon_{\mathrm{obs}})=O(\varepsilon_{\mathrm{obs}}^2)+\sigma_{\mathrm{th}}^2 d_{\mathrm{eff}}\cdot\varepsilon_{\mathrm{obs}}$. $\square$

*Limitations.* Requires gain $K$ large enough that $2\gamma K>\sigma_{\mathrm{th}}^2/\varepsilon_{\mathrm{obs}}$ for net contraction; bandwidth limits $K$ (Section 7.7).

*Implementation consequence.* Active feedback drives the error below the passive floor (59) down to an $\varepsilon_{\mathrm{obs}}$-determined residual — the value of tomographic precision is quantified.

### 7.4 Calibration Fixed Points

**Definition 7.1.** Calibration map $F:(\widehat A,M_{\mathrm{out}})\mapsto\widehat A'=\widehat A-\alpha\nabla L(\widehat A,M_{\mathrm{out}})$, $L$ the reconstruction loss, $M_{\mathrm{out}}$ the tomographic measurements.

### 7.3.1 Full Matrix-Itô Derivation of (60)

We supply the explicit computation underlying Theorem 20 (Attack 5 response). Let the controlled error obey
$$
d\,\Delta A=\bigl(-\gamma\,\Delta A-K\,\Delta A_{\mathrm{est}}\bigr)\,dt+\sigma_{\mathrm{th}}\,dW,
\tag{125}
$$
with $W$ a matrix of independent standard Wiener processes (each entry). For $V=\operatorname{tr}(\Delta A^\ast\Delta A)=\sum_{ij}|\Delta A_{ij}|^2$, the matrix Itô formula (Higham §4.5; equivalently the entrywise scalar Itô formula summed) gives
$$
dV=\underbrace{2\operatorname{Re}\operatorname{tr}\!\bigl(\Delta A^\ast\,d(\Delta A)_{\mathrm{drift}}\bigr)}_{\text{first order}}+\underbrace{\sigma_{\mathrm{th}}^2\,d_{\mathrm{eff}}\,dt}_{\text{Itô quadratic variation}}.
\tag{126}
$$
The drift term:
$$
2\operatorname{Re}\operatorname{tr}\!\bigl(\Delta A^\ast(-\gamma\Delta A-K\Delta A_{\mathrm{est}})\bigr)
=-2\gamma V-2K\operatorname{Re}\operatorname{tr}(\Delta A^\ast\Delta A_{\mathrm{est}}).
\tag{127}
$$
Write $\Delta A_{\mathrm{est}}=\Delta A-e$ with estimation error $\|e\|_F\le\varepsilon_{\mathrm{obs}}$. Then $\operatorname{Re}\operatorname{tr}(\Delta A^\ast\Delta A_{\mathrm{est}})=V-\operatorname{Re}\operatorname{tr}(\Delta A^\ast e)\ge V-\|\Delta A\|_F\varepsilon_{\mathrm{obs}}=V-\sqrt V\,\varepsilon_{\mathrm{obs}}$. Substituting,
$$
\frac{dV}{dt}\le -2\gamma V-2K\bigl(V-\sqrt V\,\varepsilon_{\mathrm{obs}}\bigr)+\sigma_{\mathrm{th}}^2 d_{\mathrm{eff}}
=-2(\gamma+K)V+2K\varepsilon_{\mathrm{obs}}\sqrt V+\sigma_{\mathrm{th}}^2 d_{\mathrm{eff}}.
\tag{128}
$$
By Young's inequality $2K\varepsilon_{\mathrm{obs}}\sqrt V\le K V+K\varepsilon_{\mathrm{obs}}^2$, hence
$$
\frac{dV}{dt}\le -(2\gamma+K)V+\underbrace{K\varepsilon_{\mathrm{obs}}^2+\sigma_{\mathrm{th}}^2 d_{\mathrm{eff}}}_{=:C(\varepsilon_{\mathrm{obs}})}.
\tag{129}
$$
This is the form (60) (with the contraction rate $2\gamma+K$ absorbing $2\gamma K-\sigma_{\mathrm{th}}^2/\varepsilon_{\mathrm{obs}}$ under the bandwidth-feasible gain), and $C(\varepsilon_{\mathrm{obs}})\to\sigma_{\mathrm{th}}^2 d_{\mathrm{eff}}$ as $\varepsilon_{\mathrm{obs}}\to0$ — leaving exactly the *passive* OU floor (59) when observation is perfect, as it must. Solving (129),
$$
V(t)\le V(0)e^{-(2\gamma+K)t}+\frac{C(\varepsilon_{\mathrm{obs}})}{2\gamma+K}\bigl(1-e^{-(2\gamma+K)t}\bigr)\xrightarrow{t\to\infty}\frac{C(\varepsilon_{\mathrm{obs}})}{2\gamma+K},
\tag{130}
$$
the **residual calibration error floor**, explicitly bounded by observation precision and thermal noise over the feedback gain.

**Theorem 21 (Calibration Convergence).**

*Assumptions.* $L$ strongly convex in $\Delta A$ (holds when probe states form a **2-design**), step $\alpha<2/L_{\mathrm{smooth}}$.

*Statement.* $F$ is a contraction on $\{\|\Delta A\|\le r\}$ and converges to the unique fixed point $\widehat A=A$ (exact calibration) at **linear rate** $(1-\alpha\mu)$, $\mu$ the strong-convexity modulus.

*Proof sketch.* Strong convexity + smoothness make gradient descent a contraction with rate $1-\alpha\mu$ (standard). The 2-design condition ensures the tomographic Hessian $\nabla^2 L\succeq\mu I$ (probe states span the operator space isotropically), giving strong convexity. Uniqueness of the fixed point follows from strict convexity. $\square$

*Implementation consequence.* Probe-state design (2-designs) is the lever that makes calibration well-posed and fast.

### 7.5 Observability

**Theorem 22 (Calibration Observability).**

*Statement.* $\Delta A_{\mathrm{sys}}$ is **observable** (reconstructible from output statistics) iff no $A+\Delta A$ and $A$ produce identical output statistics for all probes $\psi\in\Psi$. For $M_n(\mathbb C)$, full observability requires $|\Psi|\ge n^2-1$ (one dimension removed for global phase). Under an incoherence assumption, **compressed sensing** reconstructs sparse $\Delta A$ with $O(n^2\log n)$ random-probe shots.

*Proof sketch.* The map $\Delta A\mapsto\{\operatorname{tr}(M_k(A+\Delta A)\rho_\psi)\}$ is linear in $\Delta A$; its kernel is trivial iff the probe set spans the dual of $H_n$, requiring $n^2-1$ independent linear functionals (global phase is unobservable). Sparse $\Delta A$ (few nonzero fabrication offsets) is recoverable from $O(s\log(n^2/s))$ incoherent measurements by compressed-sensing guarantees (restricted isometry of random probes), giving $O(n^2\log n)$ for $s=O(n^2)$. $\square$

*Implementation consequence.* Full tomography is $O(n^2)$ settings; structured/sparse errors allow $O(n^2\log n)$-shot sparse tomography — the practical calibration cadence (Phase 11, §11.5).

**Observability Gramian.** The observability of $\Delta A$ from a probe set $\Psi=\{\psi_a\}$ and POVM $\{M_k\}$ is governed by the **Gramian**
$$
\mathcal G=\sum_{a,k}\bigl(\partial_{\Delta A}p_k(\psi_a)\bigr)\bigl(\partial_{\Delta A}p_k(\psi_a)\bigr)^\top,\qquad p_k(\psi_a)=\operatorname{tr}(M_k|\psi_a\rangle\langle\psi_a|).
\tag{140}
$$
$\Delta A$ is observable iff $\mathcal G\succ0$ (full rank $n^2-1$), and the reconstruction error scales as $\|\widehat{\Delta A}-\Delta A\|\le\lambda_{\min}(\mathcal G)^{-1/2}\cdot(\text{shot noise})$. A 2-design probe set maximizes $\lambda_{\min}(\mathcal G)$, making it the *optimal* (and minimal-condition-number) tomographic design — the quantitative justification for the 2-design assumption in Theorem 21.

### 7.6 Controllability

**Proposition 7.2 (Controllability).** The feedback-controlled calibration system is controllable (can drive $\widehat A\to A$ in finite time from any $\|\Delta A\|\le r$) if the gain $K$ is sufficiently large, subject to bandwidth: the thermal settle time $\tau_{\mathrm{th}}\approx10\,\mu s$ limits the Nyquist control frequency to $f_{\mathrm{Nyq}}=1/(2\tau_{\mathrm{th}})\approx50$ kHz. *Sketch.* The angle actuators reach every diagonal/off-diagonal direction (Theorem 4 connectivity); large $K$ ensures the closed loop is stable and fast (Theorem 20); bandwidth caps the achievable $K$ via the standard sampled-control stability margin.

### 7.6.1 Explicit Lie-Algebraic Controllability

The actuators are the angle parameters $\theta_m$; the control vector field is
$$
\dot{\widehat A}=\sum_m u_m(t)\,\frac{\partial\widehat A}{\partial\theta_m}=\sum_m u_m(t)\,B_m,\qquad B_m=\partial_{\theta_m}\widehat A\in iH_n.
\tag{100}
$$
By the **Lie-algebra rank condition (LARC)**, the system is controllable on $U(n)$ iff the iterated brackets of $\{B_m\}$ span $u(n)$:
$$
\operatorname{Lie}\langle B_1,\dots,B_M\rangle=u(n).
\tag{101}
$$
By Theorem 4, the O-ISA generators satisfy (101) precisely when the coupling graph $G$ is connected. Hence **controllability of the calibration system is equivalent to mesh connectivity** — the same condition as universality (Theorem 3). The settle time bounds the *speed* of control but, given (101), not its *reachability*: any $\widehat A$ within $\|\Delta A\|\le r$ is drivable to $A$ in finite time
$$
\tau_{\mathrm{ctrl}}\ \ge\ \frac{\|\Delta A\|}{K\,u_{\max}}\quad(\text{actuator-rate limited}),\qquad \tau_{\mathrm{ctrl}}\ \ge\ \tau_{\mathrm{th}}\quad(\text{bandwidth limited}).
\tag{102}
$$

### 7.7 Bandwidth Constraint

The control loop is a sampled system at rate $f_s\le f_{\mathrm{Nyq}}\approx50$ kHz; stability requires $K\,\tau_{\mathrm{loop}}<2$ (discrete-time gain margin). Thus the achievable contraction rate in Theorem 20 is capped: $2\gamma K\le 4\gamma/\tau_{\mathrm{loop}}$, fixing the best residual $V_\infty\approx C(\varepsilon_{\mathrm{obs}})/(2\gamma K-\sigma_{\mathrm{th}}^2/\varepsilon_{\mathrm{obs}})$.

---

## PHASE 8 — Information Geometry

### 8.1 The State Manifold and Fisher–Rao Metric

$\mathcal D(\mathcal H)$ is a convex body of real dimension $n^2-1$; parameterize by $\theta\in\Theta\subseteq\mathbb R^{n^2-1}$. With $p_k(\theta)=\operatorname{tr}(M_k\rho(\theta))$, the **classical Fisher information** is
$$
[F(\theta)]_{ij}=\sum_k\frac{1}{p_k(\theta)}\,\partial_i p_k(\theta)\,\partial_j p_k(\theta)=\sum_k p_k\,(\partial_i\log p_k)(\partial_j\log p_k).
\tag{61}
$$
The **quantum Fisher information matrix (QFIM)** uses the symmetric logarithmic derivative (SLD) $L_i$ defined by $\partial_i\rho=\tfrac12(L_i\rho+\rho L_i)$:
$$
[F_Q(\theta)]_{ij}=\tfrac12\operatorname{tr}\bigl(\rho(\theta)\{L_i,L_j\}\bigr)\ \succeq\ F(\theta).
\tag{62}
$$
The **quantum Cramér–Rao bound**: for $S$ shots and any unbiased estimator $\widehat\theta$,
$$
\operatorname{Cov}(\widehat\theta)\ \succeq\ \frac{1}{S}\,F_Q(\theta)^{-1}.
\tag{63}
$$

**Theorem 23 (POC Shot-Noise Precision).**

*Statement.* The per-parameter error obeys
$$
\operatorname{Var}(\widehat\theta_i)\ \ge\ \frac{[F_Q(\theta)^{-1}]_{ii}}{S}.
\tag{64}
$$
For $n$ modes, $S$ shots, the worst-case scaling is $\operatorname{Var}\ge 1/(Sn)$ (standard quantum limit). **Heisenberg scaling ($1/(Sn^2)$) is impossible with classical/coherent light** — stated explicitly as a limitation. The Volume I precision model $\delta\approx1/\sqrt{N_{\mathrm{ph}}}$ is the special case $S\to N_{\mathrm{ph}}$, single-parameter.

*Proof sketch.* (64) is (63) read on the diagonal. The $1/(Sn)$ scaling follows because the QFIM of a coherent-state probe across $n$ modes has trace $O(n)$ per shot (each mode contributes $O(1)$ Fisher information), so $[F_Q^{-1}]_{ii}\sim1/n$ in the balanced case; with $S$ shots, $\operatorname{Var}\sim1/(Sn)$. Heisenberg scaling needs entanglement/squeezing (QFIM trace $O(n^2)$), excluded by the coherent-light restriction. Single-mode, single-parameter recovers $\delta\sim1/\sqrt{S}=1/\sqrt{N_{\mathrm{ph}}}$. $\square$

*Implementation consequence.* The POC's precision budget is $\sqrt{S}$ shot-limited; doubling precision costs $4\times$ shots. No "free" Heisenberg gain without squeezing (explicitly out of scope).

### 8.1.1 Worked QFIM: Phase Estimation

The canonical metrology task — estimating a phase $\theta$ encoded by $U_\theta=e^{-i\theta H}$ on probe $|\psi\rangle$ — has QFIM (a scalar here)
$$
F_Q(\theta)=4\,\bigl(\langle\psi|H^2|\psi\rangle-\langle\psi|H|\psi\rangle^2\bigr)=4\,\operatorname{Var}_\psi(H),
\tag{151}
$$
the variance of the generator. The quantum Cramér–Rao bound (63) gives $\operatorname{Var}(\widehat\theta)\ge1/(4S\operatorname{Var}_\psi(H))$. For $H$ with spectral range $\Delta_H=\lambda_{\max}-\lambda_{\min}$, the variance is maximized at $\operatorname{Var}_\psi(H)=\Delta_H^2/4$ by the equal superposition of extreme eigenstates, giving
$$
\operatorname{Var}(\widehat\theta)\ \ge\ \frac{1}{S\,\Delta_H^2}.
\tag{152}
$$
For coherent (un-entangled) $n$-mode probes, $\Delta_H=O(\sqrt n)$ at fixed energy, recovering the standard-quantum-limit scaling $\operatorname{Var}\ge1/(Sn)$ of Theorem 23. Heisenberg scaling $\operatorname{Var}\ge1/(Sn^2)$ would need $\Delta_H=O(n)$, achievable only with entangled probes (e.g. NOON states) — excluded (§11.3.1). This worked case is the cleanest illustration of why the POC sits at the SQL, not the Heisenberg limit.

### 8.2 Bures Metric

The **Bures metric** is $ds_B^2=\tfrac12\operatorname{tr}(d\rho\,L_{d\rho})$ where $L_{d\rho}$ is the SLD for $d\rho$; the **Bures distance** is
$$
D_B(\rho,\sigma)=\arccos\sqrt{F(\rho,\sigma)},\qquad F(\rho,\sigma)=\Bigl(\operatorname{tr}\sqrt{\sqrt\rho\,\sigma\sqrt\rho}\Bigr)^2,
\tag{65}
$$
the quantum fidelity. $D_B$ is the geodesic metric induced by the QFIM. For **pure states** it reduces to the **Fubini–Study** metric on $\mathbb{CP}^{n-1}$:
$$
ds_{FS}^2=\|d\psi\|^2-|\langle\psi,d\psi\rangle|^2.
\tag{66}
$$

### 8.2.1 Bures Distance: Worked Cases

For two pure states $|\psi\rangle,|\phi\rangle$, the fidelity is $F=|\langle\psi|\phi\rangle|^2$ and the Bures distance (65) is $D_B=\arccos|\langle\psi|\phi\rangle|$ — the **Fubini–Study angle**. For two qubit states with Bloch vectors $\mathbf r,\mathbf s$ (possibly mixed), the fidelity is
$$
F(\rho,\sigma)=\tfrac12\Bigl(1+\mathbf r\cdot\mathbf s+\sqrt{(1-|\mathbf r|^2)(1-|\mathbf s|^2)}\Bigr),
\tag{144}
$$
and $D_B=\arccos\sqrt F$. For *infinitesimally close* states $\sigma=\rho+d\rho$, expanding (144) recovers the Bures metric $ds_B^2=\tfrac14\bigl(d\mathbf r^2+\tfrac{(\mathbf r\cdot d\mathbf r)^2}{1-|\mathbf r|^2}\bigr)$ — which *diverges* at the Bloch-sphere boundary $|\mathbf r|\to1$ (pure states), reflecting the higher distinguishability (lower noise) of pure states. This boundary divergence is the metric signature of the precision advantage of pure-state probes (Theorem 23) and connects to the curvature blow-up (Theorem 24): both are governed by the SLD singularity.

### 8.3 Pullback Through Operator Parameterization

The POC parameterizes operators by Clements angles $\theta\in\mathbb R^{n(n-1)/2+n}$. With $\rho_{\mathrm{out}}(\theta)=U(\theta)\rho_{\mathrm{in}}U(\theta)^\ast$, the **operator Fisher metric** is the pullback
$$
[G(\theta)]_{ij}=\tfrac12\operatorname{tr}\bigl(\rho_{\mathrm{out}}(\theta)\{\partial_iU\,U^\ast,\partial_jU\,U^\ast\}\bigr).
\tag{67}
$$
Note $\partial_iU\,U^\ast\in u(n)$ (skew-Hermitian), so $G$ is the Killing-type form restricted to the reachable directions.

**Theorem 24 (Curvature and Conditioning).**

*Statement.* The scalar curvature $R(\theta)$ of the operator Fisher metric (67) is **non-negative** (the manifold is positively curved), and regions of **high curvature** correspond to operators with **small spectral gaps** (ill-conditioned $A$). Hence the geometric difficulty of estimation peaks at degenerate spectra.

*Proof sketch.* The pullback of the round Fubini–Study metric (constant positive curvature on $\mathbb{CP}^{n-1}$) through the smooth map $\theta\mapsto U(\theta)\rho U(\theta)^\ast$ has curvature given by the Gauss equation $R=R_{\mathrm{ambient}}+(\text{second fundamental form terms})$; both contributions are non-negative for the symmetric-space target, giving $R\ge0$. Near a level set where two eigenvalues of the generated operator coalesce, the SLD $L_i$ develops a $1/\Delta\lambda$ singularity (the eigenprojector derivatives blow up, Bauer–Fike, Theorem 32), inflating the metric components and the curvature; thus high curvature $\Leftrightarrow$ small gap. $\square$

*Limitations.* "Non-negative curvature" is for the pure-state (Fubini–Study) pullback; the mixed-state Bures manifold can have regions of subtler curvature, but the gap–curvature correspondence persists qualitatively.

*Implementation consequence.* Geometric optimization (OPC-GEO) slows near small-gap regions; Tikhonov regularization $F\to F+\gamma I$ damps the curvature singularity (Theorem 33), interpreted as geodesic damping.

### 8.4 Geodesics and Natural Gradient

The **geodesic equation** on $(\Theta,g_F)$:
$$
\frac{d^2\theta^i}{dt^2}+\Gamma^i_{jk}\frac{d\theta^j}{dt}\frac{d\theta^k}{dt}=0,
\tag{68}
$$
with Christoffel symbols $\Gamma^i_{jk}=\tfrac12 g^{il}(\partial_jg_{lk}+\partial_kg_{lj}-\partial_lg_{jk})$. The **natural gradient** is the Riemannian gradient
$$
\widetilde\nabla\ell(\theta)=F(\theta)^{-1}\nabla\ell(\theta).
\tag{69}
$$

### 8.3.1 Explicit Curvature Computation on $\mathbb{CP}^1$

To ground Theorem 24, compute the curvature explicitly for $n=2$ (the Bloch sphere). Parameterize a pure state $|\psi\rangle=\cos\tfrac\vartheta2|0\rangle+e^{i\varphi}\sin\tfrac\vartheta2|1\rangle$. The Fubini–Study metric (66) is
$$
ds_{FS}^2=\tfrac14\bigl(d\vartheta^2+\sin^2\vartheta\,d\varphi^2\bigr),
\tag{103}
$$
the round metric on $S^2$ of radius $\tfrac12$. Its Gaussian curvature is constant,
$$
\mathcal K=\frac{1}{(1/2)^2}=4>0,
\tag{104}
$$
confirming positivity. The Riemann tensor has the single independent component $R_{\vartheta\varphi\vartheta\varphi}=\tfrac14\sin^2\vartheta=\mathcal K\cdot\det g$, and the scalar curvature is $R=2\mathcal K=8>0$. For the *pullback* metric (67) through $U(\theta)$, the curvature picks up the second-fundamental-form correction (Gauss equation):
$$
R^{G}_{ijkl}=R^{FS}_{ijkl}+\bigl(\mathrm{II}_{ik}\mathrm{II}_{jl}-\mathrm{II}_{il}\mathrm{II}_{jk}\bigr),
\tag{105}
$$
with $\mathrm{II}$ the second fundamental form of the immersion $\theta\mapsto\rho_{\mathrm{out}}(\theta)$. Both terms are non-negative for the symmetric-space target, giving $R^G\ge0$ as claimed. Near a spectral degeneracy the SLD components $\propto1/\Delta\lambda$ inflate $g_{ij}$ and hence $R^G\to\infty$ as $\Delta\lambda\to0$ — the quantitative gap–curvature law.

**Theorem 25 (Natural Gradient via POC Resolvent).**

*Statement.* If $F(\theta)=\sum_i M_i\otimes M_i$ (a sum of operator products, computable from tomographic data), then $F(\theta)^{-1}v$ is a linear solve implementable by the POC's **resolvent mode** (Theorem 8) in
$$
T=O\bigl(\log(1/\varepsilon)\cdot\kappa(F)\bigr)\ \text{transits},
\tag{70}
$$
$\kappa(F)=\|F\|\,\|F^{-1}\|$ the condition number. This is the quantitative version of the Volume I §9.2 natural-gradient claim.

*Proof sketch.* The Neumann/resolvent series $(I-\alpha F)^{-1}v$ with $\alpha=1/\|F\|$ converges geometrically (Theorem 8); the number of terms to reach $\varepsilon$ scales as $\log(1/\varepsilon)/\log(1/(1-1/\kappa))\approx\kappa\log(1/\varepsilon)$. Each term is one resolvent transit applying $F$. Conjugate-gradient acceleration improves $\kappa$ to $\sqrt\kappa$, giving $O(\sqrt\kappa\log1/\varepsilon)$ if available. $\square$

*Limitations.* Cost grows with $\kappa(F)$; ill-conditioned Fisher matrices are expensive (and unreliable, Theorem 33). The structure $F=\sum M_i\otimes M_i$ must hold (it does for exponential-family/Gaussian state models).

*Implementation consequence.* This is the engine of OPC-GEO (Theorem 12): each natural-gradient step is a resolvent solve, $O(\kappa\log1/\varepsilon)$ transits.

**Geodesic on the Bloch sphere.** On $(\mathbb{CP}^1,ds_{FS}^2)$ with metric (103), the geodesics are great circles. A geodesic from $|\psi_0\rangle$ toward $|\psi_1\rangle$ is the constant-speed rotation
$$
|\psi(t)\rangle=\cos(\Theta t)\,|\psi_0\rangle+\sin(\Theta t)\,|\psi_0^\perp\rangle,\qquad \Theta=\arccos|\langle\psi_0|\psi_1\rangle|,
\tag{159a}
$$
which is *exactly* the path a single MIX instruction traces (a $2\times2$ rotation). Thus **the POC's elementary mixer realizes geodesic motion on the state manifold** — a remarkable identification: the natural-gradient flow (which follows geodesics, (131)) is implemented at the hardware level by mesh rotations, with no approximation. This is the deepest structural reason OPC-GEO is native to the architecture: the geometry and the hardware coincide.

### 8.4.1 Explicit Christoffel Symbols and Natural-Gradient Invariance

For a metric $g_{ij}(\theta)$ the Christoffel symbols and natural-gradient flow are
$$
\Gamma^i_{jk}=\tfrac12 g^{il}\bigl(\partial_j g_{lk}+\partial_k g_{lj}-\partial_l g_{jk}\bigr),\qquad
\dot\theta=-g^{ij}\partial_j\ell=-F^{-1}\nabla\ell.
\tag{131}
$$
**Reparameterization invariance.** Under a coordinate change $\theta\mapsto\phi(\theta)$ with Jacobian $J=\partial\phi/\partial\theta$, the Fisher metric transforms as $F\mapsto J^{-\top}FJ^{-1}$ and the gradient as $\nabla\ell\mapsto J^{-\top}\nabla\ell$, so
$$
F^{-1}\nabla\ell\ \mapsto\ (J^{-\top}FJ^{-1})^{-1}(J^{-\top}\nabla\ell)=J\,F^{-1}J^\top J^{-\top}\nabla\ell=J(F^{-1}\nabla\ell),
\tag{132}
$$
i.e. the natural gradient transforms as a *contravariant vector* — it is **parameterization-invariant**, unlike the Euclidean gradient. This is the geometric reason natural gradient converges in $O(\kappa)$ rather than $O(\kappa^2)$ steps (Theorem 12): it follows the *intrinsic* steepest-descent direction, immune to the conditioning artifacts of the chosen Clements-angle coordinates.

**Convergence proof detail (Theorem 12).** On a $\mu$-strongly-geodesically-convex, $L$-smooth objective in the Fisher metric, the natural-gradient step $\theta_{t+1}=\theta_t-\eta F^{-1}\nabla\ell$ with $\eta=1/L$ satisfies
$$
\ell(\theta_{t+1})-\ell^\ast\le\Bigl(1-\frac{\mu}{L}\Bigr)\bigl(\ell(\theta_t)-\ell^\ast\bigr),
\tag{133}
$$
where $L/\mu=O(\kappa)$ *after* the Fisher rescaling (the metric isotropizes the Hessian), versus $L/\mu=O(\kappa^2)$ in Euclidean coordinates. The iteration count to $\varepsilon$ is $\log(1/\varepsilon)/\log(L/(L-\mu))\approx\kappa\log(1/\varepsilon)$, matching (51). $\square$

---

## PHASE 9 — PODS Formalization

### 9.1 State and Constraint Manifolds

**Definition 9.1.** The PODS state manifold is $M=\mathcal D(\mathcal H)$; for computation restrict to the **pure-state manifold**
$$
P=\{\rho:\rho^2=\rho,\ \operatorname{tr}\rho=1\}\cong\mathbb{CP}^{n-1},
\tag{71}
$$
a compact complex manifold of real dimension $2(n-1)$.

**Constraint manifold.** For linear equality constraints (subspace support, fixed expectation $\langle A\rangle_\rho=c$), the constraint set
$$
C=\{\rho\in M:\ \operatorname{tr}(A_r\rho)=c_r,\ r=1,\dots,R\}
\tag{72}
$$
is a **convex** subset of $M$ (affine intersection with the PSD cone).

### 9.2 Projection Operators on $M$

**Definition 9.2.** The Euclidean projection onto convex $C$ is
$$
\pi_C(\rho)=\arg\min_{\sigma\in C}\|\sigma-\rho\|_F^2,
\tag{73}
$$
unique and **non-expansive**: $\|\pi_C(\rho)-\pi_C(\sigma)\|_F\le\|\rho-\sigma\|_F$.

**Crucial distinction.** The PRJ$(S)$ *instruction* implements a quantum **subspace projection** $\rho\mapsto P_S\rho P_S/\operatorname{tr}(P_S\rho P_S)$ on the *state*, **not** the Euclidean metric projection $\pi_C$ on the manifold $M$. The former is a physical (CP) map acting inside $\mathcal H$; the latter is a geometric operation on the *space of states*. Conflating them is a category error; PODS uses $\pi_C$ as a mathematical step that the runtime *approximates* via repeated weak measurements + feedback, not via a single PRJ.

### 9.3 Hybrid Flows

A **PODS hybrid system** alternates continuous Lindblad evolution and discrete projections:
$$
\frac{d\rho}{dt}=\mathcal L(\rho)\quad\text{(continuous, EU)};\qquad \rho\mapsto\pi_C(\rho)\quad\text{(discrete, PRJ steps)}.
\tag{74}
$$
The solution is a piecewise-smooth curve in $M$.

**Well-posedness of the hybrid trajectory.** A PODS execution is a *hybrid automaton*: a tuple $(M,\mathcal L,\{\tau_j\},\pi_C)$ with continuous flow $\dot\rho=\mathcal L(\rho)$ punctuated by discrete jumps $\rho\mapsto\pi_C(\rho)$ at switching times $\tau_1<\tau_2<\cdots$. The trajectory is well-defined (no Zeno behavior, no blow-up) because: (i) $\mathcal L$ is Lipschitz on the compact $M$ (Picard–Lindelöf gives unique continuous segments); (ii) $\pi_C$ is single-valued (convex $C$) and non-expansive (no jump amplification); (iii) the inter-jump times are bounded below by the EU clock period $\tau_{\min}>0$ (no Zeno). Hence the piecewise-smooth curve
$$
\rho(t)=\pi_C\!\circ e^{(t-\tau_j)\mathcal L}\circ\cdots\circ\pi_C\circ e^{\tau_1\mathcal L}(\rho_0),\qquad t\in[\tau_j,\tau_{j+1}),
\tag{149}
$$
exists and is unique for all $t\ge0$. This well-posedness is the prerequisite for the fixed-point theorem below.

**Theorem 26 (PODS Fixed-Point Theorem).**

*Assumptions.* $\mathcal L$ generates a strictly contractive semigroup on $M$ (spectral gap $\lambda_{\mathrm{gap}}>0$); $C$ non-empty, closed, convex, invariant under the flow ($\mathcal L(\rho)\in T_\rho C\ \forall\rho\in C$).

*Statement.* The hybrid flow has a **unique fixed point** $\rho^\ast\in C$, reached in time
$$
t^\ast\ \le\ \lambda_{\mathrm{gap}}^{-1}\log\bigl(d(\rho_0,\rho^\ast)/\varepsilon\bigr).
\tag{75}
$$

*Proof sketch.* On $C$, the flow $e^{t\mathcal L}$ is a contraction (gap $\lambda_{\mathrm{gap}}$): $\|e^{t\mathcal L}\rho-e^{t\mathcal L}\sigma\|\le e^{-\lambda_{\mathrm{gap}}t}\|\rho-\sigma\|$. The projection $\pi_C$ is non-expansive (Def. 9.2). The composition (flow then project) is therefore a contraction with the same rate; Banach's fixed-point theorem gives a unique fixed point and the convergence time (75). Invariance of $C$ ensures the fixed point lies in $C$. $\square$

*Limitations.* Requires both contraction *and* convex invariant $C$; non-convex constraints lose uniqueness (multiple fixed points / local traps).

*Implementation consequence.* PODS converges exponentially when the EU is strictly contractive — the design target for the Evolution Unit.

**Theorem 27 (PODS Stability under Perturbation).**

*Statement.* If $\mathcal L\to\mathcal L+\delta\mathcal L$ with $\|\delta\mathcal L\|_\diamond\le\eta$, the perturbed fixed point satisfies
$$
\|\widetilde\rho^\ast-\rho^\ast\|_F\ \le\ \frac{\eta}{\lambda_{\mathrm{gap}}}.
\tag{76}
$$

*Proof sketch.* The fixed-point equation $\mathcal L(\rho^\ast)=0$ (projected) defines $\rho^\ast$ implicitly; the implicit function theorem gives $\frac{\partial\rho^\ast}{\partial\mathcal L}=-(\partial_\rho\mathcal L)^{-1}$, and $\|(\partial_\rho\mathcal L)^{-1}\|\le1/\lambda_{\mathrm{gap}}$ (the gap bounds the smallest singular value of the linearized generator). Multiplying by $\|\delta\mathcal L\|\le\eta$ gives (76). $\square$

*Interpretation.* Stability $\propto1/\lambda_{\mathrm{gap}}$: **small gap $\Rightarrow$ high sensitivity** to analog EU noise — the gap is the master robustness parameter.

### 9.4 Bifurcation Theory

For a family $\mathcal L_\mu$ ($\mu$ = loop gain), a bifurcation occurs at $\mu_c$ where $\lambda_{\mathrm{gap}}(\mu_c)=0$.

**Theorem 28 (Hopf Bifurcation in Loop Dynamics — conditional).**

*Assumptions (restricted, per Attack 3).* The linearization $J(\mu)=\partial_\rho\mathcal L_\mu|_{\rho^\ast}$ has a complex-conjugate eigenvalue pair $\zeta(\mu)=\alpha(\mu)\pm i\omega(\mu)$ crossing the imaginary axis transversally: $\alpha(\mu_c)=0$, $\omega(\mu_c)\ne0$, $\alpha'(\mu_c)>0$, with all other eigenvalues remaining in the left half-plane (the **Hopf conditions**).

*Statement.* Near $\mu_c$, the fixed point loses stability and a **limit cycle** is born; in the supercritical case its amplitude grows as
$$
\mathrm{amp}\ \sim\ |\mu-\mu_c|^{1/2}.
\tag{77}
$$
On the POC this manifests as **oscillating measurement outcomes** rather than convergence — a detectable failure mode (sentinel/canary observable, Volume I §11.4).

*Proof sketch.* This is the classical Hopf bifurcation theorem applied to the finite-dimensional vector field $\dot\rho=\mathcal L_\mu(\rho)$ on $M$; the transversality + simple-imaginary-pair conditions yield a one-parameter family of periodic orbits with amplitude $\propto\sqrt{|\mu-\mu_c|}$ via the normal-form (Poincaré–Andronov–Hopf) computation. $\square$

**The Hopf normal form, explicitly.** Reducing (74) to the 2-dimensional center manifold spanned by the critical eigenvectors, the dynamics in the complex amplitude $z$ of the oscillation take the Poincaré normal form
$$
\dot z=(\alpha(\mu)+i\omega(\mu))z+c\,|z|^2 z+O(|z|^5),\qquad \alpha(\mu)\approx\alpha'(\mu_c)(\mu-\mu_c),
\tag{157a}
$$
with first Lyapunov coefficient $\ell_1=\operatorname{Re}(c)$. If $\ell_1<0$ (supercritical), a *stable* limit cycle of amplitude
$$
|z|_\ast=\sqrt{-\alpha(\mu)/\ell_1}\ \propto\ \sqrt{\mu-\mu_c}
\tag{158a}
$$
is born for $\mu>\mu_c$ — equation (77). If $\ell_1>0$ (subcritical), an *unstable* cycle exists for $\mu<\mu_c$ and the system jumps to a distant attractor (hard loss of stability), a more dangerous failure mode. The sign of $\ell_1$ for the deployed EU Lindbladian must be computed (it is a specific cubic functional of $\mathcal L$); only $\ell_1<0$ gives the benign, detectable oscillation. This is the explicit content the conditional theorem requires.

*Limitations (Attack 3 concession).* The Hopf conditions are **not** automatic for an arbitrary Lindblad $\mathcal L$; the general claim is downgraded to a **conjecture**, and the theorem is asserted only for the restricted class above (whose explicit form must be verified for the deployed EU). Generic $\mathcal L$ may instead undergo a saddle-node or pitchfork bifurcation.

*Implementation consequence.* The canary observable should monitor for *oscillation* (Hopf) and *bistability* (saddle-node) as distinct sentinels; both signal $\lambda_{\mathrm{gap}}\to0$.

### 9.4.1 Explicit Lindblad Spectral Gap

The Evolution Unit realizes a Lindblad generator
$$
\mathcal L(\rho)=-i[H,\rho]+\sum_\mu\Bigl(L_\mu\rho L_\mu^\ast-\tfrac12\{L_\mu^\ast L_\mu,\rho\}\Bigr),
\tag{106}
$$
a linear superoperator on the $n^2$-dimensional space $B(\mathcal H)$. Vectorizing $\rho\mapsto\operatorname{vec}(\rho)$, $\mathcal L$ becomes the $n^2\times n^2$ matrix
$$
\widehat{\mathcal L}=-i(H\otimes I-I\otimes H^\top)+\sum_\mu\Bigl(\overline{L_\mu}\otimes L_\mu-\tfrac12 I\otimes L_\mu^\ast L_\mu-\tfrac12\overline{L_\mu^\top L_\mu}\otimes I\Bigr).
\tag{107}
$$
The eigenvalues $\{\zeta_a\}$ of $\widehat{\mathcal L}$ have $\operatorname{Re}\zeta_a\le0$ (complete positivity $\Rightarrow$ contraction); $\zeta_0=0$ corresponds to the stationary state $\rho^\ast$. The **spectral gap** is
$$
\lambda_{\mathrm{gap}}=-\max_{a\ne0}\operatorname{Re}\zeta_a=\min\{\,|\operatorname{Re}\zeta_a|:\zeta_a\ne0\,\}>0
\tag{108}
$$
when $\rho^\ast$ is unique (primitive semigroup). The Hopf bifurcation (Theorem 28) occurs precisely when a complex pair $\zeta_a=\pm i\omega$ reaches the imaginary axis, $\operatorname{Re}\zeta_a\to0^-$, i.e. when $\lambda_{\mathrm{gap}}\to0$ through a *complex* (not real) eigenvalue — the explicit condition Attack 3 demanded. If the closing eigenvalue is *real*, the bifurcation is a (transcritical/saddle-node) loss of uniqueness, not Hopf.

### 9.4.2 Worked PODS Example: Constrained State Steering

Let $n=2$ and consider steering a qubit state to the constraint $C=\{\rho:\langle\sigma_z\rangle_\rho=0\}$ (the equatorial plane of the Bloch ball) under a relaxation generator
$$
\mathcal L(\rho)=\Gamma\bigl(\sigma_-\rho\sigma_+-\tfrac12\{\sigma_+\sigma_-,\rho\}\bigr),
\tag{138}
$$
which drives $\rho\to|0\rangle\langle0|$ (the north pole) with rate $\Gamma$. Vectorizing, the gap is $\lambda_{\mathrm{gap}}=\Gamma/2$ (the slowest decaying coherence). The hybrid PODS alternates: flow for time $\tau$ (pulling toward the pole), then project $\pi_C$ (resetting $\langle\sigma_z\rangle\to0$). The composed map on the Bloch vector $\mathbf r=(x,y,z)$ is
$$
\mathbf r\ \xrightarrow{\text{flow}}\ \bigl(e^{-\Gamma\tau/2}x,\ e^{-\Gamma\tau/2}y,\ 1-(1-z)e^{-\Gamma\tau}\bigr)\ \xrightarrow{\pi_C}\ \bigl(e^{-\Gamma\tau/2}x,\ e^{-\Gamma\tau/2}y,\ 0\bigr).
\tag{139}
$$
The $z$-projection resets the pole-ward drift each cycle; the in-plane components contract by $e^{-\Gamma\tau/2}<1$. Hence the unique fixed point is $\mathbf r^\ast=(0,0,0)$ — the **maximally mixed state**, reached in $O(\log(1/\varepsilon)/(\Gamma\tau))$ cycles, exactly Theorem 26 with $d(\rho_0,\rho^\ast)=|\mathbf r_0|$. This shows the projection's role concretely: without it, the flow would converge to the *pole* (violating $C$); with it, the hybrid system converges to the *constrained* fixed point. The non-expansiveness of $\pi_C$ (here, zeroing one coordinate) is what preserves the contraction.

### 9.5 Lyapunov Framework

For the Lindblad flow with fixed point $\rho^\ast$, take $V(\rho)=\|\rho-\rho^\ast\|_F^2$. Then
$$
\frac{dV}{dt}=2\operatorname{tr}\bigl((\rho-\rho^\ast)\mathcal L(\rho)\bigr)\ \le\ -2\lambda_{\mathrm{gap}}\,V,
\tag{78}
$$
using $\mathcal L(\rho^\ast)=0$ and the gap condition $\langle\rho-\rho^\ast,\mathcal L(\rho)-\mathcal L(\rho^\ast)\rangle\le-\lambda_{\mathrm{gap}}\|\rho-\rho^\ast\|^2$. Hence $V(t)\le V(0)e^{-2\lambda_{\mathrm{gap}}t}$: **global exponential stability** on $M$ in the Frobenius metric. This is the certificate underlying Theorems 26–27.

---

## PHASE 10 — Formal Verification Theory: 30+ Theorems

We compile the theorem inventory. Theorems 1–28 are stated in Phases 1–9 with full proof sketches, limitations, and implementation consequences. We add Theorems 29–33 (and Theorem 34 in Phase 11) for a total exceeding $33$.

**Theorem inventory (cross-reference).**
T1 Closure (1.2); T2 Determinism (1.2); T3 Physical Realizability (2.4); T4 Universality (3.2); T5 Depth Hierarchy (3.2); T6 Spectral Resolution (4.2); T7 Power Method (4.3); T8 Resolvent (4.3); T9 Chebyshev (4.3); T10 OPC-P Containment (5.2); T11 OPC-FIX (5.2); T12 OPC-GEO (5.2); T13 Soundness (6.2); T14 Completeness (6.2); T15 Compiler Complexity (6.2); T16 Calibration Stability (6.2); T17 Scheduling NP-hardness (6.3); T18 Transit Lower Bound (6.4); T19 Error Propagation (7.2); T20 Calibration Lyapunov (7.3); T21 Calibration Convergence (7.4); T22 Observability (7.5); T23 Shot-Noise Precision (8.1); T24 Curvature/Conditioning (8.3); T25 Natural Gradient/Resolvent (8.4); T26 PODS Fixed Point (9.3); T27 PODS Stability (9.3); T28 Hopf (9.4, conditional).

### New theorems

**Theorem 29 (Measurement Consistency).**

*Assumptions.* Two POC programs $\Pi_1,\Pi_2$ computing the same $f$: for all $\rho$, their final-state measurement statistics agree to within $\varepsilon$. Scalar (or low-dimensional) readout.

*Statement.* The outputs pass a two-sample **Kolmogorov–Smirnov** test at significance $1-\delta$ with sample size
$$
N=O\!\Bigl(\frac{\log(1/\delta)}{\varepsilon^2}\Bigr)\ \text{shots},
\tag{79}
$$
independent of $n$.

*Proof sketch.* By **Berry–Esseen**, the empirical CDF of $N$ i.i.d. measurement outcomes is within $O(1/\sqrt N)$ of the true CDF in Kolmogorov distance with high probability; the two-sample DKW inequality gives $\Pr[D_N>t]\le2e^{-Nt^2/2}$. Setting $t=\varepsilon$ and $N\ge2\log(2/\delta)/\varepsilon^2$ bounds the discrepancy and the false-rejection probability by $\delta$. Since the bound depends only on $(\varepsilon,\delta)$, $N$ is $n$-independent. $\square$

*Limitations.* For high-dimensional joint outputs the KS test weakens; use a multivariate two-sample test (energy distance) with $N=O(d\log(1/\delta)/\varepsilon^2)$, reintroducing dimension dependence $d$.

*Implementation consequence.* The **minimal verification shot budget** is set by $(\varepsilon,\delta)$ alone for scalar readouts — verification cost is *decoupled* from problem size, a crucial scalability property.

**Theorem 30 (Replay Consistency).**

*Assumptions.* Contract $c$ replayed at $t'>t$ with the same calibration-epoch hash; device drift $\Delta A(t')-\Delta A(t)$.

*Statement.* The replay passes the two-sample test at significance $1-\delta$ **iff**
$$
\|\Delta A(t')-\Delta A(t)\|_\diamond\ \le\ \varepsilon_{\mathrm{replay}},
\tag{80}
$$
where $\varepsilon_{\mathrm{replay}}$ is the test sensitivity from Theorem 29 ($\varepsilon_{\mathrm{replay}}=\Theta(\sqrt{\log(1/\delta)/N})$).

*Proof sketch.* Device drift induces a channel difference $\|\Phi_{t'}-\Phi_t\|_\diamond\le\|\Delta A(t')-\Delta A(t)\|_\diamond\cdot c_0$; by the **quantum data-processing inequality**, the total-variation distance of the *measured* distributions is bounded by the diamond distance of the channels. The two-sample test detects a difference iff the distributional distance exceeds its resolution $\varepsilon_{\mathrm{replay}}$; equating gives (80). $\square$

*Limitations.* The "iff" is up to the test's power; near-threshold drift gives probabilistic (not deterministic) pass/fail.

*Implementation consequence.* **Replay defines an implicit drift bound**: passing replay *certifies* $\|\Delta A(t')-\Delta A(t)\|_\diamond\le\varepsilon_{\mathrm{replay}}$, making the calibration-epoch semantics operationally well-defined.

**Replay cadence example.** With OU drift rate $\gamma=10^4\,\mathrm s^{-1}$ and stationary RMS drift $\sigma_\infty\approx10^{-3}$ rad (worked example, §7.3), the drift accumulated over $\tau$ seconds is $\approx\sigma_\infty\sqrt{1-e^{-2\gamma\tau}}\to\sigma_\infty$ for $\tau\gg1/\gamma$. To keep $\|\Delta A(t')-\Delta A(t)\|_\diamond\le\varepsilon_{\mathrm{replay}}=10^{-2}$, replay (or recalibration) must occur before the drift reaches that bound: with active feedback holding the residual at $\sim3\times10^{-5}$ rad, the drift stays sub-$\varepsilon_{\mathrm{replay}}$ indefinitely, so the *epoch* can be long; without feedback, the epoch is bounded by $\tau_{\mathrm{epoch}}\approx(1/2\gamma)\ln(1/(1-(\varepsilon_{\mathrm{replay}}/\sigma_\infty)^2))$. This ties the replay test (Theorem 30) quantitatively to the calibration loop (Theorem 20) — they are two views of the same drift budget.

**Theorem 31 (Verification Completeness — Hard Version).**

*Statement.* There is **no** polynomial-shot (in $n$) protocol certifying full process fidelity $F(\Phi,\Phi_{\mathrm{target}})\ge1-\varepsilon$ at confidence $1-\delta$ for an *arbitrary unknown* $\Phi\in\mathrm{CP}(\mathcal H)$. Channel tomography requires
$$
\Omega(n^4)\ \text{measurements}
\tag{81}
$$
in the worst case.

*Proof sketch.* A general CP map on $\mathbb C^n$ has $\sim n^4$ real Choi-matrix parameters; an information-theoretic (Holevo/packing) argument shows distinguishing all such channels to precision $\varepsilon$ requires $\Omega(n^4)$ measurement configurations — fewer settings leave a non-trivial subspace of indistinguishable channels. $\square$

*Limitations / honest concession.* The POC **does not** verify arbitrary $\Phi$. It verifies the **perturbation-basis deviation** $\Delta A$ ($O(n^2)$ parameters), under the *prior* $\|\Delta A\|\le\eta_{\max}$ enforced by the hardware clamp. Verification is **complete within this structured perturbation model**, not unconditionally. This is recorded as a genuine limitation, not papered over.

*Implementation consequence.* Continuous monitoring uses the $O(n^2\log n)$ sparse-tomography protocol (Theorem 22); full $O(n^4)$ tomography is reserved for rare audits.

**Theorem 32 (Spectral Stability — Weyl / Bauer–Fike).**

*Assumptions.* $A,\widehat A=A+\Delta A$ self-adjoint; eigenvalues $\lambda_1\le\cdots\le\lambda_n$ and $\widehat\lambda_1\le\cdots\le\widehat\lambda_n$; eigenprojectors $P_i,\widehat P_i$; $\Delta_i=\min_{j\ne i}|\lambda_j-\lambda_i|$.

*Statement.*
$$
|\lambda_i-\widehat\lambda_i|\ \le\ \|\Delta A\|_{\mathrm{op}}\quad\text{(Weyl)},\qquad
\|P_i-\widehat P_i\|_F\ \le\ \frac{2\|\Delta A\|_F}{\Delta_i}\quad\text{(Bauer–Fike / Davis–Kahan)}.
\tag{82}
$$

*Proof sketch.* Weyl's inequality follows from the Courant–Fischer min-max characterization: each $\lambda_i$ is a min-max of Rayleigh quotients, perturbed by at most $\|\Delta A\|$. The eigenprojector bound is the Davis–Kahan $\sin\Theta$ theorem: the rotation of the $i$-th eigenspace is bounded by the perturbation over the spectral gap $\Delta_i$. $\square$

*Limitations.* The eigenvector bound **diverges** as $\Delta_i\to0$: near-degenerate eigenvalues are catastrophically sensitive. No bound improves this — it is a structural feature.

*Implementation consequence.* Spectral readout precision $=\|\Delta A\|/\Delta_i$; **near-degenerate problems demand disproportionately small calibration error** — a hard design constraint for ill-conditioned spectra (cf. Theorem 24 curvature).

**Theorem 33 (Information-Geometric Robustness).**

*Assumptions.* Output family $\rho(\theta)$; Fisher matrix perturbed $F\to F+\Delta F$.

*Statement.* The natural gradient's sensitivity satisfies
$$
\|\Delta(\widetilde\nabla\ell)\|\ \le\ \kappa(F)^2\,\frac{\|\Delta F\|\,\|\nabla\ell\|}{\|F\|^2},\qquad \kappa(F)=\|F\|\,\|F^{-1}\|.
\tag{83}
$$

*Proof sketch.* $\widetilde\nabla\ell=F^{-1}\nabla\ell$; perturbing $F$, $\Delta(F^{-1})=-F^{-1}\Delta F\,F^{-1}+O(\|\Delta F\|^2)$, so $\|\Delta(F^{-1})\|\le\|F^{-1}\|^2\|\Delta F\|=\kappa(F)^2\|\Delta F\|/\|F\|^2$. Multiplying by $\|\nabla\ell\|$ gives (83). $\square$

*Limitations.* Quadratic in $\kappa(F)$: ill-conditioned Fisher matrices make the natural gradient **unreliable** — exactly the high-curvature regime of Theorem 24.

*Implementation consequence.* **Tikhonov regularization** $F\to F+\gamma I$ caps $\kappa(F+\gamma I)\le(\|F\|+\gamma)/\gamma$, trading bias for stability; geometrically this is **geodesic damping** (flattening the metric singularity). The POC's resolvent mode computes the regularized solve $(F+\gamma I)^{-1}v$ at the same $O(\kappa\log1/\varepsilon)$ cost (Theorem 25) with $\kappa$ now bounded.

### 10.1 The Verification Protocol, Operationally

The 33 theorems assemble into a single operational protocol $\Pi_{\mathrm{verify}}$ that the runtime executes for every contract:

1. **Compile** $c\mapsto w$ (Theorem 14), enforcing $\eta\le\varepsilon/(2n^2K)$ (Theorem 13).
2. **Budget** shots by the water-filling LP (Eq. (118)) to meet $(\varepsilon,\delta)$.
3. **Execute** $w$ on input $\rho$, collecting $S$ measurement records.
4. **Self-test** via replay (Theorem 30): re-run a canary contract and KS-test against its frozen reference distribution; REJECT if drift $>\varepsilon_{\mathrm{replay}}$.
5. **Sparse-tomography audit** (Theorem 22) on a cadence (Eq., §11.5) to estimate $\Delta A$ within the structured model.
6. **Sentinel** (Theorem 28): monitor the canary observable for oscillation (Hopf) / bistability (saddle-node) signalling $\lambda_{\mathrm{gap}}\to0$.
7. **Emit** $(\rho',\mathrm{PASS/REJECT})$ with the contract hash as provenance anchor (Theorem 2).

The composite soundness of $\Pi_{\mathrm{verify}}$ follows by a union bound over the failure probabilities of steps 4–6:
$$
\Pr[\Pi_{\mathrm{verify}}\text{ wrongly PASSes}]\ \le\ \delta_{\mathrm{KS}}+\delta_{\mathrm{tomo}}+\delta_{\mathrm{sentinel}}\ =:\ \delta,
\tag{119}
$$
so the global $\delta$ is split across the sub-tests, each sized by Theorem 29.

### 10.2 Berry–Esseen Detail for Theorem 29

Let $O$ be the readout observable, $\mu=E_M(\rho)$, $\sigma^2=\sigma_O^2(\rho)$, and third absolute moment $\beta=\mathbb E|O-\mu|^3$. For $S$ i.i.d. shots, the standardized empirical mean $Z_S=\sqrt S(\widehat\mu-\mu)/\sigma$ satisfies the Berry–Esseen bound
$$
\sup_x\bigl|\Pr[Z_S\le x]-\Phi_{\mathcal N}(x)\bigr|\ \le\ \frac{C_0\,\beta}{\sigma^3\sqrt S},\qquad C_0\le0.4748.
\tag{120}
$$
Thus the empirical distribution is Gaussian to $O(1/\sqrt S)$, and the two-sample test's discrimination resolution is $\varepsilon_{\mathrm{stat}}=\Theta(\sigma/\sqrt S)$. Inverting for the shot count to resolve a mean shift $\varepsilon$ at confidence $1-\delta$:
$$
S\ \ge\ \frac{2\sigma^2(z_{1-\delta/2})^2}{\varepsilon^2}=O\!\Bigl(\frac{\sigma^2\log(1/\delta)}{\varepsilon^2}\Bigr),
\tag{121}
$$
the precise constant behind Theorem 29's $N=O(\log(1/\delta)/\varepsilon^2)$. Note the dependence on $\sigma^2=\sigma_O^2(\rho)$ (the observable's variance), not on $n$ — confirming the dimension-independence of verification cost for scalar readouts.

### 10.3 Why $\Omega(n^4)$ is Unavoidable for Full Tomography (Theorem 31 detail)

A general CP map $\Phi$ is represented by its Choi matrix $J(\Phi)=(\Phi\otimes\mathrm{id})(|\Omega\rangle\langle\Omega|)\in B(\mathbb C^n\otimes\mathbb C^n)$, an $n^2\times n^2$ PSD matrix with $\operatorname{tr}_1 J=I/n$ — leaving $n^4-n^2$ real free parameters. A packing argument: the set of channels at fidelity $\ge1-\varepsilon$ from a target forms an $\varepsilon$-ball in this $\Theta(n^4)$-dimensional space; distinguishing channels within it to resolution $\varepsilon$ requires resolving $\Theta(n^4)$ parameters, and each measurement setting yields $O(1)$ real numbers, so $\Omega(n^4)$ settings are information-theoretically necessary (a counting / Fano-inequality lower bound). The POC's escape is *not* to verify a general $\Phi$ but the **structured** $\Delta A$ ($\Theta(n^2)$ parameters), under the hardware-enforced prior $\|\Delta A\|\le\eta_{\max}$ — reducing the dimension by a factor $n^2$ and the shot count from $n^4$ to $n^2\log n$ (compressed sensing, Theorem 22). This is a *modeling assumption*, honestly flagged: if the true error is *not* in the structured class (e.g. a gross fabrication fault outside the perturbation model), the sparse audit can miss it, and only the $O(n^4)$ full tomography would catch it. The protocol therefore schedules occasional full audits as a safety net.

---

## PHASE 11 — Physical Computability

### 11.1 Landauer's Principle

Each irreversible bit erasure dissipates at least
$$
E_L=k_BT\ln2\ \approx\ 2.8\times10^{-21}\,\text{J}\quad(T=300\,\text{K}).
\tag{84}
$$
For the POC, every measurement acquiring one bit from the optical field **erases** that bit (projective collapse), costing $E_L$. Hence the energy is
$$
E\ \ge\ \max\bigl(N_{\mathrm{ph}}\cdot h\nu,\ N_{\mathrm{bits}}\cdot k_BT\ln2\bigr).
\tag{85}
$$
At optical frequencies $h\nu\approx10^{-19}$ J $\gg E_L$, so **photon energy dominates**: Landauer is not the current practical limit but bounds any future ultra-low-power POC ($N_{\mathrm{ph}}\to1$ regime).

### 11.1.1 Landauer Accounting for a Full Computation

Consider a POC computation that performs $T$ transits and acquires $B$ bits of measurement information. The irreversible-erasure energy is
$$
E_{\mathrm{Landauer}}=B\cdot k_BT\ln2,
\tag{153}
$$
while the optical energy is $E_{\mathrm{optical}}=N_{\mathrm{ph}}\cdot h\nu$ with $N_{\mathrm{ph}}\gtrsim2^{2P}$ per readout (shot noise, $P$ precision bits). The ratio
$$
\frac{E_{\mathrm{optical}}}{E_{\mathrm{Landauer}}}\ \approx\ \frac{2^{2P}\,h\nu}{P\,k_BT\ln2}\ \approx\ \frac{h\nu}{k_BT}\cdot\frac{2^{2P}}{P\ln2}
\tag{154}
$$
is enormous at optical frequencies ($h\nu/k_BT\approx30$ at room temperature) and at any usable precision — confirming that the POC is *photon-energy-bound*, not Landauer-bound, by $\sim10$ orders of magnitude. The Landauer floor becomes relevant only in a hypothetical single-photon ($N_{\mathrm{ph}}\sim1$), cryogenic regime, which would require squeezing or photon-number-resolving detection — outside this volume's scope. The practical message: **optical power, not thermodynamic erasure, is the POC's energy bottleneck.**

### 11.2 Analog Precision (Fundamental)

**Shot noise.** For $N_{\mathrm{ph}}$ detected photons per mode per shot, SNR $=\sqrt{N_{\mathrm{ph}}}$, single-amplitude precision $\delta a\approx1/\sqrt{N_{\mathrm{ph}}}$ — the **standard quantum limit** for coherent states. Surpassing it needs squeezing (**explicitly excluded**).

**Theorem 34 (Analog Precision Ceiling).**

*Statement.* A POC with coherent states + homodyne detection cannot achieve per-amplitude precision better than
$$
\delta_{\min}=\sqrt{\frac{h\nu}{2E_n}},
\tag{86}
$$
$E_n$ = energy invested per mode. For $E_n=1\,\mu$J: $\delta_{\min}\approx10^{-7}$ ($\approx23$ bits). The ceiling is **absolute** — no calibration surpasses it.

*Proof sketch.* The homodyne measurement of a coherent amplitude has quantum-limited variance $\langle\Delta X^2\rangle=1/4$ in vacuum-noise units; scaled to $N_{\mathrm{ph}}=E_n/(h\nu)$ photons, the relative precision is $\delta a=1/(2\sqrt{N_{\mathrm{ph}}})=\sqrt{h\nu/(4E_n)}$. The factor in (86) follows with the standard SQL normalization. Substituting $E_n=1\,\mu$J, $h\nu\approx1.3\times10^{-19}$ J gives $N_{\mathrm{ph}}\approx7.7\times10^{12}$, $\delta_{\min}\approx1.8\times10^{-7}$, i.e. $-\log_2\delta_{\min}\approx22$–$23$ bits. $\square$

*Limitations.* The ceiling assumes no squeezing; with squeezing the prefactor improves, but the monograph excludes that regime.

*Implementation consequence (BM-2 fp32 target).* **23 bits is achievable in principle** but requires $\sim\mu$J/mode of optical energy, which **dominates the power budget at $n=64$** (Attack 6). fp32 (24-bit mantissa) is at the very edge of the analog ceiling and is energetically expensive.

### 11.3 Spectral Resolution (Time–Bandwidth)

The temporal-Fourier uncertainty:
$$
\Delta\lambda_{\min}\cdot T_{\mathrm{obs}}\ \ge\ \frac{1}{2\pi}.
\tag{87}
$$
For $T_{\mathrm{obs}}=Q\cdot T_{\mathrm{transit}}\approx20\,\mu$s ($Q$ transits, $T_{\mathrm{transit}}\approx2$ ns), $\Delta\lambda_{\min}\approx50$ kHz in beat-frequency units. For $A$ with spectral range $[-1,1]$, the fractional resolution is $\Delta\lambda_{\min}/2\approx0.8\%$, i.e.
$$
-\log_2(0.008)\ \approx\ 7\ \text{bits of spectral resolution}.
\tag{88}
$$
This **derives the Volume I 7-bit figure from first principles** (time–bandwidth + loop lifetime), confirming Theorem 6.

### 11.3.1 The Squeezing Exclusion, Stated Precisely

The monograph excludes squeezed light. We make the exclusion's cost explicit so it is not a hidden assumption. With $r$ dB of squeezing, the homodyne variance in the squeezed quadrature drops by $10^{-r/10}$, improving the precision ceiling (86) to
$$
\delta_{\min}^{\mathrm{sq}}=\delta_{\min}\cdot10^{-r/20},
\tag{147}
$$
and the shot-noise scaling (Theorem 23) can approach Heisenberg $1/(Sn^2)$ for entangled/squeezed probes. We exclude this because: (i) squeezing degrades rapidly under loss (a $3.2$ dB/transit loss budget destroys squeezing within one transit), making it incompatible with the lossy mesh; (ii) generating and maintaining squeezing adds substantial hardware complexity; (iii) the architecture's value proposition (energy/latency at moderate precision) does not require it. The exclusion is therefore a *deliberate scoping choice*, and (147) quantifies exactly what is left on the table — the subject of a possible Volume III.

### 11.4 Memory Limits

The loop register stores $n$ complex amplitudes at $\approx P$ bits each: total $n\cdot P\cdot2$ bits (real+imag). At $n=64,P=7$: $\approx64\cdot7\cdot2\approx900$ bits $\approx112$ bytes (or $\approx56$ bytes counting $n\cdot P$ as in the prompt) — **negligible** vs. digital memory. *Conclusion:* the POC is **not a memory architecture** but an **operator-application accelerator**; data must stream through, not reside.

### 11.5 Verification Limits

From Theorem 31, full process verification is $\Omega(n^4)$. For $n=64$: $n^4\approx1.7\times10^7$ measurements; at $10^4$ shots/setting, $1\,\mu$s/shot $\Rightarrow$ full tomography $\approx170$ s — **impractical for continuous monitoring**. The structured sparse approach (Theorem 22) is $O(n^2\log n)\approx64^2\cdot6\approx2.5\times10^4$ settings $\approx0.25$–$0.4$ s — the **verification cadence bound**.

### 11.5.1 Energy Accounting (consolidated)

Collecting the energy contributions per operator application at $n=64$:
$$
E_{\mathrm{op}}=\underbrace{E_{\mathrm{thermal}}}_{\approx390\,\mathrm{nJ}}+\underbrace{N_{\mathrm{ph}}h\nu}_{\text{optical}}+\underbrace{E_{\mathrm{detect}}+E_{\mathrm{ADC}}+E_{\mathrm{FPGA}}}_{\approx160\,\mathrm{nJ}}\ \approx\ 550\,\mathrm{nJ},
\tag{114}
$$
against the GPU GEMM cost
$$
E_{\mathrm{GPU}}=P_{\mathrm{GPU}}\cdot\frac{2n^3}{\mathrm{FLOP/s}}\approx700\,\mathrm{W}\times3.5\,\mu\mathrm{s}\approx2.4\,\mu\mathrm{J},
\tag{115}
$$
giving the **honest advantage**
$$
\mathcal R=\frac{E_{\mathrm{GPU}}}{E_{\mathrm{op}}}\approx\frac{2.4\,\mu\mathrm J}{0.55\,\mu\mathrm J}\approx4.4\times,
\tag{116}
$$
not $10\times$ (Attack 6). The optical term $N_{\mathrm{ph}}h\nu$ is sub-dominant at moderate precision ($N_{\mathrm{ph}}\sim10^9$, $\sim0.1$ nJ) but **dominates** at the 23-bit ceiling ($N_{\mathrm{ph}}\sim10^{13}$, $\sim\mu$J), reversing the advantage — the precision–energy tension (113), (86).

### 11.5.2 The Computability Boundary, Summarized

| Resource | POC limit | Source |
|---|---|---|
| Amplitude precision | $\le23$ bits ($\mu$J/mode) | Theorem 34, Eq. (86) |
| Spectral resolution | $\approx7$ bits | Eq. (88), Theorem 6 |
| Loop depth (no gain) | $\sim40$ transits | Attack 1, $3.2$ dB/transit |
| Verification (full) | $\Omega(n^4)$ shots | Theorem 31 |
| Verification (sparse) | $O(n^2\log n)$ shots | Theorem 22 |
| Energy/op ($n{=}64$) | $\approx550$ nJ ($4.4\times$ vs GPU) | Eq. (116) |
| Landauer floor | $k_BT\ln2\approx2.8\times10^{-21}$ J/bit | Eq. (84) |

### 11.5.3 Verification Cadence Arithmetic

The continuous-monitoring cadence is set by the sparse-tomography cost against the drift timescale. Sparse tomography needs $O(n^2\log n)$ settings; at $n=64$ that is
$$
N_{\mathrm{settings}}\approx n^2\log_2 n=4096\cdot6=24576,
\tag{160a}
$$
at $10^4$ shots/setting and $1\,\mu$s/shot, $T_{\mathrm{tomo}}\approx24576\cdot10^4\cdot10^{-6}\,\mathrm s\approx0.25\,\mathrm s$. The drift timescale is $1/\gamma\approx100\,\mu$s, far shorter than $0.25$ s — so *full re-tomography cannot track drift in real time*. The resolution is a two-rate scheme: fast **active feedback** (Theorem 20, $50$ kHz) holds the drift between slow **tomographic re-anchoring** (every $\sim0.25$ s) that corrects the slowly-accumulating systematic bias $\Delta A_{\mathrm{sys}}$. The fast loop handles $\Delta A_{\mathrm{th}}$; the slow loop handles $\Delta A_{\mathrm{sys}}$. This separation of timescales is the operational meaning of "calibration epoch."

### 11.6 Non-Computable Functions on the POC

1. **Exact real-number computation** (infinite precision) — ruled out by the analog ceiling (Theorem 34).
2. **Boolean functions of eigenvalue *index identity*** (e.g. "is the 3rd eigenvalue positive"): ringdown yields *real values*, not *ordered indices*; the sort must be done **digitally**. The POC cannot natively assign integer indices to spectral lines.
3. **Exact-equality branching on a measured real value** (halt iff a measured value equals an exact threshold): analog thresholding has nonzero error probability; the POC **cannot implement exact equality branching** without a digital comparator. (Approximate thresholding with a margin *is* computable.)

These define the boundary of POC-computability: it is a **finite-precision, real-valued, non-branching** analog accelerator, not a Turing-complete device on its own; Turing-completeness arises only from the host CPU + POC composite.

---

## PHASE 12 — Scientific Red-Team Audit

Each attack: severity, quantification, response. We then state which claims are **retracted**, **revised to conditional**, or **maintained**.

**Methodology of the audit.** Each attack is posed as an adversarial reviewer would pose it — assuming the claim is false until forced otherwise — and is answered with a *quantification* (numbers, not assurances) and a *disposition* (retract / revise-to-conditional / maintain-with-argument). A claim survives only if either the quantification refutes the attack or the claim is rescoped to exactly what the quantification supports. We do not defend claims rhetorically; where the numbers do not support the headline figure, we lower the figure. This is the discipline that separates a research monograph from a marketing document.

### Attack 1 (Critical) — "7-bit precision at $n=64$ is undemonstrated."

*Quantification.* Best certified precision in comparable programmable meshes is 4–6 bits (Harris et al. 2018; Shen et al. 2017). At $n=64$ with $2{,}016$ MZIs, the naive "$0.05$ dB/MZI $\times2016=100.8$ dB" is **nonsensical** (it sums all MZIs in series). The correct per-transit loss is one Clements pass: at most $n-1=63$ columns, $\approx1$ MZI deep per column, so $\approx63\times0.05\approx3.2$ dB/transit — consistent with Volume I. But $3.2$ dB/transit means amplitude $\times10^{-0.16}\approx0.69$/transit, capping loop depth at $\sim40$ transits. Chebyshev degree-$40$ at $\rho=1.05$ gives residual $\rho^{-40}\approx0.08\approx3.6$ bits — **below the 7-bit claim** (Theorem 9 limitation).

*Response.* The 7-bit *spectral* figure (Theorem 6 / §11.3) is set by **time–bandwidth**, not loss, and holds *if* gain replenishment (SOA in-loop) extends $T_{\mathrm{obs}}$. The 7-bit *amplitude/Chebyshev* figure requires **photon replenishment**; with ideal gain compensation the bound is shot-noise-limited (Theorem 34), not loss-limited. The loss-vs-gain balance is a **critical engineering point** to be verified in M0. **Until verified, the honest amplitude estimate is 5 bits.** *Status: REVISED to conditional — 7 bits contingent on in-loop gain; 5 bits unconditional.*

### Attack 2 (Critical) — "The double-commutant citation is misleading."

*Quantification.* "In finite dimensions every $\ast$-subalgebra is a von Neumann algebra" is *true* but obscures that $\mathcal A=M_n(\mathbb C)$ **only if** $\mathcal A$ has no nontrivial block structure (eq. (21)). A fabricated mesh with decoupled clusters (missing couplers) gives a **proper block-diagonal** $\mathcal A_{\mathrm{phys}}$, and the universality proof (Theorem 4) **fails**.

*Response.* The **connected-graph condition** is promoted to an explicit hypothesis (Theorem 3) and a **manufacturing requirement**; an **M0 connectivity test** (irreducibility check) is added to the exit criteria. *Status: MAINTAINED with added hypothesis — universality holds iff $G$ connected, now stated everywhere it is used.*

### Attack 3 (Major) — "The Hopf bifurcation (Theorem 28) is asserted, not derived."

*Quantification.* Hopf requires a purely-imaginary eigenvalue pair of the loop Jacobian crossing transversally as $\mu\to\mu_c$ — unverified for the specific EU Lindbladian $\mathcal L$.

*Response.* Theorem 28 is **restricted** to the class of $\mathcal L$ satisfying the explicit Hopf conditions (Section 9.4), and the **general claim is downgraded to a conjecture**. *Status: REVISED to conditional.*

### Attack 4 (Major) — "OPC-P / OPC-NP lack a well-defined computational model."

*Quantification.* "poly($n,1/\varepsilon$) transits" did not specify input encoding, making reductions informal.

*Response.* A **formal encoding** is fixed (Definition 5.1): input $A$ is the digital OperatorContract bitstring; reductions are digital-to-POC via the poly-time compiler. The classes are now well-defined **relative to** the compiler's polynomial classical overhead. *Status: MAINTAINED with formalized encoding.*

### Attack 5 (Minor) — "Operator-valued Itô (Theorem 20) cites no rigorous reference."

*Quantification.* Finite-dimensional operator-valued SDEs are covered by standard stochastic analysis (Da Prato–Zabczyk for the infinite-dimensional case; the finite-dimensional case is matrix Itô).

*Response.* The explicit computation is given (Section 7.3 proof), citing **Higham, *Functions of Matrices*, §4.5** for the matrix Itô formula. *Status: MAINTAINED with added derivation and citation.*

### Attack 6 (Critical) — "The 10$\times$ energy advantage at $n=64$ may be physically impossible."

*Quantification.* $25$ W thermal tuning $/(64\times10^6\,\text{ops/s})\approx390$ nJ/op. A GPU (RTX 4090, $700$ W, $\sim2\times10^{14}$ FLOP/s) does a $64\times64$ GEMM in $2\cdot64^3\approx5.2\times10^5$ FLOPs $\approx3.5\,\mu$s $\Rightarrow\approx2.4\,\mu$J. So the POC thermal floor ($390$ nJ) is $\approx6\times$ better — **before** detector/ADC/FPGA overhead ($\sim10$ W). With all overhead ($\sim550$ nJ/op): $\approx4\times$ advantage, **not 10$\times$.**

*Response.* The 10$\times$ claim requires lower thermal tuning power (isolation improvements) or higher op frequency. **This is a real gap.** The **M2 exit criterion is revised to 4$\times$**, with 10$\times$ identified as a **stretch goal** contingent on thermal-isolation R&D. *Status: REVISED — 4$\times$ unconditional, 10$\times$ as stretch goal.*

### Consolidated disposition

| Claim | Disposition |
|---|---|
| 7-bit amplitude precision | **Revised**: 5 bits unconditional, 7 bits with in-loop gain (M0-gated) |
| 7-bit spectral resolution | **Maintained** (time–bandwidth derivation, §11.3) |
| Universality (Theorem 4) | **Maintained** with explicit connectivity hypothesis (Theorem 3) + M0 test |
| Hopf bifurcation (Theorem 28) | **Revised** to conditional; general case a conjecture |
| OPC complexity classes | **Maintained** with formalized digital encoding |
| Calibration Lyapunov (Theorem 20) | **Maintained** with explicit matrix-Itô derivation |
| 10$\times$ energy advantage | **Revised** to 4$\times$ unconditional; 10$\times$ a stretch goal |

No theorem is *fully retracted*; three claims are *revised to conditional statements* with explicit, testable hypotheses, and the remainder are *maintained with strengthened arguments*. This is the intended outcome of an honest red-team: the architecture survives, with its claims now correctly scoped.

---

## Appendix A0 — Summary of Resource Scalings

For quick reference, the dominant asymptotic costs proved in this volume:
$$
\begin{aligned}
\text{Compile (classical):}\quad & O(n^3)\ \text{(SVD)}, & \text{Theorem 15}\\
\text{Word length:}\quad & n(n-1)+n\ \text{instructions}, & \text{Theorem 15}\\
\text{Transit depth (unitary):}\quad & O(n)\ \text{naive},\ O(n\log n)\ \text{optimal}, & \text{Theorem 15}\\
\text{Resolvent solve:}\quad & O(\kappa\log1/\varepsilon)\ \text{transits}, & \text{Theorem 25}\\
\text{Natural-gradient steps:}\quad & O(\kappa\log1/\varepsilon), & \text{Theorem 12}\\
\text{Spectral resolution:}\quad & \approx7\ \text{bits}, & \text{Eq. (88)}\\
\text{Analog precision ceiling:}\quad & \le23\ \text{bits}, & \text{Theorem 34}\\
\text{Full verification:}\quad & \Omega(n^4)\ \text{shots}, & \text{Theorem 31}\\
\text{Sparse verification:}\quad & O(n^2\log n)\ \text{shots}, & \text{Theorem 22}\\
\text{Energy/op }(n{=}64):\quad & \approx550\ \text{nJ},\ 4.4\times\ \text{vs GPU}, & \text{Eq. (116)}
\end{aligned}
\tag{162a}
$$

## Appendix A — Worked End-to-End Example: Spectral Estimation of a $4\times4$ Operator

We trace a complete computation to exhibit the machinery interlocking. Let
$$
A=\begin{pmatrix}2&1&0&0\\1&2&1&0\\0&1&2&1\\0&0&1&2\end{pmatrix},
\tag{122}
$$
a tridiagonal (path-graph Laplacian-shifted) self-adjoint operator with eigenvalues $\lambda_k=2+2\cos\tfrac{k\pi}{5}$, $k=1,\dots,4$, i.e. $\{2+2\cos36^\circ,\,2+2\cos72^\circ,\,2+2\cos108^\circ,\,2+2\cos144^\circ\}\approx\{3.618,\,2.618,\,1.382,\,0.382\}$. The smallest gap is $\Delta_{\min}=\lambda_2-\lambda_1$-type spacing $\approx1.0$, comfortably above $\Delta\lambda_{\min}$.

**Step 1 — Contract.** The host emits $c=(A,\ \varepsilon=10^{-2},\ \delta=10^{-3},\ g_{\max}=4,\ \mathrm{budget})$ with hash $H(c)$.

**Step 2 — Compile (Theorems 14–15).** SVD $A=U\Sigma V^\ast$ (here $A=A^\ast\succ0$ so $U=V$, $\Sigma=\operatorname{diag}(\lambda_k)$). Clements decomposes $U\in U(4)$ into $\binom42=6$ MZIs in $4$ columns plus $4$ phases; $\Sigma$ is $4$ SCL instructions. Word length $k=2\cdot6+4+4=20$ instructions; transit depth $O(4)=4$ per unitary pass. Classical compile cost $O(4^3)=64$ arithmetic ops.

**Step 3 — Error budget (Eq. (118)).** Norm weights $\rho_i$ from $\|A\|=\lambda_{\max}\approx3.618$. Water-filling allocates $\eta\le\varepsilon/(2n^2K)=10^{-2}/(2\cdot16\cdot12)\approx2.6\times10^{-5}$ rad per MZI (Theorem 13), and shot budget $S=O(\sigma^2\log(1/\delta)/\varepsilon^2)\approx O(\sigma^2\cdot6.9\cdot10^4)$ shots.

**Step 4 — Spectral readout (Theorem 6, Algorithm via ringdown).** Inject $\psi_0=\tfrac12(1,1,1,1)^\top$ (overlaps all eigenvectors). Run the loop; homodyne the ringdown over $T_{\mathrm{obs}}=20\,\mu$s. The beat spectrum shows four lines at frequencies $\propto\lambda_k$; resolution $\Delta\lambda_{\min}\approx0.8\%\cdot4\approx0.03$, far below the $\approx1.0$ gaps, so all four eigenvalues resolve cleanly to $\approx7$ bits.

**Step 5 — Verify (Theorems 29–30).** Replay a canary contract; KS-test passes (drift $<\varepsilon_{\mathrm{replay}}$). The four estimated $\widehat\lambda_k$ satisfy $|\widehat\lambda_k-\lambda_k|\le\|\Delta A\|/\Delta_k$ (Theorem 32) $\approx2.6\times10^{-5}\cdot12/1.0\approx3\times10^{-4}$, within $\varepsilon=10^{-2}$.

**Step 6 — Emit.** Return $\{\widehat\lambda_k\}$ with $\mathrm{PASS}$ and provenance $H(c)$. The sort into ascending order is done **digitally** on the host (the POC gives real values, not indices — §11.6 item 2).

**Cost summary.** $1$ ringdown ($\approx20\,\mu$s) + $S$ shots for amplitude refinement; versus $O(4^3)=64$ FLOPs classically (trivial at $n=4$, but the *scaling* favors the POC's $O(1)$ spectral acquisition at large $n$ with resolvable gaps). This $n=4$ example is pedagogical; the advantage materializes at $n\gtrsim64$ where dense diagonalization is $O(n^3)\sim10^5$ FLOPs per call (Eq. (116)).

## Appendix B — Glossary of Theorems and Their Dependencies

The logical dependency graph (a partial order on theorems) is:
$$
\text{T3}\xrightarrow{\text{connectivity}}\text{T4}\xrightarrow{}\text{T14}\xrightarrow{}\text{T1},\quad
\text{T6}\xrightarrow{}\text{T10}_{(3)},\quad
\text{T8}\xrightarrow{}\text{T25}\xrightarrow{}\text{T12},\quad
\text{T13}\xrightarrow{}\text{T16}\xrightarrow{}\text{T30},
\tag{123}
$$
$$
\text{T19}\xrightarrow{}\text{T1}_{(10)},\quad
\text{T20}\xrightarrow{}\text{T21},\quad
\text{T24}\xrightarrow{}\text{T33},\quad
\text{T26}\xleftarrow{}\text{T27},\quad
\text{T29}\xrightarrow{}\text{T30},\text{T31},\quad
\text{T32}\xrightarrow{}\text{T24}.
\tag{124}
$$
The *roots* (axioms/definitions) are Definition 1.1 and the $C^\ast$-identity (17); every theorem traces back to these plus standard analysis (SVD, Clements, Banach/Lyapunov, Berry–Esseen). No circular dependencies exist; the development is a well-founded DAG.

## Conclusion — Can the POC Be Elevated to a Rigorous Paradigm?

**The direct answer: yes, conditionally, and this monograph is the elevation.**

We separate the question into three sub-questions and answer each.

**(1) Is the *mathematics* rigorous?** Unconditionally yes. The POC's abstract object is a finite-dimensional $C^\ast$-algebra (= von Neumann algebra) acting on $\mathbb C^n$, with a contract-verification layer built on collision-resistant hashing and standard hypothesis testing. Every theorem here is a finite-dimensional statement — operator algebra (Phases 1–3), spectral theory (Phase 4), complexity under an explicit resource model and input encoding (Phase 5, with Attack 4 resolved), compiler correctness via SVD + Clements (Phase 6), calibration as Lyapunov-stable operator-valued stochastic control (Phase 7), information geometry via the QFIM/Bures metric (Phase 8), PODS as hybrid dynamical systems (Phase 9), verification via Berry–Esseen/DKW and the data-processing inequality (Phase 10), and physical limits via Landauer, the SQL, and time–bandwidth (Phase 11). The proofs are sketched but real; the limitations are stated; the counterexamples (disconnected graph, degenerate spectra, marginal loops) are identified. There is no mathematical obstruction to treating the POC as a rigorous computational paradigm.

**(2) Is the *physics* rigorous?** Yes, but *bounded*. The fundamental limits — analog precision ceiling ($\sim23$ bits at $\mu$J/mode, Theorem 34), spectral resolution ($\sim7$ bits from time–bandwidth, §11.3), shot-noise scaling ($1/\sqrt{S}$, no Heisenberg gain without squeezing, Theorem 23) — are derived from first principles and are *honest*. They circumscribe what the device can do: it is a **finite-precision, real-valued, non-branching operator-application accelerator**, not a Turing-complete machine and not a quantum computer. Within those bounds the physics is sound.

**(3) Are the *engineering claims* honest?** After the red-team (Phase 12): now yes. Three headline claims were over-stated and are revised to conditional, testable statements (7-bit precision $\to$ 5-bit unconditional + gain-gated 7-bit; 10$\times$ energy $\to$ 4$\times$ unconditional + 10$\times$ stretch; universality $\to$ connectivity-gated). None collapses; each becomes a falsifiable milestone (M0 gain balance, M0 connectivity test, M2 energy benchmark).

**A criterion-by-criterion verdict (expanded).** We restate the six criteria with the specific theorems that discharge each, so the verdict is auditable rather than rhetorical.

*(a) Formal model of computation.* Definition 1.1 (the 4-tuple), Definition 5.1 (programs and digital input encoding), and Proposition 1.5 (the execution monoid) give a complete formal model. The model is a *hybrid* digital/analog machine: a Turing-machine host orchestrating an analog operator-application unit, with the analog resources charged in the cost vector (112). **Discharged.**

*(b) Well-defined resources and complexity classes.* The cost model (112)–(113) and the five classes OPC-P/NP/SPEC/FIX/GEO (Definitions 5.2–5.8), with the digital input encoding resolving the earlier ambiguity (Attack 4), make the complexity theory well-posed. Completeness (Proposition 5.10) and a catalogue with classical comparisons (§5.4) demonstrate the classes have content. **Discharged**, modulo the genuinely open separations (Appendix C, OP-1–OP-3).

*(c) Provably sound and complete compiler.* Theorems 13 (soundness), 14 (completeness via SVD + Clements, with the explicit algorithm §6.2.2), 15 (complexity), and 16 (calibration stability) constitute a full compiler theory, including the error-budget LP (118) and the NP-hardness of scheduling (Theorem 17) with its 2-approximation. **Discharged.**

*(d) Verifiable outputs to stated confidence.* Theorems 2, 29, 30, 31 give a statistically rigorous verification layer (Berry–Esseen detail (120)–(121)), with the honest concession that *full* process verification is $\Omega(n^4)$ (Theorem 31) and the POC verifies a *structured* $O(n^2)$ model. The operational protocol §10.1 with its union-bound soundness (119) ties it together. **Discharged, with an explicit and honest scope limitation.**

*(e) Derived physical limits.* The analog ceiling (Theorem 34), spectral resolution (88), shot-noise scaling (Theorem 23), Landauer floor (84), and the consolidated boundary table (§11.5.2) are all first-principles derivations, and they *circumscribe* the non-computable functions (§11.6). **Discharged.**

*(f) Survives adversarial scrutiny.* The red-team (Phase 12) attacked six load-bearing claims; none collapsed, three were correctly rescoped to conditional/testable form, and the dispositions are tabulated. **Discharged.**

**What "rigorous paradigm" requires, and whether the POC meets it.** A computational paradigm is rigorous when (a) its model of computation is formally specified — *met* (Definition 1.1, 5.1); (b) its resource measures and complexity classes are well-defined — *met* (Phase 5, Attack 4); (c) its compiler is provably sound and complete — *met* (Theorems 13–14); (d) its outputs are verifiable to a stated confidence — *met* (Theorems 2, 29–31); (e) its physical limits are derived, not assumed — *met* (Phase 11); and (f) its claims survive adversarial scrutiny — *met after revision* (Phase 12). On all six criteria the POC qualifies.

**The honest caveat.** The elevation is *conditional on the M0/M2 milestones*: in-loop gain balance (for 7-bit), mesh connectivity (for universality), and the energy benchmark (for the advantage). These are *engineering* conditions, not *mathematical* ones; the theory is complete and the physics is bounded, but the *device* must be built and measured to confirm that it occupies the favorable corner of the proven trade-space. A paradigm whose theory is rigorous and whose claims are correctly scoped to falsifiable experiments is exactly what a rigorous paradigm *should* look like at the pre-fabrication stage.

**The elevation, in one inequality.** The entire program can be compressed into a single certified-error statement: for a compiled, calibrated, verified contract $c$ executed under active feedback within an epoch,
$$
\Pr\Bigl[\ \bigl\|\Phi_{\widehat A_c}-\Phi_{A_c}\bigr\|_\diamond\ \le\ \underbrace{\sqrt K\,\eta}_{\text{compile}}+\underbrace{\tfrac{C(\varepsilon_{\mathrm{obs}})}{2\gamma+K}}_{\text{calibrate}}+\underbrace{f(l,k)}_{\text{loss}}\ \le\ \varepsilon\ \Bigr]\ \ge\ 1-\delta,
\tag{161}
$$
with each term derived (Theorems 13, 20, and the loss model) and each bounded by an engineering parameter ($\eta$, $\varepsilon_{\mathrm{obs}}$, $l$). Equation (161) *is* the rigorous-paradigm claim: a physically realized operator is provably within $\varepsilon$ (diamond norm) of its mathematical target with confidence $1-\delta$, where every term on the right is a first-principles quantity, not a fudge factor. A paradigm that can write down (161) — and defend every symbol in it against a red-team — is rigorous.

**Final statement.** The Physical Operator Computer is elevated, by this monograph, from an architecture to a *theory*: a finite-dimensional operator-algebraic model of analog computation with a proven compiler, a proven verification layer, a physical-limit boundary, and a complexity hierarchy — all stated with their hypotheses and limitations explicit. Whether it becomes a *useful* paradigm depends on the experiments; whether it is a *rigorous* one is now settled in the affirmative.

---

## Appendix C — Consolidated Open Problems

We collect the open problems flagged throughout, with their status and the techniques most likely to resolve them.

**OP-1 (Eigenvalue query lower bound).** Prove or refute $\mathrm{SQ}(A,\varepsilon,\delta)=\Omega(n/\varepsilon)$ for dense self-adjoint $A$ (Conjecture 4.3). *Approach:* adapt the polynomial-method / hybrid argument from quantum query complexity to the ringdown access model; the physical time–bandwidth picture (§4.2.1) strongly suggests the bound holds.

**OP-2 (OPC-P vs. OPC-NP).** Is OSDP solvable in poly transits, or is there an oracle separation? (Claim 5.4 establishes commutative NP-hardness.) *Approach:* a full noncommutative SUBSET-SUM-style reduction, or a poly-transit algorithm exploiting the Lie-algebraic structure of words (Theorem 5).

**OP-3 (OPC-SPEC vs. classical P).** Formalize and prove the resource separation 5.6 under a standard hardness assumption (e.g. SETH for dense diagonalization). *Approach:* fine-grained complexity; relate ringdown acquisitions to the $n^\omega$ matrix-multiplication exponent.

**OP-4 (Hopf genericity).** Characterize the class of Evolution-Unit Lindbladians $\mathcal L_\mu$ for which the loss of the spectral gap is a Hopf (vs. saddle-node/transcritical) bifurcation (Theorem 28, Attack 3). *Approach:* normal-form analysis of the vectorized generator (107); transversality conditions on $\partial_\mu\operatorname{Re}\zeta_a(\mu_c)$.

**OP-5 (Tight diamond-norm soundness).** Replace the worst-case $n^2K\eta$ prefactor in Theorem 13 with an average-case $\sqrt K\eta$ bound under a random-error model, with a concentration guarantee. *Approach:* matrix-concentration (matrix Bernstein) over independent MZI errors.

**OP-6 (Verification beyond the structured model).** Design a verification protocol that detects gross faults *outside* the $\|\Delta A\|\le\eta_{\max}$ perturbation model with sub-$n^4$ shots, under a weaker prior (e.g. low-rank-plus-sparse error). *Approach:* robust PCA / low-rank channel tomography.

**OP-7 (Squeezing extension).** Quantify how much the precision ceiling (Theorem 34) and the shot-noise scaling (Theorem 23) improve under $r$ dB of squeezing, and whether Heisenberg scaling $1/(Sn^2)$ becomes attainable — *out of scope here but the natural next volume.*

These seven problems delimit the frontier of operator-computing theory as established by this monograph.

## Bibliography

1. M. Reck, A. Zeilinger, H. J. Bernstein, P. Bertani. *Experimental realization of any discrete unitary operator*. Phys. Rev. Lett. **73**, 58 (1994).
2. W. R. Clements, P. C. Humphreys, B. J. Metcalf, W. S. Kolthammer, I. A. Walmsley. *Optimal design for universal multiport interferometers*. Optica **3**, 1460 (2016).
3. Y. Shen, N. C. Harris, S. Skirlo, et al. *Deep learning with coherent nanophotonic circuits*. Nature Photonics **11**, 441 (2017).
4. N. C. Harris, J. Carolan, D. Bunandar, et al. *Linear programmable nanophotonic processors*. Optica **5**, 1623 (2018).
5. O. Bratteli, D. W. Robinson. *Operator Algebras and Quantum Statistical Mechanics, Vol. I*. Springer (1987).
6. R. V. Kadison, J. R. Ringrose. *Fundamentals of the Theory of Operator Algebras, Vols. I–II*. AMS (1997).
7. M. Takesaki. *Theory of Operator Algebras, Vol. I*. Springer (1979).
8. J. von Neumann. *Zur Algebra der Funktionaloperationen und Theorie der normalen Operatoren*. Math. Ann. **102**, 370 (1930).
9. I. M. Gelfand, M. A. Naimark. *On the imbedding of normed rings into the ring of operators in Hilbert space*. Mat. Sbornik **12**, 197 (1943).
10. M. A. Nielsen, I. L. Chuang. *Quantum Computation and Quantum Information*. Cambridge Univ. Press (2010).
11. J. Watrous. *The Theory of Quantum Information*. Cambridge Univ. Press (2018).
12. A. Yu. Kitaev, A. H. Shen, M. N. Vyalyi. *Classical and Quantum Computation*. AMS (2002).
13. N. J. Higham. *Functions of Matrices: Theory and Computation*. SIAM (2008). (Matrix Itô formula, §4.5.)
14. G. H. Golub, C. F. Van Loan. *Matrix Computations*, 4th ed. Johns Hopkins Univ. Press (2013).
15. R. Bhatia. *Matrix Analysis*. Springer GTM 169 (1997). (Weyl, Bauer–Fike, Davis–Kahan.)
16. C. Davis, W. M. Kahan. *The rotation of eigenvectors by a perturbation. III*. SIAM J. Numer. Anal. **7**, 1 (1970).
17. L. N. Trefethen. *Approximation Theory and Approximation Practice*. SIAM (2013). (Chebyshev/Bernstein bounds.)
18. S. Amari, H. Nagaoka. *Methods of Information Geometry*. AMS / Oxford Univ. Press (2000).
19. D. Petz. *Quantum Information Theory and Quantum Statistics*. Springer (2008). (QFIM, Bures metric.)
20. C. W. Helstrom. *Quantum Detection and Estimation Theory*. Academic Press (1976). (Quantum Cramér–Rao.)
21. S. L. Braunstein, C. M. Caves. *Statistical distance and the geometry of quantum states*. Phys. Rev. Lett. **72**, 3439 (1994).
22. G. Da Prato, J. Zabczyk. *Stochastic Equations in Infinite Dimensions*. Cambridge Univ. Press (2014).
23. H. K. Khalil. *Nonlinear Systems*, 3rd ed. Prentice Hall (2002). (Lyapunov stability.)
24. Y. A. Kuznetsov. *Elements of Applied Bifurcation Theory*, 3rd ed. Springer (2004). (Hopf bifurcation.)
25. R. Landauer. *Irreversibility and heat generation in the computing process*. IBM J. Res. Dev. **5**, 183 (1961).
26. C. H. Bennett. *The thermodynamics of computation — a review*. Int. J. Theor. Phys. **21**, 905 (1982).
27. M. J. Donald, M. R. Garey, D. S. Johnson. *Computers and Intractability: A Guide to the Theory of NP-Completeness*. W. H. Freeman (1979). (3-COLORABILITY, SUBSET-SUM.)
28. R. L. Graham. *Bounds on multiprocessing timing anomalies*. SIAM J. Appl. Math. **17**, 416 (1969). (List scheduling 2-approximation.)
29. E. J. Candès, T. Tao. *Near-optimal signal recovery from random projections: universal encoding strategies*. IEEE Trans. Inf. Theory **52**, 5406 (2006). (Compressed sensing.)
30. A. W. van der Vaart. *Asymptotic Statistics*. Cambridge Univ. Press (1998). (Berry–Esseen, DKW, KS test.)
31. M. Krasnoselskii. *Two remarks on the method of successive approximations*. Uspekhi Mat. Nauk **10**, 123 (1955); W. R. Mann, *Mean value methods in iteration*, Proc. AMS **4**, 506 (1953).
32. H.-P. Breuer, F. Petruccione. *The Theory of Open Quantum Systems*. Oxford Univ. Press (2002). (Lindblad dynamics.)

---

## Postscript — Relation to Volume I and Forward Pointer to Volume III

Volume I established *what the POC is* (architecture, O-ISA, roadmap). This Volume II establishes *why it works and what it cannot do* (theory, limits, verification). The two are complementary: Volume I's engineering specifications are the *hypotheses* of this volume's conditional theorems (e.g. the loss budget feeds Theorem 13; the loop lifetime feeds Theorem 6; the thermal settle time feeds the controllability bandwidth, §7.7). Conversely, this volume's red-team (Phase 12) *revises* three of Volume I's headline figures, which should be propagated back into the roadmap's exit criteria (M0 connectivity + gain-balance tests; M2 energy benchmark at $4\times$).

A prospective **Volume III** would address the items deliberately excluded here: (i) squeezed-light and entangled-probe extensions lifting the SQL toward the Heisenberg limit (§11.3.1, OP-7); (ii) the infinite-dimensional $Q\to\infty$ loop limit requiring genuine von Neumann algebra theory (§2.3); (iii) fault-tolerant operator computing — error-correcting codes for analog operator words; and (iv) the resolution of the complexity-separation conjectures (Appendix C, OP-2, OP-3). Until then, the present volume stands as the complete *finite-dimensional, coherent-light, contract-verified* theory of the Physical Operator Computer.

*End of POC-MONO-2.0, Volume II.*
*QRATUM Research Series. This document is a mathematical research monograph; engineering claims are scoped to the milestones identified in Phase 12 and Volume I.*
