# Figure and Table Index

## CIIR Formal System Manuscript

**Author:** Robert Ringler (Independent Researcher)
**Date:** March 22, 2026
**Document:** Constraint-Induced Interface Realism: A Complete Axiomatic Framework

---

## FIGURES

### Figure 1: CIIR System Architecture Diagram

**File:** `fig01_ciir_architecture.pdf`
**Format:** PDF, vector graphics
**Resolution:** N/A (vector)
**Content:**

- Block diagram showing the six components of $\mathcal{T}_\mathrm{CIIR}$
- Arrows indicating functional relationships: $\mathcal{R} \to \mathcal{M} \to \mathcal{H_C} \to \mathcal{A}$
- Interface map $\Phi_X$ highlighted with bidirectional arrow
- Interpretation map $\mathcal{I}$ to observable outputs $\mathbb{R}^n$

**Caption:** Schematic of the CIIR formal system. The reality space $\mathcal{R}$
(top left) is accessed through the constraint manifold $\mathcal{M}$ via
the interface map $\Phi_X$, producing cognitive states in $\mathcal{H_C}$
from which observables in $\mathcal{A}(\mathcal{H_C})$ are extracted and
mapped to empirical outputs $\mathbb{R}^n$ by the interpretation map $\mathcal{I}$.

---

### Figure 2: Constraint Manifold Curvature and Hilbert-Space Dimension

**File:** `fig02_curvature_dimension.pdf`
**Format:** PDF, matplotlib
**Resolution:** N/A (vector)
**Content:**

- X-axis: $\kappa_\mathrm{max} \cdot \mathrm{vol}(\mathcal{M})$ (dimensionless)
- Y-axis: $\log_2 \dim \mathcal{H_C}$ (bits)
- Curve: $\dim \mathcal{H_C} = \exp(\kappa_\mathrm{max} V)$ from Axiom 4
- Shaded region: physically realizable systems

**Caption:** Hilbert-space dimension as a function of the constraint-curvature–volume
product $\kappa_\mathrm{max} V$. The exponential growth (solid curve) is the
maximum permitted by Axiom~4; physically realised systems lie at or below this
curve (shaded).

---

### Figure 3: Qubit Bloch-Sphere Interface Map

**File:** `fig03_bloch_sphere.pdf`
**Format:** PDF, 3D matplotlib
**Resolution:** N/A (vector)
**Content:**

- Standard Bloch sphere representation of $\mathbb{C}^2$
- Interface map $\Phi_X(\theta)$ shown as a meridional family of states
- Trajectory of $\rho(t)$ under CIIR master equation indicated by arrows
- Equilibrium state $\rho_* = \frac{1}{2}\mathbf{1}$ marked at centre

**Caption:** Bloch sphere representation of the qubit CIIR model (Example 1).
The interface map $\Phi_X(\theta)$ maps the constraint manifold $S^1$ to the
equatorial circle of diagonal states. Under the master equation, all initial
states relax to $\rho_* = \frac{1}{2}\mathbf{1}$ (centre) at rate $\gamma$.

---

### Figure 4: Decoherence Decay of Off-Diagonal Elements

**File:** `fig04_decoherence.pdf`
**Format:** PDF, matplotlib
**Resolution:** N/A (vector)
**Content:**

- X-axis: time $t$ (units of $1/\gamma$)
- Y-axis: $|\rho_{01}(t)|$ (normalised coherence)
- Curves: analytical prediction $|\rho_{01}(0)|e^{-\gamma t/2}$ vs. numerical integration
- Multiple curves for $\gamma \in \{0.5, 1.0, 2.0\}$

**Caption:** Exponential decay of off-diagonal coherence $|\rho_{01}(t)|$
in the qubit model, consistent with Theorem 4 (Decoherence Monotonicity).
Analytical prediction (dashed) and numerical GKSL integration (solid) agree
to numerical precision.

---

### Figure 5: Constraint Entropy Evolution

**File:** `fig05_entropy_evolution.pdf`
**Format:** PDF, matplotlib
**Resolution:** N/A (vector)
**Content:**

- X-axis: time $t$ (units of $1/\gamma$)
- Y-axis: von Neumann entropy $S(\rho(t))$ (bits)
- Monotone increase from pure state ($S=0$) to maximum ($S=\log 2$)
- Horizontal dashed line at equilibrium

**Caption:** Monotone non-decrease of von Neumann entropy $S(\rho(t))$
under CIIR dynamics (Theorem~5). The system evolves from a pure initial state
to the maximally mixed equilibrium $\rho_*$; entropy is non-decreasing
throughout, consistent with Spohn's theorem.

---

### Figure 6: Interference Term vs. Decoherence Rate

**File:** `fig06_interference_scaling.pdf`
**Format:** PDF, matplotlib
**Resolution:** N/A (vector)
**Content:**

- X-axis: decoherence rate $\Gamma$ (units)
- Y-axis: interference term $|\mathrm{Re}\langle\psi_1|\hat{O}|\psi_2\rangle(t)|$
- Family of curves for $t \in \{1, 2, 4, 8\}$ illustrating Prediction P2
- Exponential decay verified against analytical formula

**Caption:** Verification of Prediction P2 (Eq.~P2): decay of the cognitive
interference term as a function of decoherence rate $\Gamma$ for various
observation times $t$. The exponential scaling confirms the CIIR prediction.

---

### Figure 7: Constraint Capacity as a Function of Curvature-Volume

**File:** `fig07_capacity_curvature.pdf`
**Format:** PDF, matplotlib
**Resolution:** N/A (vector)
**Content:**

- X-axis: $\kappa_\mathrm{max} V_\mathcal{M}$ (nats)
- Y-axis: channel capacity $C_\Phi$ (nats)
- Theoretical bound (Theorem~11) shown as solid line
- Points from qubit and harmonic oscillator models

**Caption:** Constraint channel capacity $C_\Phi$ as a function of the
curvature-volume product (Theorem~11). The theoretical upper bound
$\kappa_\mathrm{max} V + \log\dim\mathcal{H_C}$ (solid) is saturated in
the limit of zero decoherence ($\mathcal{D}_\mathcal{M} = 0$).

---

### Figure 8: Classical--Quantum--CIIR Embedding Diagram

**File:** `fig08_embedding.pdf`
**Format:** PDF, vector
**Resolution:** N/A (vector)
**Content:**

- Nested set diagram: CIIR ⊃ Quantum Cognition ⊃ Classical Probability
- Annotations showing which axioms are relaxed at each embedding level
- Arrow from Bayesian inference to classical probability corner

**Caption:** Embedding structure of CIIR relative to special cases
(Theorems~12--14). Classical Kolmogorov probability arises when the constraint
manifold is flat ($\kappa = 0$); quantum cognition corresponds to the special
case $\Phi_X = \mathrm{id}$; Bayesian inference is the Abelian measurement
update.

---

## TABLES

### Table 1: Axiom Summary

**Columns:** Axiom number | Name | Formal statement reference | Justification | Independence demonstrated by
**Rows:** 6 rows (one per axiom)

| # | Name | Ref. | Justification | Independence Counter-Model |
|---|------|------|---------------|---------------------------|
| A1 | Ontological Priority | Def. 1 | Realist commitment | Empty reality space |
| A2 | Constraint Boundedness | Def. 2 | Physical boundedness | Negatively curved manifold |
| A3 | Interface Completeness | Def. 4 | Uniqueness of representation | Zero interface map |
| A4 | Hilbert Representation | Def. 3 | Quantum framework | Non-separable Hilbert space |
| A5 | Measurement Collapse | Def. (Sect. 7) | Born rule | Nonlinear positive map |
| A6 | Constraint Composition | Eq. (1) | Multi-agent systems | Direct sum composition |

---

### Table 2: Theorem Summary

**Columns:** Theorem | Name | Statement | Proof method | Applications

| # | Name | Key Result | Method | Use |
|---|------|-----------|--------|-----|
| T1 | CIIR Representation | Functorial Hilbert rep. | GNS construction | Foundations |
| T2 | Axiom Independence | No axiom is redundant | Counter-models | Minimality |
| T3 | CIIR Master Equation | GKSL dynamics | Lindblad theory | Evolution |
| T4 | Decoherence Monotonicity | Off-diag. exponential decay | Gronwall lemma | Predictions |
| T5 | Entropy Non-Decrease | $dS/dt \geq 0$ | Spohn's theorem | Irreversibility |
| T6 | CIIR Nöther | $Q$ conserved under $G_\mathcal{M}$ | Trace cyclicity | Conservation |
| T7 | Interface Non-Invertibility | $\Phi_X$ non-injective | Cardinality | Information loss |
| T8 | Cognitive Interference | Non-zero cross term | Direct calculation | QC effects |
| T9 | Constraint Distortion | $d_\mathcal{H} \leq C \cdot d_\mathcal{R}$ | Toponogov | Measurement bounds |
| T10 | Constraint Capacity | $C_\Phi \leq \kappa V + \log\dim$ | Holevo bound | Learning bounds |

---

### Table 3: Falsifiable Predictions Summary

| Prediction | Observable | Functional Form | Falsification Condition |
|-----------|-----------|-----------------|------------------------|
| P1 | Decision latency $\tau$ | $\tau = \tau_0 e^{\alpha\kappa V} + \tau_1$ | Sub-exponential growth in $\kappa V$ |
| P2 | Interference decay | $\|$off-diag$\|(t) = \|$off-diag$\|(0) e^{-\Gamma t/2}$ | Super-exponential or non-monotone |
| P3 | Mutual information saturation | $I(R;C) \to \kappa V + O(1)$ | Saturation above or failure to saturate |

---

### Table 4: Model Verification

| Model | $\dim\mathcal{H_C}$ | $\mathcal{M}$ | Axioms satisfied | Notes |
|-------|---------------------|----------------|------------------|-------|
| Qubit | 2 | $S^1$ | A1–A6 | Explicit Lindblad |
| $n$-level | $n$ | General compact | A1–A6 | Theorem 15 |
| Harmonic oscillator | $\infty$ (countable) | $\mathbb{R}_+$ | A1–A6 | Thermal equilibrium |

---

### Table 5: Comparative Embedding Summary

| Framework | CIIR Conditions | Section Reference |
|----------|----------------|-------------------|
| Kolmogorov probability | $\kappa = 0$ (flat $\mathcal{M}$) | Theorem 12 |
| Bayesian inference | Abelian + flat + Lüders rule | Theorem 13 |
| Quantum cognition (BB) | $\Phi_X = \mathrm{id}$, $\hat{H}_\mathcal{M} = 0$ | Theorem 14 |

---

## SUPPLEMENTARY FIGURES

### Supplementary Figure S1: Spectral Decomposition of $\hat{H}_\mathcal{M}$

Eigenvalue spectrum of the constraint Hamiltonian for the $n=4$ (ququart) model,
showing discrete spectrum bounded by Axiom~4 constraint.

### Supplementary Figure S2: Phase Portrait of CIIR Dynamics

Phase portrait in the space of density matrices (parameterised by Bloch vector
coordinates) showing convergence to equilibrium under various initial conditions.

### Supplementary Figure S3: Mutual Information Saturation Curve

Numerical simulation of Prediction P3: mutual information $I(R;C)$ as a function
of training iterations for $\kappa V \in \{1, 2, 4, 8\}$ nats.

---

## PRODUCTION SPECIFICATIONS

- All figures: PDF format, vector graphics where possible
- Minimum raster resolution: 600 DPI
- Font: Computer Modern (consistent with LaTeX manuscript)
- Colour scheme: WCAG 2.1 AA accessible
- Figure dimensions: Two-column width = 86 mm; single column = 174 mm

---

*End of Figure and Table Index*
