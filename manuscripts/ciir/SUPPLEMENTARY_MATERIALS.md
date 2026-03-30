# Supplementary Materials

## CIIR Formal System: Constraint-Induced Interface Realism

**Author:** Robert Ringler (Independent Researcher)
**Date:** March 22, 2026
**Manuscript:** "Constraint-Induced Interface Realism: A Complete Axiomatic Framework
for Bounded Cognitive Systems and Observable Reality Reconstruction"

---

## S1. Extended Mathematical Derivations

### S1.1 GNS Construction for CIIR Observable Algebra

The GNS (Gelfand–Naimark–Segal) construction is central to the CIIR Representation
Theorem (Theorem 1). We give the full construction here.

**Setup.** Let $\mathcal{A}(\mathcal{H_C})$ be the observable algebra (a
$W^*$-algebra) and $\omega : \mathcal{A} \to \mathbb{C}$ the cognitive state
functional defined by $\omega(A) = \mathbb{E}[A]$.

**Construction.**

1. Define the inner product space $(\mathcal{A}, \langle \cdot, \cdot \rangle_\omega)$
   where $\langle A, B \rangle_\omega = \omega(A^* B)$.
2. The null space $\mathcal{N} = \{A : \omega(A^*A) = 0\}$ is a left ideal.
3. The quotient $\mathcal{A}/\mathcal{N}$ with induced inner product is a
   pre-Hilbert space; its completion is $\mathcal{H_C}$.
4. The cyclic vector $\Omega = [I] \in \mathcal{H_C}$ satisfies
   $\omega(A) = \langle\Omega|\pi(A)|\Omega\rangle$ for all $A$.

**Uniqueness.** Two GNS triples $(\mathcal{H}_1, \pi_1, \Omega_1)$ and
$(\mathcal{H}_2, \pi_2, \Omega_2)$ are unitarily equivalent: there exists a
unitary $U : \mathcal{H}_1 \to \mathcal{H}_2$ with $U\pi_1(A) = \pi_2(A)U$
for all $A$.

### S1.2 Lindblad Operator Derivation from Constraint Spectral Factorisation

Given the constraint operator $\hat{C}$ with spectral decomposition
$\hat{C} = \sum_k c_k |k\rangle\langle k|$, the Lindblad operators are derived as:

$$L_k = \sqrt{\gamma_k} |k\rangle\langle k+1|, \quad
L_k^\dagger = \sqrt{\gamma_k} |k+1\rangle\langle k|,$$

where $\gamma_k > 0$ are the constraint-induced transition rates satisfying the
detailed balance condition:

$$\frac{\gamma_k^+}{\gamma_k^-} = e^{-(c_{k+1} - c_k)/k_B T_\mathrm{eff}}.$$

This ensures that the equilibrium state is the Gibbs state
$\rho_* \propto \exp(-\hat{H}_\mathcal{M}/k_B T_\mathrm{eff})$, consistent
with Proposition 2 (Existence of Equilibrium).

### S1.3 Proof of Constraint Capacity Theorem (Full)

The proof of Theorem 10 (Constraint Capacity Theorem) follows the Holevo
capacity formula. Define the Holevo quantity for ensemble
$\mathcal{E} = \{p_i, \sigma_i\}$:

$$\chi(\mathcal{E}) = S\!\left(\sum_i p_i \Phi(\sigma_i)\right) - \sum_i p_i S(\Phi(\sigma_i)).$$

The classical capacity $C_\Phi = \sup_{\mathcal{E}} \chi(\mathcal{E})$.
Using $S(\rho) \leq \log\dim\mathcal{H_C}$ and the concavity of entropy:

$$\chi(\mathcal{E}) \leq \log\dim\mathcal{H_C} - \min_\sigma S(\Phi(\sigma)).$$

By Axiom 4, $\dim\mathcal{H_C} \leq \exp(\kappa_\mathrm{max} V)$, giving
$C_\Phi \leq \kappa_\mathrm{max} V + \log\dim\mathcal{H_C}$ as stated.

---

## S2. Numerical Simulation Protocols

### S2.1 CIIR Master Equation Simulation

The CIIR master equation (Eq. 3) is solved numerically using:

**Algorithm:** Fourth-order Runge-Kutta integration with adaptive step size.

**Parameters for qubit model:**
- Hilbert space dimension: $n = 2$
- Constraint frequency: $\omega = 1.0$ (normalised units)
- Decoherence rate: $\gamma \in \{0.5, 1.0, 2.0\}$
- Time range: $t \in [0, 10/\gamma]$
- Step size: $\Delta t = 0.001/\gamma$

**Initial states tested:**
- Pure states on the Bloch sphere equator: $|\psi_0\rangle = (|0\rangle + e^{i\phi}|1\rangle)/\sqrt{2}$, $\phi \in \{0, \pi/4, \pi/2, \pi\}$
- Mixed states: $\rho_0 = (1-p)|0\rangle\langle 0| + p|1\rangle\langle 1|$, $p \in \{0.1, 0.3, 0.5\}$
- Maximally mixed state: $\rho_0 = \mathbf{1}/2$

**Validation:** All simulations verified against analytical solutions where
available (qubit two-level system has closed-form solution in Appendix C of
manuscript).

### S2.2 Decoherence Rate Measurement Protocol

To test Prediction P2 (Eq. P2) experimentally:

1. Prepare two cognitive systems in states $\psi_1, \psi_2$ (non-orthogonal)
2. Form superposition $\psi_+ = (\psi_1 + \psi_2)/\sqrt{2}$
3. Measure observable $\hat{O}$ at times $t \in \{0, 0.1, 0.2, \ldots, 5.0\}$ (units of $1/\Gamma$)
4. Extract $|\mathrm{Re}\langle\psi_1|\hat{O}|\psi_2\rangle(t)|$ by quantum process tomography
5. Fit to $A e^{-(\Gamma_1+\Gamma_2)t/2}$; report $\chi^2$ per degree of freedom
6. Test: Is the exponential decay law satisfied within $3\sigma$?

**Falsification criterion:** Any consistent deviation from exponential decay,
or $\Gamma$ estimated from fit inconsistent with independently measured
$\Gamma_1, \Gamma_2$, would falsify P2.

### S2.3 Constraint Capacity Saturation Protocol

To test Prediction P3 (Eq. P3):

1. Define a random process $R$ with known entropy $H(R)$
2. Train a cognitive system (e.g. neural network as operationalisation) on
   samples from $R$ for $T$ steps
3. Measure mutual information $I(R;C)$ using $k$-nearest-neighbour estimator
   at each checkpoint $T \in \{10^2, 10^3, \ldots, 10^6\}$
4. Fit saturation curve $I(R;C)(T) = C_\Phi (1 - e^{-T/T_0})$
5. Compare $C_\Phi^\mathrm{fit}$ against theoretical prediction
   $C_\Phi^\mathrm{theory} = \kappa_\mathrm{max} V + O(1)$

---

## S3. Alternative Interpretations and Robustness

### S3.1 Non-Markovian Extensions

The CIIR master equation (Eq. 3) assumes a Markov (memoryless) constraint
environment. When constraint environment correlations are non-negligible, the
Nakajima–Zwanzig equation applies:

$$\frac{d\rho(t)}{dt} = \mathcal{L}_0 \rho(t) + \int_0^t \mathcal{K}(t-s)\rho(s)\,ds,$$

where $\mathcal{K}(t-s)$ is the memory kernel. CIIR predictions P1–P3 hold
in the Markov limit $\mathcal{K} \to \delta(t)\mathcal{K}_0$ and are modified
by memory corrections of order $\int_0^\infty |\mathcal{K}(s)|\,ds$.

### S3.2 Classical Noise Model as Alternative

One competing explanation for Prediction P2 is classical stochastic noise:
if cognitive states are classical probability distributions perturbed by
Gaussian noise with variance $\sigma^2$, the coherence decay is also
approximately exponential with rate $\Gamma \approx \sigma^2/2$. CIIR predicts:

$$\Gamma_\mathrm{CIIR} = \sum_k \gamma_k \|L_k\|^2,$$

while the classical noise model predicts:

$$\Gamma_\mathrm{classical} = \sigma^2/2.$$

These can be distinguished by measuring the full process matrix via quantum
process tomography: CIIR predicts a completely positive process matrix, while
classical noise admits a non-completely-positive effective description.

### S3.3 Finite-N Effects

For finite-dimensional cognitive state spaces ($\dim\mathcal{H_C} = N < \infty$),
recurrence phenomena (Poincaré recurrence) may cause apparent violations of
Theorem 5 (Entropy Non-Decrease) on timescales $T_\mathrm{rec} \sim N!$.
This does not falsify Theorem 5, which applies only in the infinite-time limit
or for primitive dissipators on infinite-dimensional spaces. For practical
purposes, $T_\mathrm{rec} \gg T_\mathrm{exp}$ (experimental timescale) for
$N \geq 10$.

---

## S4. Reproducibility Statement

All numerical simulations described in S2 are reproducible using:

- Python 3.10+
- NumPy $\geq$ 1.24
- SciPy $\geq$ 1.10
- QuTiP $\geq$ 4.7 (quantum toolbox in Python)

Code will be deposited at the GitHub repository:
[https://github.com/robertringler/QRATUM](https://github.com/robertringler/QRATUM)
under `manuscripts/ciir/code/`.

All derivations are analytic and self-contained within the manuscript.

---

## S5. Notation Reference

| Symbol | Definition | First used |
|--------|-----------|-----------|
| $\mathcal{R}$ | Reality space | Def. 1 |
| $\mathcal{M}$ | Constraint manifold | Def. 2 |
| $\mathcal{H_C}$ | Cognitive state space | Def. 3 |
| $\Phi_X$ | Interface map | Def. 4 |
| $\mathcal{A}(\mathcal{H_C})$ | Observable algebra | Def. 5 |
| $\mathcal{H}_\mathcal{M}$ | Constraint-image subspace | Def. 6 |
| $\hat{P}_\mathcal{M}$ | Projection onto $\mathcal{H}_\mathcal{M}$ | Def. 6 |
| $\mathfrak{D}(\mathcal{H_C})$ | Mixed cognitive state space | Def. 3 |
| $\kappa(m)$ | Constraint curvature at $m$ | Def. 2 |
| $\hat{H}_\mathcal{M}$ | Constraint Hamiltonian | Def. 7 |
| $\mathcal{D}_\mathcal{M}$ | Constraint dissipator | Def. 8 |
| $L_k$ | Lindblad collapse operators | Def. 8 |
| $\gamma_k$ | Decoherence rates | Def. 8 |
| $G_\mathcal{M}$ | Constraint symmetry group | Def. 10 |
| $S_\mathcal{M}(\rho)$ | Constraint entropy | Def. 9 |
| $\mathcal{I}$ | Empirical interpretation map | Def. 11 |
| $C_\Phi$ | Constraint channel capacity | Theorem 10 |
| $D_B(\rho_1,\rho_2)$ | Bures distance | Theorem 3 |
| $\hbar_\mathrm{eff}$ | Effective constraint action scale | Prop. 1 |

---

*End of Supplementary Materials*
