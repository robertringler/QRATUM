# QQ-Equality Empirical Falsification Test for CIIR

**Status:** Design document (pre-implementation)
**Target:** Empirical contact between CIIR and external data via the Wang–Busemeyer (2013) QQ-equality on the Wang et al. (2014, *PNAS*) Gallup corpus.

---

## §1. Phenomenon Selection

**Choice:** Quantum cognition order effects / QQ-equality.

**Why this over alternatives:**

- **CHSH / LGI** require entangled or coherent quantum systems; CIIR is a classical control framework on $\mathfrak{D}(H_C)$, so they don't bind.
- **Ellsberg / Allais** are well-explained by prospect theory; CIIR adds no parameter-free prediction.
- **Contextuality-by-Default (CbD)** lacks a single sharp numerical knife-edge.
- **QQ-equality** is the *only* phenomenon where standard projector quantum cognition (PQC) makes a parameter-free knife-edge prediction ($q \equiv 0$). Any non-unitary inter-measurement dynamics — exactly what CIIR's projected gradient flow on $\mathfrak{D}(H_C)$ introduces — generically breaks it.

## §2. Discriminating Prediction

**Setup:** Single-qubit constraint Hilbert space $H_C$, with

- $\rho_0 = \tfrac{1}{2}(I + \vec{r}\cdot\vec{\sigma})$, $|\vec{r}|=r_0$
- $K = \sigma_x$ (CIIR generator)
- $O_A = \sigma_z$ (question A projector basis)
- $O_B = \cos\phi\,\sigma_z + \sin\phi\,\sigma_x$ (question B basis, angle $\phi$ from $O_A$)
- Inter-measurement evolution: projected gradient flow on $\mathfrak{D}$ for parameter time $D$ at rate $K_{\min}$

**Closed-form prediction:**
$$
q^{\text{CIIR}} \;=\; p(A=+, B=+) - p(B=+, A=+) \;=\; \tfrac{1}{2}\,(1 - e^{-K_{\min} D})\,\sin\phi\,\cos\theta
$$
where $\theta = \angle(\vec{r}, \hat{z})$.

**Comparison:**

| Model | $q$ |
|---|---|
| **PQC** (standard projector quantum cognition) | $0$ (parameter-free) |
| **Markov / Bayes-net** | depends on transition matrix; fit per-pair |
| **CIIR** | $\tfrac{1}{2}(1-e^{-K_{\min}D})\sin\phi\cos\theta$ |

**Key non-commutator:** $[K, O_B] = -2i\cos\phi\,\sigma_y$. Setting either $[K,O_A]=0$ or $[K,O_B]=0$ collapses $q^{\text{CIIR}}\to 0$, isolating the prediction to CIIR's non-commuting constraint structure (not generic loss-function flexibility).

**Worked numbers:** at $\theta=\phi=\pi/3$, $K_{\min}D = 1$:
$$q^{\text{CIIR}} = \tfrac{1}{2}(1 - e^{-1})(\sqrt{3}/2)(1/2) \approx 0.137$$
This is four orders of magnitude above the empirical bound from Wang 2014 ($|\bar q_{\text{obs}}| \lesssim 10^{-3}$), so existing data already constrains $K_{\min} D \gtrsim 6$.

## §3. Baselines

1. **PQC** — standard projector quantum cognition, parameter-free $q\equiv 0$.
2. **Classical Markov / Bayes-net** — per-pair fitted transition matrix, expected to match marginals but underfit the structure of $q$ across pairs.
3. **PQC + isotropic depolarization** (sanity foil) — preserves $q=0$ on average; rules out "any decoherence breaks PQC" as a confound.

## §4. Experimental Protocol

**Data:**

- **Primary corpus:** Wang et al. (2014, *PNAS*) — ~70 question pairs, $N \approx 10^5$ total respondents.
- **Extension:** 30 new pairs on Prolific, designed to maximize $|\sin\phi\cos\theta|$ (predicted high-effect regime). Independent recruitment; pre-registered.

**Primary endpoint:**
OLS regression of observed $q_i$ on $x_i := \sin\phi_i \cos\theta_i$ across pairs, with HC3 robust standard errors:
$$\hat q_i \;=\; \alpha + \beta\, x_i + \varepsilon_i$$
Pre-registered split-half (random by pair index, seed locked) to address $(\phi, \theta)$ identifiability.

**Significance:** $\alpha = 0.005$ (Benjamin et al. 2018).

**Power:** simulation-based;

- $\geq 0.999$ at the predicted central effect ($\beta_{\text{CIIR}} \approx 0.13$ at $K_{\min}D=1$)
- $\geq 0.9$ down to $\beta = 0.020$
At the predicted central effect, separation from PQC is $\geq 17\sigma$.

**Multiple comparisons:** Holm–Bonferroni across 4 secondary endpoints (per-cohort slope, $\phi$-only, $\theta$-only, residual structure).

**Compute:** $\leq 100$ CPU-hours total ($\sim 10^{12}$ FLOPs). Wall-clock $\leq 6$ months end-to-end including Prolific data collection.

## §5. Killshot Conditions

CIIR is **falsified** if both of the following hold on the locked split-half:

1. $|\hat\beta| < 0.010$ with 99.5% CI excluding $\beta_{\text{CIIR}} \geq 0.020$, AND
2. Posterior on $K_{\min}D$ places $\geq 99\%$ of mass at $K_{\min}D > 8$ (the regime where CIIR is observationally indistinguishable from PQC).

CIIR is **corroborated** if $\hat\beta$ is consistent with the predicted $\beta_{\text{CIIR}}(\phi,\theta)$ map within the locked prior on $K_{\min}D$ AND $BF_{10} > 30$ vs the PQC and Markov baselines jointly.

## §6. Bayes Factor & Pre-Registration

**Locked priors** (OSF, hash-committed before unblinding):

- $\lambda \sim \text{LogNormal}(0, 1)$ — projection rate
- $\mu \sim \text{LogNormal}(0, 1)$ — drift coefficient
- $\gamma \sim \text{HalfNormal}(0.1)$ — coupling
- $\eta \sim \text{HalfNormal}(0.1)$ — noise floor
- $K_{\min} \sim \text{LogNormal}(0, 0.5)$
- $D \sim \text{LogNormal}(0, 0.5)$
- $\langle K \rangle_\perp \sim \text{HalfNormal}(0.5)$

**Decision thresholds:** $BF_{10} > 30$ (corroborate CIIR) or $BF_{10} < 1/30$ (corroborate baseline).

**Estimator:** PyMC bridge sampling, dual-chain convergence diagnostics ($\hat R < 1.01$), 10k warmup + 10k draws per chain.

## §7. Adversarial Review

**Three threats identified:**

1. **Split-half identifiability** *(mitigable)* — $\phi$ and $\theta$ are jointly fitted from marginal frequencies; degeneracy in some pairs. **Mitigation:** restrict primary regression to pairs where the locked Wang-2014 fit gives Fisher-information-determinant $> \tau$ (threshold pre-registered).

2. **Observational equivalence to PQC + anisotropic Lindblad** *(partially fatal to uniqueness, not falsifiability)* — a PQC model with a fitted anisotropic Lindbladian can reproduce $q^{\text{CIIR}}$. CIIR's claim becomes "the *specific* tensor structure derived from $K=\sigma_x$ on a constraint manifold". A positive result would require a follow-up dimensional-analysis study to discriminate; a null result still falsifies CIIR.

3. **Prior falsification by existing data** *(acceptable)* — Wang 2014 already shows $|\bar q_{\text{obs}}| \lesssim 10^{-3}$, forcing $K_{\min}D \gtrsim 6$ in CIIR's prior. **Mitigation:** the 30 new max-effect Prolific pairs provide independent constraint; the regression slope (not the mean $|q|$) is the formal test, and slope is preserved across $K_{\min}D$ regimes up to amplitude.

## §8. Implementation Map (QRATUM)

New modules under `quasim/ciir/cognition/`:

- `qq_equality.py` — closed-form $q^{\text{CIIR}}(\rho, \phi, \theta, K_{\min}, D)$; reuses existing `density_matrix.py` and projection channel utilities from `multi_qubit/`.
- `baselines.py` — `pqc_q()` (returns 0 with proper marginal handling) and `markov_q(transition_matrix)`.
- `regression.py` — split-half OLS with HC3 SE; deterministic seeded split.
- `bayes_factor.py` — PyMC bridge sampling wrapper with locked-prior loader.

Tests: `tests/test_qq_equality.py` (≥25 tests):

- closed-form recovers Wang–Busemeyer (2013) Theorem 1 in $\alpha\to 1$ unitary limit
- $[K,O_A]=0$ and $[K,O_B]=0$ sub-cases give $q=0$
- numerical stability of the projection channel $\Pi$ near rank-deficient $\rho$
- regression recovers planted $\beta$ on simulated data within 99.5% CI

Runner: `run_qq_falsification.py` (mirrors structure of existing `run_falsification.py`).

Manuscript: `manuscripts/ciir_monograph/chapters/ch23_qq_falsification.tex` with **Theorem 23.1** (CIIR QQ-equality formula) and registration of secondary endpoints in the gap registry.

OSF pre-registration: locked-prior hash committed before unblinding; data collection proceeds only after registration receipt.

## §9. Calibrated Confidence

| Quantity | Estimate |
|---|---|
| $P(\text{CIIR survives the test})$ | $0.18$ |
| $P(\text{study is executable as designed})$ | $0.80$ |
| $P(\text{result convincing to PRX/Cognition referees})$ | $0.35$ |

Joint $P(\text{published positive result}) \approx 0.05$. The expected information value is dominated by the null branch (clean falsification), which is the design intent.

---

## Implementation checklist

- [ ] `quasim/ciir/cognition/qq_equality.py`
- [ ] `quasim/ciir/cognition/baselines.py`
- [ ] `quasim/ciir/cognition/regression.py`
- [ ] `quasim/ciir/cognition/bayes_factor.py`
- [ ] `tests/test_qq_equality.py` (≥25 tests)
- [ ] `run_qq_falsification.py`
- [ ] `manuscripts/ciir_monograph/chapters/ch23_qq_falsification.tex` (Theorem 23.1)
- [ ] OSF pre-registration with hash-locked priors
- [ ] 30-pair Prolific recruitment in the predicted high-$|\sin\phi\cos\theta|$ regime
